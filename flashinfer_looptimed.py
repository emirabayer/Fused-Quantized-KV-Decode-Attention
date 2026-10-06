# Reproduces the methodology behind benchmarks/flashinfer_looptimed.json: the corrected, byte-fair
# comparison of the GQA-aware split-K kernel against FlashInfer single_decode.
#
# Two things this fixes over the earlier comparison:
#  - Byte fairness: both sides read the same compact 8-head GQA KV cache. The earlier version timed
#    our kernel on KV expanded to 32 heads while FlashInfer read 8 heads, a 4x byte difference.
#  - Timing: each measurement loops the call 2000 times so the per-call launch and dispatch overhead
#    is present but identical on both sides. CUDA graph capture failed on the custom extension, so
#    loop timing both sides the same way was the fallback.
#
# The scratch buffers are preallocated once, outside the timed loop. GB/s is kv_bytes over time, where
# kv_bytes = 2 * N * Hkv * D * 2 (K and V, fp16). Short-N GB/s is launch-floor-limited and partly
# L2-resident, so it is not sustained DRAM bandwidth.
#
# The committed JSON holds the recorded A100 run used in the README. This script regenerates it.
#   pip install flashinfer-python
#   python flashinfer_looptimed.py

import json
import torch, torch.nn.functional as F
from torch.utils.cpp_extension import load
import flashinfer

dev = "cuda"
torch.manual_seed(0)
PEAK_GBS = 1555.0
Hq, Hkv = 32, 8
G = Hq // Hkv

mod = load(name="decode_attn_gqa_sk_fast", sources=["kernels/decode_attn_gqa_sk_fast.cu"],
           extra_cuda_cflags=["-O3"], verbose=False)


def loop_time(fn, iters=2000, warmup=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3   # microseconds per call


def launch_floor_us():
    # smallest possible launch of our extension, as an estimate of the per-call floor
    q = torch.randn(Hq, 64, dtype=torch.float16, device=dev).contiguous()
    K = torch.randn(Hkv, 1, 64, dtype=torch.float16, device=dev).contiguous()
    V = torch.randn(Hkv, 1, 64, dtype=torch.float16, device=dev).contiguous()
    mp = torch.empty(Hq, 1, dtype=torch.float32, device=dev)
    lp = torch.empty(Hq, 1, dtype=torch.float32, device=dev)
    op = torch.empty(Hq, 1, 64, dtype=torch.float32, device=dev)
    o = torch.empty(Hq, 64, dtype=torch.float16, device=dev)
    return loop_time(lambda: mod.gqa_sk_fast(q, K, V, mp, lp, op, o, 1, 2, 4))


def compare(D, N):
    q = torch.randn(Hq, D, dtype=torch.float16, device=dev).contiguous()
    K = torch.randn(Hkv, N, D, dtype=torch.float16, device=dev).contiguous()   # compact KV
    V = torch.randn(Hkv, N, D, dtype=torch.float16, device=dev).contiguous()
    # FlashInfer reads [N, Hkv, D]
    k_fi = K.transpose(0, 1).contiguous()
    v_fi = V.transpose(0, 1).contiguous()

    kE = K.repeat_interleave(G, 0)
    vE = V.repeat_interleave(G, 0)
    ref = F.scaled_dot_product_attention(
        q.view(Hq, 1, 1, D), kE.view(Hq, 1, N, D), vE.view(Hq, 1, N, D)).view(Hq, D)

    best = (1e9, None, None, None)
    for S in [16, 32, 64, 128]:
        mp = torch.empty(Hq, S, dtype=torch.float32, device=dev)
        lp = torch.empty(Hq, S, dtype=torch.float32, device=dev)
        op = torch.empty(Hq, S, D, dtype=torch.float32, device=dev)
        o = torch.empty(Hq, D, dtype=torch.float16, device=dev)
        for W in [4, 8]:
            for CW in [4, 8]:
                mod.gqa_sk_fast(q, K, V, mp, lp, op, o, S, W, CW)
                if not torch.allclose(o, ref, atol=2e-2):
                    continue
                t = loop_time(lambda: mod.gqa_sk_fast(q, K, V, mp, lp, op, o, S, W, CW))
                if t < best[0]:
                    best = (t, S, W, CW)
    ours_us, S, W, CW = best
    fi_us = loop_time(lambda: flashinfer.single_decode_with_kv_cache(q, k_fi, v_fi))

    kv_bytes = 2 * N * Hkv * D * 2
    gbs = lambda us: round(kv_bytes / (us * 1e-6) / 1e9)
    print(f"D={D:3d} N={N:6d} | ours {ours_us:6.2f}us {gbs(ours_us):4d} GB/s | "
          f"fi {fi_us:6.2f}us {gbs(fi_us):4d} GB/s | ratio {fi_us/ours_us:.2f}x  S{S}W{W}C{CW}")
    return dict(D=D, N=N, kv_bytes=kv_bytes, ours_us=round(ours_us, 2), ours_GBs=gbs(ours_us),
                fi_us=round(fi_us, 2), fi_GBs=gbs(fi_us), ratio=round(fi_us / ours_us, 2),
                best_cfg=f"S{S}W{W}C{CW}", kv_fits_L2=kv_bytes <= 40 * 2**20)


floor = launch_floor_us()
rows = [compare(D, N) for D in [64, 128] for N in [512, 1024, 2048, 4096, 8192, 16384]]
out = dict(
    note="Loop-timed (2000 calls/measurement), launch/API overhead present but identical on both "
         "sides. Equal-byte compact 8-head GQA KV. Short-context wins come from lower dispatch "
         "overhead while the kernel itself is no faster. On sustained bandwidth (N>=8K) FlashInfer "
         "wins. fp16 only.",
    launch_floor_us=round(floor, 2), peak_GBs=PEAK_GBS, rows=rows)
json.dump(out, open("benchmarks/flashinfer_looptimed.json", "w"), indent=2)
print(f"\nlaunch floor ~{floor:.2f}us. saved benchmarks/flashinfer_looptimed.json")
