# Reproduces the headline split-K vs FlashInfer comparison in benchmarks/flashinfer_compare.json.
# fp16 only. For each (head_dim, N) it finds the best split count S by measurement, checks the
# output against SDPA, then times split-K against FlashInfer single_decode and writes the JSON.
#
# The scratch buffers (m_part, l_part, o_part) are allocated inside decode_attn_splitk, so they
# are included in the timed call. That penalizes split-K, so the real speedups are a little higher
# than what this prints.
#
# Needs an A100 (or similar), PyTorch with CUDA, nvcc, and flashinfer-python.
#   pip install flashinfer-python
#   python flashinfer_compare.py

import os, json
import torch, torch.nn.functional as F
from torch.utils.cpp_extension import load
import flashinfer

dev = "cuda"
torch.manual_seed(0)

mod = load(name="decode_attn_splitk", sources=["kernels/decode_attn_splitk.cu"],
           extra_cuda_cflags=["-O3"], verbose=False)


def bench(fn, iters=100, warmup=30):
    warm = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
    for _ in range(30):
        _ = warm @ warm
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def best_s_compare(N, D, H=32, Hkv=8, W=8):
    G = H // Hkv
    q_h = torch.randn(H, D, dtype=torch.float16, device=dev)
    k_kv = torch.randn(N, Hkv, D, dtype=torch.float16, device=dev)   # FlashInfer layout
    v_kv = torch.randn(N, Hkv, D, dtype=torch.float16, device=dev)
    kE = k_kv.transpose(0, 1).repeat_interleave(G, 0).contiguous()   # [H,N,D] for our kernel
    vE = v_kv.transpose(0, 1).repeat_interleave(G, 0).contiguous()
    qc = q_h.contiguous()

    o_ref = F.scaled_dot_product_attention(
        q_h.view(H, 1, 1, D), kE.view(H, 1, N, D), vE.view(H, 1, N, D)).view(H, D)

    best = (1e9, None)
    for S in [1, 4, 8, 16, 32, 64, 128]:
        o = mod.decode_attn_splitk(qc, kE, vE, S, W)
        if not torch.allclose(o.float(), o_ref.float(), atol=2e-2):
            continue
        t = bench(lambda: mod.decode_attn_splitk(qc, kE, vE, S, W))
        if t < best[0]:
            best = (t, S)
    t_sk, S_best = best
    t_fi = bench(lambda: flashinfer.single_decode_with_kv_cache(q_h, k_kv, v_kv))
    print(f"D={D:3d} N={N:6d} | split-K(S={S_best:3d}) {t_sk:.4f} ms | "
          f"flashinfer {t_fi:.4f} ms | sk/fi {t_fi / t_sk:.2f}x")
    return dict(D=D, N=N, best_S=S_best, splitk_ms=t_sk, flashinfer_ms=t_fi, ratio=t_fi / t_sk)


rows = []
for D in [64, 128]:
    for N in [512, 1024, 2048, 4096, 8192, 16384]:
        rows.append(best_s_compare(N, D))

os.makedirs("benchmarks", exist_ok=True)
json.dump(
    dict(note="split-K decode vs FlashInfer single_decode; each shape at measured-best S; "
              "scratch alloc inside timed call (penalizes split-K)", rows=rows),
    open("benchmarks/flashinfer_compare.json", "w"), indent=2)
print("saved benchmarks/flashinfer_compare.json")
