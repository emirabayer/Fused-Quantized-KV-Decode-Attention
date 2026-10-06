# Runs one GQA-aware split-K launch at the long-context shape (D=128, N=16384) so Nsight Compute can
# profile the partial kernel and the parallel combine kernel.
#
# At this shape the partial kernel shows nothing saturated: about 52% SM throughput, 44% memory
# throughput, 30% DRAM throughput, and 80% occupancy. So the long-context gap to FlashInfer is overall
# memory-pipeline efficiency rather than one isolated bottleneck, and it is not from tensor cores.
#
# Profile with:
#   ncu --target-processes all --kernel-name regex:gqa_sk --launch-count 2 \
#       --section SpeedOfLight --section Occupancy --section WarpStateStats \
#       python profile_gsk.py

import torch
from torch.utils.cpp_extension import load

mod = load(name="decode_attn_gqa_sk_fast", sources=["kernels/decode_attn_gqa_sk_fast.cu"],
           extra_cuda_cflags=["-O3"], verbose=False)
dev = "cuda"
torch.manual_seed(0)

Hq, Hkv, D, N, S, W, CW = 32, 8, 128, 16384, 64, 8, 8
q = torch.randn(Hq, D, dtype=torch.float16).to(dev).contiguous()
K = torch.randn(Hkv, N, D, dtype=torch.float16).to(dev).contiguous()
V = torch.randn(Hkv, N, D, dtype=torch.float16).to(dev).contiguous()
mp = torch.empty(Hq, S, dtype=torch.float32, device=dev)
lp = torch.empty(Hq, S, dtype=torch.float32, device=dev)
op = torch.empty(Hq, S, D, dtype=torch.float32, device=dev)
o = torch.empty(Hq, D, dtype=torch.float16, device=dev)

for _ in range(20):
    mod.gqa_sk_fast(q, K, V, mp, lp, op, o, S, W, CW)
torch.cuda.synchronize()
mod.gqa_sk_fast(q, K, V, mp, lp, op, o, S, W, CW)
torch.cuda.synchronize()
print("done")
