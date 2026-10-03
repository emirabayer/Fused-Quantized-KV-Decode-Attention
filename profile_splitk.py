# Runs one split-K launch at the long-context shape (D=128, N=16384) so Nsight Compute can profile
# both the partial kernel and the combine kernel. This is the shape where split-K is about 0.34x of
# FlashInfer, and the profile shows why: the partial kernel is memory bound (around 57% occupancy,
# ~69% DRAM throughput, ~29% compute), so the gap is bandwidth rather than tensor cores.
#
# Profile with:
#   ncu --target-processes all --kernel-name regex:splitk --launch-count 2 \
#       --section SpeedOfLight --section Occupancy --section WarpStateStats \
#       python profile_splitk.py

import torch
from torch.utils.cpp_extension import load

mod = load(name="decode_attn_splitk", sources=["kernels/decode_attn_splitk.cu"],
           extra_cuda_cflags=["-O3"], verbose=False)
dev = "cuda"
torch.manual_seed(0)

H, N, D, S, W = 32, 16384, 128, 16, 8
q = torch.randn(H, D, dtype=torch.float16).to(dev).contiguous()
K = torch.randn(H, N, D, dtype=torch.float16).to(dev).contiguous()
V = torch.randn(H, N, D, dtype=torch.float16).to(dev).contiguous()

for _ in range(20):
    _ = mod.decode_attn_splitk(q, K, V, S, W)
torch.cuda.synchronize()
_ = mod.decode_attn_splitk(q, K, V, S, W)
torch.cuda.synchronize()
print("done")
