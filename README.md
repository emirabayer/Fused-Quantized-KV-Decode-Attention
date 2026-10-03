# Fused Quantized-KV Decode-Attention CUDA Kernel

A single-query decode-step attention kernel for CUDA, written for the A100. One kernel does qK^T, an
online softmax, and the weighted sum over V, with fp32 accumulation and no global score buffer. There
are four single-block variants (fp16, int8 KV, int4 KV, and GQA-aware fp16) and a split-K version for
long context that is compared against FlashInfer. Profiled on an A100-SXM4-40GB.

## Results

All numbers below were measured on the A100. results.json holds the consolidated output, and
notebooks/integration_and_benchmarks.ipynb regenerates everything here.

End-to-end decode in Llama-3.2-1B, prompt 512, generate 128:

- stock PyTorch SDPA: 51.9 tok/s
- this kernel: 54.7 tok/s (1.05x)

The kernel produced the same generated tokens as the stock model in that run.

Standalone KV precision (microbenchmark on random data, wall-clock ms, speedup vs the fp16 fused
kernel):

| B | N | int8 vs fp16 | int4 vs fp16 |
|---|---|---|---|
| 8 | 4096 | 0.91x | 0.85x |
| 256 | 4096 | 0.99x | 0.98x |
| 2048 | 4096 | 1.25x | 1.34x |

Quantization only pays off once the batch is large enough to be bandwidth bound. At small batch the
dequant cost dominates and fp16 wins.

GQA-aware kernel against expanding the KV heads with repeat_kv and calling the plain kernel:

| B | speedup |
|---|---|
| 1 | 2.44x |
| 4 | 1.90x |
| 16 | 3.58x |

This is the cost that repeat_kv materialization adds to the baseline. In the live model the GQA kernel
runs at parity with SDPA's own GQA path.

## Split-K decode and FlashInfer comparison

The newest kernel is a split-K (FlashDecoding-style) version in kernels/decode_attn_splitk.cu. The
single-block kernels above give one block per head, so at long context that one block walks the whole
KV cache while most of the GPU sits idle. Split-K cuts the sequence into S contiguous chunks and gives
each (head, chunk) pair its own block. A partial kernel runs the online softmax over its chunk and
writes unnormalized (m, l, o) partials to fp32 scratch, then a small combine kernel merges the S
partials per head into the final output. S is chosen per shape by measurement.

Measured against FlashInfer's single_decode on the A100, fp16, as speedup over FlashInfer (above 1.0
means split-K is faster):

| N | head_dim 64 | head_dim 128 |
|---|---|---|
| 512 | 1.59x | 1.62x |
| 1024 | 1.48x | 1.52x |
| 2048 | 1.28x | 0.94x |
| 4096 | 0.82x | 0.63x |
| 8192 | 0.49x | 0.43x |
| 16384 | 0.35x | 0.34x |

Split-K is faster than FlashInfer up to about 2K context at head_dim 64, and up to about 1K at head_dim
128, reaching about 1.6x at short context. head_dim 128 crosses over to FlashInfer at a smaller N
because each token carries twice the bytes, so the memory-bound regime where FlashInfer wins starts
earlier. The crossover is in plots/flashinfer_crossover.png.

At long context the split-K parallelization is what makes the kernel usable. At head_dim 64, N=16384 a
single block (S=1) takes 1.69 ms, about 29x slower than FlashInfer's 0.058 ms. Split-K brings that to
0.166 ms, about 2.8x slower, so the long-context gap goes from roughly 29x to about 3x.

The rest of the gap is bandwidth. Profiling the partial kernel at head_dim 128, N=16384 with Nsight
Compute shows about 57% occupancy, 69% DRAM throughput, and 29% compute, so the kernel is memory bound
and FlashInfer's long-context edge does not come from tensor cores. profile_splitk.py and the ncu
command inside it reproduce this.

Two things about the comparison:

- The split-K timings allocate the fp32 scratch inside the timed call, which penalizes split-K. The
  real speedups are a little higher than the table shows.
- The comparison is fp16 only. FlashInfer's quantized decode path is fp8, while the quantized kernels
  here are int8 and int4, so there is no matching quantized comparison to run.

I tried two changes that did not help and kept them out of the main kernel. A vectorized partial kernel
(kernels/decode_attn_splitk_vec.cu, 32-bit and 64-bit K and V loads with all 32 lanes active) came
within 2 to 3% of the scalar version, because the kernel is bandwidth bound rather than limited by load
issue. Lowering W to raise occupancy also did not help: at head_dim 128, N=16384 the best time was W=8,
S=16, and W=4 and W=2 were slower.

flashinfer_compare.py reproduces the comparison table.

## How it works

Each attention head is one CUDA block. The keys are split across W warps: each warp strides through
the sequence keeping a running max, a running denominator, and a running output accumulator (the online
softmax recurrence), so the full score vector is never written to global memory and the accumulation
stays in fp32. One warp then combines the per-warp partials through shared memory. The head dimension
is spread across the 32 lanes, D/32 elements per lane.

Nsight Compute profiling drove the layout. A first version with one thread per head barely touched the
A100's memory bandwidth. Splitting the work across warps and blocks moves it close to the
bandwidth-bound limit at large N (peak on this A100 is around 1555 GB/s, recorded in results.json).

The int8 and int4 variants keep K and V quantized with per-token symmetric scales (one fp16 scale per
token) and dequantize in registers inside the loop. int4 packs two values per byte. The GQA-aware
variant reads KV head h/G directly instead of expanding the KV heads first, which is where its speedup
comes from.

I also tried char2 vectorized dequant and register prefetch on the quantized path. Neither beat the
simple version on the A100, so I removed both.

## Build and run

There is no separate build step. The tests compile the .cu files at import time with
torch.utils.cpp_extension.load, so you need PyTorch with CUDA and a working nvcc.

```
pip install torch pytest
pytest tests/
```

The suite has 15 tests, covering all six kernels, and skips automatically on a machine with no CUDA.

Each kernel is loaded as a function, for example:

```python
from torch.utils.cpp_extension import load
m = load(name="fp16", sources=["kernels/decode_attn_fp16_mw.cu"], extra_cuda_cflags=["-O3"])
out = m.decode_attn_fp16_mw(q, K, V, W)   # q:[B,D]  K,V:[B,N,D]  W warps
```

## Limitations

- fp16 and int8 support D=64 and D=128. int4 and the GQA-aware kernel are D=64 only.
- int8 per-token symmetric quantization holds about 1% error against fp32. int4 with the same simple
  scheme shows about 15 to 23% error on the microbenchmark and was not validated in the live model. It
  would need per-channel or group-wise scales and outlier handling to be usable, so its speedup is
  listed only for completeness.
- The quantized kernels only help at large batch, as the table shows.

## Layout

- kernels/ : the six CUDA sources (four single-block, plus split-K scalar and vectorized)
- tests/ : pytest suite
- benchmarks/ : raw measured JSON, consolidated into results.json
- plots/ : charts, including flashinfer_crossover.png
- flashinfer_compare.py : reproduces the split-K vs FlashInfer table
- profile_splitk.py : runs the long-context shape under Nsight Compute
- notebooks/ : integration_and_benchmarks.ipynb (the live Llama-3.2-1B integration, int4 accuracy, and
  end-to-end benchmark) and splitk_flashinfer.ipynb (the split-K kernel development and the FlashInfer
  comparison)
