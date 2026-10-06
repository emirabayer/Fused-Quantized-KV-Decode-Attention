# Fused Quantized-KV Decode-Attention CUDA Kernel

A single-query decode-step attention kernel for CUDA, written for the A100. One kernel does qK^T, an
online softmax, and the weighted sum over V, with fp32 accumulation and no global score buffer. There
are four single-block variants (fp16, int8 KV, int4 KV, and GQA-aware fp16) and a GQA-aware split-K
kernel for long context, compared against FlashInfer. Profiled on an A100-SXM4-40GB.

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

## GQA-aware split-K decode and FlashInfer comparison

The newest kernel is a GQA-aware split-K (FlashDecoding-style) decode kernel in
kernels/decode_attn_gqa_sk_fast.cu, compared against FlashInfer's single_decode. The single-block
kernels above give one block per head, so at long context that block walks the whole KV cache while
most of the GPU sits idle. Split-K cuts the sequence into S contiguous chunks and gives each (head,
chunk) pair its own block. A partial kernel runs the online softmax over its chunk into preallocated
fp32 scratch, then a parallel combine kernel (CW warps) merges the S partials per head. The kernel
reads compact KV head h/G directly, so it never expands the KV heads. S, W, and CW are chosen per shape
by measurement.

An earlier version of this comparison was wrong in two ways and has been replaced. It timed our kernel
reading KV expanded to 32 heads while FlashInfer read the compact 8-head GQA cache, a 4x byte
difference, and it mixed launch overhead into only part of the timing. The numbers here read the same
compact 8-head KV on both sides and loop-time both sides the same way (2000 calls per measurement, with
the scratch preallocated outside the timed loop).

Loop-timed on the A100, fp16, as FlashInfer call latency over ours (above 1.0 means our call returns
sooner):

| N | head_dim 64 | head_dim 128 |
|---|---|---|
| 512 | 2.72x | 2.42x |
| 1024 | 2.21x | 1.92x |
| 2048 | 1.66x | 1.41x |
| 4096 | 1.12x | 0.93x |
| 8192 | 0.72x | 0.45x |
| 16384 | 0.32x | 0.48x |

Short context: our call returns sooner, up to 2.7x, and stays ahead until the crossover around N=4K
(head_dim 64 is 1.12x at 4K, head_dim 128 is 0.93x). This is a lower-per-call-overhead effect. Our
extension dispatches in about 10us while FlashInfer has a floor near 28us, so at short context the call
completes sooner even though the kernel itself is no faster. The crossover is in
plots/flashinfer_crossover.png.

Long context (N at or above 8K): FlashInfer wins. At 16K we are 0.32x at head_dim 64 and 0.48x at
head_dim 128. On sustained bandwidth FlashInfer reaches about 1100 GB/s at long context, around 71% of
the A100's 1555 GB/s peak, while ours peaks around 500 GB/s, about a third of peak and lower at head_dim
64, so its kernel is roughly 2x more bandwidth-efficient. Profiling the partial kernel at head_dim 128,
N=16384 shows nothing saturated: about 52% SM throughput, 44% memory throughput, 30% DRAM throughput,
and 80% occupancy. The long-context gap is overall memory-pipeline efficiency rather than one isolated
bottleneck. Tensor cores do not explain it. profile_gsk.py and the ncu command inside it reproduce the
profile.

Two notes on reading the table:

- The GB/s at short N is launch-floor-limited and partly served from L2 (the A100 L2 is 40MB, and the
  KV cache fits in it for every shape here except head_dim 128 at 16K). Those are call-rate numbers, so
  read the long-context rows for sustained DRAM bandwidth.
- The comparison is fp16 only by design. FlashInfer's quantized decode path is fp8, while the quantized
  kernels here are int8 and int4, so there is no matching quantized comparison to run.

flashinfer_looptimed.py reproduces the table.

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

The suite has 13 tests, covering all five kernels, and skips automatically on a machine with no CUDA.

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

- kernels/ : the five CUDA sources (four single-block, plus the GQA-aware split-K kernel)
- tests/ : pytest suite
- benchmarks/ : raw measured JSON, consolidated into results.json
- plots/ : charts, including flashinfer_crossover.png
- flashinfer_looptimed.py : reproduces the GQA-aware split-K vs FlashInfer table
- profile_gsk.py : runs the long-context shape under Nsight Compute
- notebooks/ : integration_and_benchmarks.ipynb (the live Llama-3.2-1B integration, int4 accuracy, and
  end-to-end benchmark)
