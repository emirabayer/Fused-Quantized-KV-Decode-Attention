# Fused Quantized-KV Decode-Attention CUDA Kernel

A single-query decode-step attention kernel for CUDA, written for the A100. One kernel does qK^T, an
online softmax, and the weighted sum over V, with fp32 accumulation and no global score buffer. There
are four variants: fp16, int8 KV, int4 KV, and a GQA-aware fp16 version. Profiled on an
A100-SXM4-40GB.

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

The suite has 7 tests and skips automatically on a machine with no CUDA.

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

- kernels/ : the four CUDA sources
- tests/ : pytest suite
- benchmarks/ : raw measured JSON, consolidated into results.json
- plots/ : charts
- notebooks/ : integration_and_benchmarks.ipynb, the Colab notebook that builds the kernels, runs the
  live Llama-3.2-1B integration, the int4 accuracy checks, and the end-to-end benchmark, and writes
  results.json
