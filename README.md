# Open Kernel Challenge — Results

GPU inference optimizations evaluated by the [Open Kernel Challenge](https://openkernelchallenge.dev/) against its SGLang baseline on NVIDIA H200 GPUs.

## Standings

**Overall: 647 of 1,271 participants**, with submissions in four workloads, under **`xharshvardhan`**.

Snapshot checked **27 October 2026, 03:20 UTC**. Rankings change as submissions arrive.

| Workload | Speedup over SGLang | Rank | Area optimized |
| --- | ---: | ---: | --- |
| [Dense projections](kernels/dense_projections.py) | **1.1991×** | 639 / 1,109 | Matrix projections and expert selection |
| [Routed experts](kernels/routed_experts.py) (4-bit weights, 16-bit activations) | **1.3749×** | 579 / 877 | Expert computation and output aggregation |
| [Attention decoding](kernels/attention_decode.py) | **1.2990×** | 663 / 836 | Attention processing and result combination |
| [Complete decoder layer](kernels/decoder_layer.py) | **1.2252×** | 611 / 758 | Integration of projection, attention and expert improvements |

The listed scored submissions passed the official correctness checks. Speedups are workload-level measurements relative to SGLang; the overall leaderboard reports a rank rather than a single combined speedup.

## What the code optimizes

- **Dense projections:** selects matrix-vector or matrix-matrix kernels by shape, repacks BF16 weights losslessly, and uses fused or split-K router kernels while retaining FP32 accumulation.
- **Routed experts:** fuses expert-route mapping and alignment, reduces unnecessary padding, and combines normalization with weighted output aggregation. Larger batches group experts by compatible tile sizes without adding padded rows. Weights retain the native compressed NVFP4 path.
- **Attention:** fuses normalization, rotary position processing and FP8 cache writes, then uses grouped-query decode kernels with parallel combination of partial attention results. All required keys remain included.
- **Decoder layer:** integrates projection, attention and expert improvements while preserving normalization boundaries, residual precision and stream dependencies. It is measured as a complete workload, independently of the component scores.

## Using the implementations

Each file in [`kernels/`](kernels/) is a self-contained challenge entry with the original problem identifier. Weight-based entries expose `prepare(weights)` and `kernel_fn(data, state)`; attention exposes `kernel_fn(data)`. Input contracts and shapes are defined in the [official problem specifications](https://openkernelchallenge.dev/#problems).

Use the challenge's CUDA environment and its PyTorch, Triton, SGLang/sgl-kernel, FlashInfer and DeepGEMM dependencies. The implementations target SGLang commit `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`; they are not standalone CPU programs.

Published copies have cleaned comments and module descriptions. Their executable syntax trees match the verified submissions; the listed measurements belong to those official scored runs, not a new benchmark of the edited presentation.

## Verify the results

Visit the [official leaderboard](https://openkernelchallenge.dev/leaderboard) and find **`xharshvardhan`** on the overall board or the relevant problem board:

- `muse_bf16_proj` — dense projections
- `muse_moe_w4a16` — routed experts
- `muse_attention_decode` — attention decoding
- `muse_decode_layer` — complete decoder layer

Portions are adapted from SGLang. See [attribution](NOTICE) and the [upstream Apache-2.0 license](licenses/Apache-2.0.txt).
