# okc: problem=muse_bf16_proj
"""Shape-selected BF16 projections and an FP32-accumulating router.

Uses specialized small-batch kernels, lossless BF16 weight layouts, and
shape-selected fused, split-K or packed router products.

Adapted from SGLang, copyright 2023-2024 SGLang Team (Apache-2.0).
Modified for the Open Kernel Challenge; see NOTICE.
"""

from typing import TypedDict

import deep_gemm
import torch
import triton
import triton.language as tl
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv
from flashinfer.gemm.routergemm import tinygemm_bf16


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


@triton.jit
def _one_token_projection(
    X, W, Y,
    N: tl.constexpr,
    K: tl.constexpr,
    X_STRIDE: tl.constexpr,
    W_ROW_STRIDE: tl.constexpr,
    W_COL_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    # Each program calculates ROWS output features independently.
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    x = tl.load(X + cols * X_STRIDE, mask=cols < K, other=0).to(tl.float32)
    w = tl.load(
        W + rows[:, None] * W_ROW_STRIDE + cols[None, :] * W_COL_STRIDE,
        mask=(rows[:, None] < N) & (cols[None, :] < K),
        other=0,
        cache_modifier=".cg",
    ).to(tl.float32)
    # Read every input and weight at its original precision; accumulate in FP32.
    result = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + rows, result, mask=rows < N)


@triton.jit
def _fused_router_vector(
    X, W, Y,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    SXM: tl.constexpr, SXK: tl.constexpr,
    SWN: tl.constexpr, SWK: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr,
):
    # One program computes BM tokens for one expert logit. The weight
    # vector is shared by these tokens; all partial sums stay on chip.
    n = tl.program_id(0)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BK), dtype=tl.float32)
    for tile in range(tl.cdiv(K, BK)):
        k = tile * BK + rk
        w = tl.load(W + n * SWN + k * SWK, mask=k < K, other=0)
        x = tl.load(
            X + m[:, None] * SXM + k[None, :] * SXK,
            mask=(m[:, None] < M) & (k[None, :] < K), other=0,
        ).to(tl.float32)
        acc = tl.fma(x, w[None, :], acc)
    out = tl.sum(acc, axis=1)
    tl.store(Y + m * N + n, out, mask=m < M)


@triton.jit
def _router_splitk_products(
    X, WH, WL, P,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    SXM: tl.constexpr, SXK: tl.constexpr,
    SWK: tl.constexpr, SWN: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLITS: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    rk = tl.arange(0, BK)
    tiles = tl.cdiv(K, SPLITS * BK)
    high = tl.zeros((BM, BN), dtype=tl.float32)
    low = tl.zeros((BM, BN), dtype=tl.float32)
    for tile in range(tiles):
        k = split * tiles * BK + tile * BK + rk
        x = tl.load(
            X + m[:, None] * SXM + k[None, :] * SXK,
            mask=(m[:, None] < M) & (k[None, :] < K), other=0,
        )
        offsets = k[:, None] * SWK + n[None, :] * SWN
        mask = (k[:, None] < K) & (n[None, :] < N)
        wh = tl.load(WH + offsets, mask=mask, other=0)
        wl = tl.load(WL + offsets, mask=mask, other=0)
        high = tl.dot(x, wh, high)
        low = tl.dot(x, wl, low)
    tl.store(
        P + (split * M + m[:, None]) * N + n[None, :], high + low,
        mask=(m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _router_reduce_partials(P, Y, SIZE: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, SPLITS)
    parts = tl.load(P + split[:, None] * SIZE + i[None, :], mask=i[None, :] < SIZE, other=0)
    tl.store(Y + i, tl.sum(parts, axis=0), mask=i < SIZE)


@triton.jit
def _router_sum_packed_columns(R, Y, SIZE: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // N, i % N
    high = tl.load(R + row * (2 * N) + col, mask=i < SIZE, other=0)
    low = tl.load(R + row * (2 * N) + N + col, mask=i < SIZE, other=0)
    tl.store(Y + i, high + low, mask=i < SIZE)


def prepare(weights: Weights):
    """Use the same router representation as the provided SGLang baseline."""
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.T.contiguous()
    state["router_lo_t"] = lo.T.contiguous()
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_packed_t"] = torch.cat((hi.T, lo.T), dim=1).contiguous()
    # Repacking preserves every BF16 weight exactly.
    for name in ("pooler", "unpooler"):
        state[name + "_t"] = weights[name].T.contiguous()
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x, config = data["x"], data["config"]
    layer, n = config["layer"], config["n"]
    if layer == "router":
        m, k = x.shape
        if m <= 16:
            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            w = state["router"]
            bm = 1 if m == 1 else (4 if m <= 8 else 8)
            bk = 8192 if m == 1 else (1024 if m <= 8 else 512)
            _fused_router_vector[(n, triton.cdiv(m, bm))](
                x, w, out, m, n, k,
                x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                BM=bm, BK=bk, num_warps=4,
            )
            return out
        if m > 128:
            packed = torch.mm(x, state["router_packed_t"], out_dtype=torch.float32)
            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            _router_sum_packed_columns[(triton.cdiv(m * n, 256),)](
                packed, out, m * n, N=n, BLOCK=256, num_warps=4,
            )
            return out
        bm = 16 if m <= 32 else 64
        splits = 32 if m <= 32 else 16
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        partials = torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
        wh, wl = state["router_hi"], state["router_lo"]
        _router_splitk_products[(triton.cdiv(m, bm), triton.cdiv(n, 32), splits)](
            x, wh, wl, partials, m, n, k,
            x.stride(0), x.stride(1), wh.stride(1), wh.stride(0),
            BM=bm, BN=32, BK=128, SPLITS=splits, num_warps=4, num_stages=2,
        )
        _router_reduce_partials[(triton.cdiv(m * n, 256),)](
            partials, out, m * n, SPLITS=splits, BLOCK=256, num_warps=4,
        )
        return out

    w = state[layer][:n]
    m = x.shape[0]
    if m == 1 and layer in ("qkv_gate", "pooler"):
        return hopper_bf16_gemv(x, w)
    if layer == "unpooler" and 1 < m <= 8:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        tinygemm_bf16(x, w, out, use_pdl=False)
        return out
    if (layer == "unpooler" and m > 8) or (layer == "pooler" and m > 128):
        return torch.mm(x, state[layer + "_t"][:, :n])
    if (layer == "pooler" and m <= 128) or (layer == "qkv_gate" and 64 <= m <= 128):
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        # Scope the public algorithm selector to this call and restore it on exit.
        deep_gemm.use_deterministic_algorithms(True)
        try:
            deep_gemm.bf16_gemm_nt(x, w, out)
        finally:
            deep_gemm.use_deterministic_algorithms(False)
        return out
    if x.shape[0] == 1 and layer != "qkv_gate":
        k = x.shape[1]
        out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
        _one_token_projection[(triton.cdiv(n, 4),)](
            x, w, out,
            N=n, K=k,
            X_STRIDE=x.stride(1),
            W_ROW_STRIDE=w.stride(0), W_COL_STRIDE=w.stride(1),
            ROWS=4, COLS=triton.next_power_of_2(k),
            num_warps=8 if k >= 4096 else 4,
        )
        return out
    return _bf16_gemm_dispatch_impl(x, w, None)
