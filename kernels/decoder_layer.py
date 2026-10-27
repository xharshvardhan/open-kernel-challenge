# okc: problem=muse_decode_layer
"""Optimized Muse Spark decoder layer for one GPU under TP8/EP8.

Integrates specialized dense projections, attention preprocessing and decode,
and expert route alignment and output aggregation. Native normalization,
residual precision and stream dependencies are preserved. Communication is
represented by the challenge's identity and zero-padding contracts.

Adapted from SGLang, copyright 2023-2024 SGLang Team (Apache-2.0).
Modified for the Open Kernel Challenge; see NOTICE.
"""

from typing import TypedDict

import types

import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv
from flashinfer.gemm.routergemm import tinygemm_bf16
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
    _post_expert_norm_weighted_sum_kernel,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    # Lossless weight-only NN layouts; native weights/boundaries stay intact.
    state["_dense_transposed"] = {
        name: state["dense"][name].T.contiguous()
        for name in ("pooler", "unpooler")
    }
    del freqs
    torch.cuda.synchronize()
    return state


def _experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    # The unchanged whole caller already computed fresh relative IDs on GPU.
    # Physical bank rows are0..31; global IDs remain untouched for the return.
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    block_size_m = min(block_size_m, 16)
    if 0 < num_tokens <= 256:
        local_ids, sorted_ids, block_experts, n_padded = _fused_local_alignment(
            topk_ids, 0, LOCAL_EXPERTS, block_size_m)
    else:
        local_ids = bank["expert_map"][topk_ids].contiguous()
        sorted_ids, block_experts, n_padded = moe_align_block_size(
            local_ids, block_size_m, N_EXPERTS)
    topk_weights = topk_weights.to(torch.float32).contiguous()
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    if num_tokens < 32:
        return _small_post_expert_norm_weighted_sum(
            y, topk_weights, state["post_expert_w"], local_ids)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = _dense_project(data["attn_input"], dense["qkv_gate"], "qkv_gate", state)
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    has_rope = bool(cfg["rope_theta"])
    # Reuse the whole layer's existing fixed theta=500000 table from prepare.
    cos_sin = state["cos_sin"] if has_rope else qkv
    q = torch.empty((tokens, hq, d), dtype=FP8, device=qkv.device)
    n_rows = tokens * (hq + 2 * hkv)
    block_m = 4 if n_rows < 1024 else (8 if n_rows < 8192 else (16 if n_rows < 65536 else 32))
    _fused_attention_preprocess[(triton.cdiv(n_rows, block_m),)](
        qkv, q, data["positions"], cos_sin,
        k_cache, v_cache, data["cache_loc"], data["k_scale"], data["v_scale"],
        STRIDE_QKV=qkv.stride(0), N_ROWS=n_rows,
        H_Q=hq, H_KV=hkv, D=d,
        BLOCK_D=triton.next_power_of_2(d), BLOCK_M=block_m,
        HAS_ROPE=has_rope, SCALED=scaled,
        EPS=RMS_EPS, num_warps=4,
    )
    gate = qkv[:, hq * d + 2 * hkv * d:]
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = _try_transposed_full_decode(data, q, cfg)
    if out is None:
        out = _try_transposed_window_decode(data, q, cfg)
    if out is None:
        out = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=data["page_table"],
            cache_seqlens=data["cache_seqlens"],
            cu_seqlens_q=data["cu_seqlens_q"],
            max_seqlen_q=length,
            softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
            causal=True,
            window_size=(window - 1, 0) if window else (-1, -1),
            scheduler_metadata=data["scheduler_metadata"],
            **descale,
        )
    if isinstance(out, tuple):
        out = out[0]
    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    attn_output = _dense_project(o, dense["o"], "o", state)  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _dense_project(ffn_input, dense["pooler"], "pooler", state)
    with torch.cuda.stream(side_stream):
        if split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _dense_project(expert_output, dense["unpooler"][:UNPOOL_SLICE], "unpooler", state)
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _dense_project(expert_output, dense["unpooler"], "unpooler", state)

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids


# Expert helpers adapted from SGLang (Apache-2.0).
def _small_post_expert_norm_weighted_sum(y, scores, norm_w, local_ids):
    """Use SGLang's existing FP32 Triton finalizer below its wrapper threshold.

    The upstream wrapper takes a multi-launch PyTorch path below 32 tokens.
    This launch retains its larger-batch kernel's original argument contract.
    Marlin output and effective norm weights are BF16; multiplication, norm,
    routing weights, and accumulation stay FP32 until the final BF16 store.
    Negative local expert IDs are skipped before their output slots are read.
    No input values are inspected on the host or retained between calls.
    """
    tokens, topk, hidden = y.shape
    out = torch.empty((tokens, hidden), dtype=y.dtype, device=y.device)
    block_h = triton.next_power_of_2(hidden)
    _post_expert_norm_weighted_sum_kernel[(tokens,)](
        y, norm_w, scores, out, local_ids,
        TOPK=topk,
        H=hidden,
        eps=POST_NORM_EPS,
        APPLY_NORM=True,
        HAS_WEIGHT=True,
        WEIGHT_FIRST=True,
        HAS_IDS=True,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=2,
    )
    return out


@triton.jit
def _fused_local_alignment_kernel(
    global_ids_ptr, local_ids_ptr, sorted_ids_ptr, block_experts_ptr,
    n_padded_ptr,
    ROUTES: tl.constexpr,
    LOCAL_START: tl.constexpr,
    LOCAL_COUNT: tl.constexpr,
    BM: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # One CTA owns every output. Initialize padding and unused descriptors.
    p = tl.arange(0, BLOCK_P)
    tl.store(sorted_ids_ptr + p, ROUTES, mask=p < CAPACITY)
    d = tl.arange(0, BLOCK_D)
    tl.store(block_experts_ptr + d, -1, mask=d < CAPACITY // BM)

    route = tl.arange(0, BLOCK_R)
    g = tl.load(global_ids_ptr + route, mask=route < ROUTES, other=-1).to(tl.int32)
    local = g - LOCAL_START
    live = (route < ROUTES) & (local >= 0) & (local < LOCAL_COUNT)
    tl.store(local_ids_ptr + route, tl.where(live, local, -1), mask=route < ROUTES)

    # Nonlocal/tail lanes use a dummy histogram bin, then disappear. There
    # are 32 local bins; bins 32..63 never contribute counts or padding.
    bins = tl.where(live, local, LOCAL_COUNT)
    hist = tl.histogram(bins, 64)
    expert = tl.arange(0, 64)
    counts = tl.where(expert < LOCAL_COUNT, hist, 0)
    prefix = tl.cumsum(counts, axis=0) - counts
    padded = ((counts + BM - 1) // BM) * BM
    padded_prefix = tl.cumsum(padded, axis=0) - padded

    # Unique nonnegative packed keys sort by expert, then original route
    # index. The largest valid key is < LOCAL_COUNT*BLOCK_R. Invalid keys
    # sort last, and are masked before any scatter or descriptor write.
    key = tl.where(live, local * BLOCK_R + route, 2147483647)
    key = tl.sort(key, descending=False)
    sorted_live = key < LOCAL_COUNT * BLOCK_R
    safe_expert = tl.minimum(key // BLOCK_R, LOCAL_COUNT - 1)
    original_route = key % BLOCK_R
    rank = route - tl.gather(prefix, safe_expert, axis=0)
    destination = tl.gather(padded_prefix, safe_expert, axis=0) + rank

    # Fill and scatter can target the same global address from different
    # threads. All initial stores must finish before any valid route writes.
    # This is a CTA barrier inside the GPU kernel, not a host synchronization.
    tl.debug_barrier()
    tl.store(sorted_ids_ptr + destination, original_route, mask=sorted_live)
    # Each padded block starts with a real route; precisely one sorted lane
    # owns that block's descriptor. Valid rows precede the ROUTES sentinels.
    tl.store(block_experts_ptr + destination // BM, safe_expert,
             mask=sorted_live & (rank % BM == 0))
    tl.store(n_padded_ptr, tl.sum(padded, axis=0))


def _fused_local_alignment(global_ids, local_start, local_count, block_size_m):
    # Shape/config-only allocation; no input values leave the GPU.
    if local_count != 32 or block_size_m not in (8, 16):
        raise ValueError("Unsupported fused local alignment configuration")
    ids = global_ids.contiguous()
    routes = ids.numel()
    if routes <= 0 or routes > 2048:
        raise ValueError("Fused local alignment supports 1..2048 routes")
    capacity = min(routes * block_size_m,
                   routes + local_count * (block_size_m - 1))
    capacity = triton.cdiv(capacity, block_size_m) * block_size_m
    local_ids = torch.empty(ids.shape, device=ids.device, dtype=torch.int32)
    sorted_ids = torch.empty((capacity,), device=ids.device, dtype=torch.int32)
    block_experts = torch.empty((capacity // block_size_m,),
                               device=ids.device, dtype=torch.int32)
    n_padded = torch.empty((1,), device=ids.device, dtype=torch.int32)
    if 1024 < routes <= 2048 and local_count == 32 and block_size_m == 16:
        _launch_parallel_alignment(
            ids, local_ids, sorted_ids, block_experts, n_padded,
            local_start, local_count, block_size_m, capacity)
    else:
        _fused_local_alignment_kernel[(1,)](
            ids, local_ids, sorted_ids, block_experts, n_padded,
            ROUTES=routes, LOCAL_START=local_start, LOCAL_COUNT=local_count,
            BM=block_size_m, CAPACITY=capacity,
            BLOCK_R=triton.next_power_of_2(routes),
            BLOCK_P=triton.next_power_of_2(capacity),
            BLOCK_D=triton.next_power_of_2(capacity // block_size_m),
            num_warps=4,
            num_stages=1,
        )
    return local_ids, sorted_ids, block_experts, n_padded


# Online-softmax/split-merge adapted from SGLang Team, Apache-2.0:
# https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/attention/decode_attention.py
# Hardware metadata and own compilation guards only; no input-derived cache.
_OWN_GQA_DIAGNOSTIC_PRINTED = False
_OWN_WINDOW_SMS = {}
_OWN_WINDOW_DIAGNOSTIC_KEYS = set()


@triton.jit
def _fused_attention_preprocess(
    QKV, Q_OUT, POS, COS_SIN, K_CACHE, V_CACHE, CACHE_LOC, K_SCALE, V_SCALE,
    STRIDE_QKV: tl.constexpr,
    N_ROWS: tl.constexpr,
    H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr,
    HAS_ROPE: tl.constexpr, SCALED: tl.constexpr,
    EPS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_D)
    row_mask = rows < N_ROWS
    mask = row_mask[:, None] & (cols < D)[None, :]
    heads: tl.constexpr = H_Q + 2 * H_KV
    token = rows // heads
    head = rows % heads
    is_q = head < H_Q
    is_k = (head >= H_Q) & (head < H_Q + H_KV)
    is_v = head >= H_Q + H_KV
    src = token * STRIDE_QKV + head * D
    raw = tl.load(QKV + src[:, None] + cols[None, :], mask=mask, other=0.0)
    value = raw.to(tl.float32)
    variance = tl.sum(tl.where(mask, value * value, 0.0), axis=1) / D
    inv_rms = tl.rsqrt(variance + EPS)
    normed = value * inv_rms[:, None]

    if HAS_ROPE:
        # The standalone norm writes BF16 before RoPE reads it.
        normed = normed.to(QKV.dtype.element_ty).to(tl.float32)
        partner = tl.load(
            QKV + src[:, None] + (cols ^ 1)[None, :],
            mask=mask, other=0.0,
        ).to(tl.float32)
        partner = (partner * inv_rms[:, None]).to(QKV.dtype.element_ty).to(tl.float32)
        position = tl.load(POS + token, mask=row_mask, other=0).to(tl.int64)
        frequency = cols // 2
        cosine = tl.load(
            COS_SIN + position[:, None] * D + frequency[None, :],
            mask=mask, other=1.0,
        )
        sine = tl.load(
            COS_SIN + position[:, None] * D + D // 2 + frequency[None, :],
            mask=mask, other=0.0,
        )
        normed = tl.where(
            (cols % 2 == 0)[None, :],
            normed * cosine - partner * sine,
            partner * sine + normed * cosine,
        )

    # Both full-layer norm and sliding-layer RoPE outputs are BF16.
    normed = normed.to(QKV.dtype.element_ty).to(tl.float32)
    q_dst = (token * H_Q + head) * D
    tl.store(Q_OUT + q_dst[:, None] + cols[None, :], normed, mask=mask & is_q[:, None])

    # V is the raw BF16 projection, without norm or RoPE.
    cache_value = tl.where(is_v[:, None], raw.to(tl.float32), normed)
    if SCALED:
        k_scale = tl.load(K_SCALE).to(tl.float32)
        v_scale = tl.load(V_SCALE).to(tl.float32)
        scale = tl.where(is_v, v_scale, k_scale)
        # BF16 tensor / zero-dimensional FP32 scalar returns BF16 in PyTorch.
        cache_value = (cache_value / scale[:, None]).to(QKV.dtype.element_ty).to(tl.float32)

    loc = tl.load(CACHE_LOC + token, mask=row_mask, other=-1).to(tl.int64)
    kv_head = tl.where(is_v, head - H_Q - H_KV, head - H_Q)
    dst = loc * (H_KV * D) + kv_head * D
    # Match store_cache(..., reserved_skip_index=-1): cache row zero is live.
    cache_mask = mask & (loc != -1)[:, None]
    tl.store(K_CACHE + dst[:, None] + cols[None, :], cache_value, mask=cache_mask & is_k[:, None])
    tl.store(V_CACHE + dst[:, None] + cols[None, :], cache_value, mask=cache_mask & is_v[:, None])


@triton.jit
def _transposed_gqa_stage1(
    Q, K, V, TABLE, LENGTHS, PARTIAL, LSE,
    Q_BATCH: tl.constexpr, Q_HEAD: tl.constexpr,
    K_SLOT: tl.constexpr, K_HEAD: tl.constexpr,
    V_SLOT: tl.constexpr, V_HEAD: tl.constexpr,
    TABLE_BATCH: tl.constexpr, TABLE_PAGE: tl.constexpr, LEN_STRIDE: tl.constexpr,
    SPLITS: tl.constexpr, SM_SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr = 64, D: tl.constexpr = 64,
    HQ: tl.constexpr = 16, GROUP: tl.constexpr = 8,
):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    n = tl.load(LENGTHS + batch * LEN_STRIDE)
    # The same balanced32-token split geometry is used by the reducer.
    split_size = tl.cdiv(tl.cdiv(n, SPLITS), 32) * 32
    start = split * split_size
    end = tl.minimum(start + split_size, n)
    if start < end:
        heads = kv_head * GROUP + tl.arange(0, GROUP)
        dim = tl.arange(0, D)
        # Q[8,64] and K[64,64] stay at the permitted original E4M3 values.
        q = tl.load(Q + batch * Q_BATCH + heads[:, None] * Q_HEAD + dim[None, :])
        maximum = tl.full((GROUP,), -float("inf"), tl.float32)
        denominator = tl.zeros((GROUP,), tl.float32)
        accumulator = tl.zeros((D, GROUP), tl.float32)
        for tile in tl.range(start, end, BLOCK_N):
            logical_k = tile + tl.arange(0, BLOCK_N)
            valid_k = logical_k < end
            physical_k = tl.load(
                TABLE + batch * TABLE_BATCH + logical_k * TABLE_PAGE,
                mask=valid_k, other=0,
            ).to(tl.int64)
            k = tl.load(
                K + physical_k[:, None] * K_SLOT + kv_head * K_HEAD + dim[None, :],
                mask=valid_k[:, None], other=0.0,
            )
            # M64/N8 permits native Hopper FP8 WGMMA; D64 is two K32 partials.
            score = tl.dot(
                k, tl.trans(q), out_dtype=tl.float32, max_num_imprecise_acc=32,
            ) * SM_SCALE
            score = tl.where(valid_k[:, None], score, -float("inf"))
            new_maximum = tl.maximum(maximum, tl.max(score, axis=0))
            rescale = tl.exp(maximum - new_maximum)
            probability = tl.exp(score - new_maximum[None, :])
            # A live split/tile has a real key; every one of the8 heads is real.
            probability = tl.where(valid_k[:, None], probability, 0.0)
            v = tl.load(
                V + physical_k[:, None] * V_SLOT + kv_head * V_HEAD + dim[None, :],
                mask=valid_k[:, None], other=0.0,
            ).to(tl.bfloat16)
            accumulator = accumulator * rescale[None, :]
            accumulator = tl.dot(
                tl.trans(v), probability.to(tl.bfloat16), accumulator,
                out_dtype=tl.float32,
            )
            denominator = denominator * rescale + tl.sum(probability, axis=0)
            maximum = new_maximum
        partial_idx = ((batch * HQ + heads[:, None]) * SPLITS + split) * D + dim[None, :]
        tl.store(PARTIAL + partial_idx, tl.trans(accumulator) / denominator[:, None])
        tl.store(LSE + (batch * HQ + heads) * SPLITS + split, maximum + tl.log(denominator))


def _own_gqa_compile_diagnostic(stage1, stage2, batch, capacity, splits):
    """Read only our normal Triton launch results, once, during ordinary warmup."""
    global _OWN_GQA_DIAGNOSTIC_PRINTED
    if _OWN_GQA_DIAGNOSTIC_PRINTED:
        return
    import re
    ptx = stage1.asm.get("ptx", "").lower()
    native_qk = bool(re.search(r"\bwgmma\.mma_async[^;\n]*\.e4m3\.e4m3\b", ptx))
    bf16_pv = bool(re.search(r"\b(?:wgmma\.mma_async|mma\.sync)[^;\n]*\.bf16\.bf16\b", ptx))
    def resources(kernel):
        return dict(registers_per_thread=getattr(kernel, "n_regs", None),
                    shared_bytes=getattr(kernel.metadata, "shared", None),
                    num_stages=getattr(kernel.metadata, "num_stages", None),
                    num_warps=getattr(kernel.metadata, "num_warps", None))
    print("OWN_FULL_PARALLEL_MERGE_COMPILE", dict(
        batch=batch, table_capacity=capacity, splits=splits,
        merge_block_s=triton.next_power_of_2(splits), merge_window=capacity,
        triton_version=triton.__version__, native_e4m3_wgmma_qk=native_qk,
        bf16_tensorcore_pv=bf16_pv, stage1=resources(stage1), stage2=resources(stage2),
    ))
    _OWN_GQA_DIAGNOSTIC_PRINTED = True


def _try_transposed_full_decode(data, q, cfg):
    batch = cfg["batch"]
    if not (
        batch >= 16 and cfg["q_len"] == 1 and cfg["window"] == 0
        and cfg["rope_theta"] == 0 and cfg["kv_scale"] == 1.0
        and cfg["page_size"] == 1 and cfg["num_q_heads"] == 16
        and cfg["num_kv_heads"] == 2 and cfg["head_dim"] == 64
    ):
        return None
    k, v = data["k_cache"], data["v_cache"]
    if k.ndim != 4 or v.ndim != 4 or k.stride(-1) != 1 or v.stride(-1) != 1:
        return None
    table, lengths = data["page_table"], data["cache_seqlens"]
    # One shape-capacity policy: about4096 tokens/split; balance actual lengths
    # on-device with the pinned grouped decoder's32-token rounding rule.
    splits = max(1, triton.cdiv(table.shape[1], 4096))
    if batch > 128 or splits > 128 or table.shape[1] == 0:
        return None
    partial = torch.empty((batch, 16, splits, 64), dtype=torch.float32, device=q.device)
    lse = torch.empty((batch, 16, splits), dtype=torch.float32, device=q.device)
    out = torch.empty((batch, 16, 64), dtype=torch.bfloat16, device=q.device)
    first = _transposed_gqa_stage1[(batch, 2, splits)](
        q, k, v, table, lengths, partial, lse,
        Q_BATCH=q.stride(0), Q_HEAD=q.stride(1),
        K_SLOT=k.stride(0), K_HEAD=k.stride(2),
        V_SLOT=v.stride(0), V_HEAD=v.stride(2),
        TABLE_BATCH=table.stride(0), TABLE_PAGE=table.stride(1), LEN_STRIDE=lengths.stride(0),
        SPLITS=splits, SM_SCALE=cfg["sm_scale"] * cfg["qk_scale_factor"],
        num_warps=4, num_stages=2,
    )
    second = _transposed_window_gqa_parallel_merge[(batch, 16)](
        partial, lse, lengths, out,
        LEN_STRIDE=lengths.stride(0), SPLITS=splits, WINDOW=table.shape[1],
        BLOCK_S=triton.next_power_of_2(splits), num_warps=4, num_stages=2,
    )
    _own_gqa_compile_diagnostic(first, second, batch, table.shape[1], splits)
    return out


@triton.jit
def _transposed_window_gqa_stage1(
    Q, K, V, TABLE, LENGTHS, PARTIAL, LSE,
    Q_BATCH: tl.constexpr, Q_HEAD: tl.constexpr,
    K_SLOT: tl.constexpr, K_HEAD: tl.constexpr,
    V_SLOT: tl.constexpr, V_HEAD: tl.constexpr,
    TABLE_BATCH: tl.constexpr, TABLE_PAGE: tl.constexpr, LEN_STRIDE: tl.constexpr,
    SPLITS: tl.constexpr, SM_SCALE: tl.constexpr, WINDOW: tl.constexpr,
    BLOCK_N: tl.constexpr = 64, D: tl.constexpr = 64,
    HQ: tl.constexpr = 16, GROUP: tl.constexpr = 8,
):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    n = tl.load(LENGTHS + batch * LEN_STRIDE)
    window_start = tl.maximum(0, n - WINDOW)
    window_length = n - window_start
    # Both stages cover the complete window with the same32-token geometry.
    split_size = tl.cdiv(tl.cdiv(window_length, SPLITS), 32) * 32
    start = window_start + split * split_size
    end = tl.minimum(start + split_size, n)
    if start < end:
        heads = kv_head * GROUP + tl.arange(0, GROUP)
        dim = tl.arange(0, D)
        # Q[8,64] and K[64,64] stay at the permitted original E4M3 values.
        q = tl.load(Q + batch * Q_BATCH + heads[:, None] * Q_HEAD + dim[None, :])
        maximum = tl.full((GROUP,), -float("inf"), tl.float32)
        denominator = tl.zeros((GROUP,), tl.float32)
        accumulator = tl.zeros((D, GROUP), tl.float32)
        for tile in tl.range(start, end, BLOCK_N):
            logical_k = tile + tl.arange(0, BLOCK_N)
            valid_k = logical_k < end
            physical_k = tl.load(
                TABLE + batch * TABLE_BATCH + logical_k * TABLE_PAGE,
                mask=valid_k, other=0,
            ).to(tl.int64)
            k = tl.load(
                K + physical_k[:, None] * K_SLOT + kv_head * K_HEAD + dim[None, :],
                mask=valid_k[:, None], other=0.0,
            )
            # M64/N8 permits native Hopper FP8 WGMMA; D64 is two K32 partials.
            score = tl.dot(
                k, tl.trans(q), out_dtype=tl.float32, max_num_imprecise_acc=32,
            ) * SM_SCALE
            score = tl.where(valid_k[:, None], score, -float("inf"))
            new_maximum = tl.maximum(maximum, tl.max(score, axis=0))
            rescale = tl.exp(maximum - new_maximum)
            probability = tl.exp(score - new_maximum[None, :])
            # A live split/tile has a real key; every one of the8 heads is real.
            probability = tl.where(valid_k[:, None], probability, 0.0)
            v = tl.load(
                V + physical_k[:, None] * V_SLOT + kv_head * V_HEAD + dim[None, :],
                mask=valid_k[:, None], other=0.0,
            ).to(tl.bfloat16)
            accumulator = accumulator * rescale[None, :]
            accumulator = tl.dot(
                tl.trans(v), probability.to(tl.bfloat16), accumulator,
                out_dtype=tl.float32,
            )
            denominator = denominator * rescale + tl.sum(probability, axis=0)
            maximum = new_maximum
        partial_idx = ((batch * HQ + heads[:, None]) * SPLITS + split) * D + dim[None, :]
        tl.store(PARTIAL + partial_idx, tl.trans(accumulator) / denominator[:, None])
        tl.store(LSE + (batch * HQ + heads) * SPLITS + split, maximum + tl.log(denominator))


@triton.jit
def _transposed_window_gqa_parallel_merge(
    PARTIAL, LSE, LENGTHS, OUT,
    LEN_STRIDE: tl.constexpr, SPLITS: tl.constexpr, WINDOW: tl.constexpr,
    BLOCK_S: tl.constexpr, D: tl.constexpr = 64, HQ: tl.constexpr = 16,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    n = tl.load(LENGTHS + batch * LEN_STRIDE)
    window_start = tl.maximum(0, n - WINDOW)
    window_length = n - window_start
    split_size = tl.cdiv(tl.cdiv(window_length, SPLITS), 32) * 32
    split = tl.arange(0, BLOCK_S)
    start = window_start + split * split_size
    end = tl.minimum(start + split_size, n)
    live = (split < SPLITS) & (start < end)
    # Padded/empty split slots are masked BEFORE either scratch load.
    log = tl.load(LSE + (batch * HQ + head) * SPLITS + split,
                  mask=live, other=-float("inf"))
    has_any = tl.sum(live.to(tl.int32), axis=0) > 0
    maximum = tl.where(has_any, tl.max(log, axis=0), 0.0)
    weight = tl.exp(log - maximum)
    dim = tl.arange(0, D)
    value = tl.load(
        PARTIAL + ((batch * HQ + head) * SPLITS + split[:, None]) * D + dim[None, :],
        mask=live[:, None], other=0.0,
    )
    numerator = tl.sum(value * weight[:, None], axis=0)
    denominator = tl.sum(weight, axis=0)
    safe_denominator = tl.where(denominator > 0.0, denominator, 1.0)
    tl.store(OUT + (batch * HQ + head) * D + dim, numerator / safe_denominator)


def _window_sm_count(device):
    # Stable hardware metadata only; no input tensor values or actual plans are cached.
    count = _OWN_WINDOW_SMS.get(device)
    if count is None:
        count = torch.cuda.get_device_properties(device).multi_processor_count
        _OWN_WINDOW_SMS[device] = count
    return count


def _own_window_compile_diagnostic(stage1, stage2, batch, sm_count, splits, capacity):
    """Only own returned compiled objects; once per static compiled specialization."""
    key = (splits, getattr(stage1, "hash", None), getattr(stage2, "hash", None))
    if key in _OWN_WINDOW_DIAGNOSTIC_KEYS:
        return
    import re
    ptx = stage1.asm.get("ptx", "").lower()
    native_qk = bool(re.search(r"\bwgmma\.mma_async[^;\n]*\.e4m3\.e4m3\b", ptx))
    bf16_pv = bool(re.search(r"\b(?:wgmma\.mma_async|mma\.sync)[^;\n]*\.bf16\.bf16\b", ptx))
    def resources(kernel):
        return dict(registers_per_thread=getattr(kernel, "n_regs", None),
                    shared_bytes=getattr(kernel.metadata, "shared", None))
    print("OWN_SLIDING_L1_PARALLELISM_COMPILE", dict(
        triton_version=triton.__version__, native_e4m3_wgmma_qk=native_qk,
        bf16_tensorcore_pv=bf16_pv, stage1=resources(stage1), stage2=resources(stage2),
        batch=batch, sm_count=sm_count, window_capacity=capacity, splits=splits,
        cta_target_per_sm=4, occupancy_queried=False,
        policy="max(1,min(ceil(4*SM/(B*2)),ceil(SM/2),ceil(window_capacity/32)))",
    ))
    _OWN_WINDOW_DIAGNOSTIC_KEYS.add(key)


def _try_transposed_window_decode(data, q, cfg):
    batch = cfg["batch"]
    if not (
        1 <= batch <= 128 and cfg["q_len"] == 1 and cfg["window"] == 2048
        and cfg["kv_scale"] == 1.0 and cfg["page_size"] == 1
        and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2
        and cfg["head_dim"] == 64
    ):
        return None
    k, v = data["k_cache"], data["v_cache"]
    if k.ndim != 4 or v.ndim != 4 or k.stride(-1) != 1 or v.stride(-1) != 1:
        return None
    table, lengths = data["page_table"], data["cache_seqlens"]
    capacity = min(cfg["window"], table.shape[1])
    if capacity == 0:
        return None
    sm_count = _window_sm_count(q.device)
    splits = max(1, min(triton.cdiv(4 * sm_count, batch * 2),
                        triton.cdiv(sm_count, 2), triton.cdiv(capacity, 32)))
    partial = torch.empty((batch, 16, splits, 64), dtype=torch.float32, device=q.device)
    lse = torch.empty((batch, 16, splits), dtype=torch.float32, device=q.device)
    out = torch.empty((batch, 16, 64), dtype=torch.bfloat16, device=q.device)
    first = _transposed_window_gqa_stage1[(batch, 2, splits)](
        q, k, v, table, lengths, partial, lse,
        Q_BATCH=q.stride(0), Q_HEAD=q.stride(1),
        K_SLOT=k.stride(0), K_HEAD=k.stride(2),
        V_SLOT=v.stride(0), V_HEAD=v.stride(2),
        TABLE_BATCH=table.stride(0), TABLE_PAGE=table.stride(1), LEN_STRIDE=lengths.stride(0),
        SPLITS=splits, SM_SCALE=cfg["sm_scale"] * cfg["qk_scale_factor"], WINDOW=cfg["window"],
        num_warps=4, num_stages=2,
    )
    second = _transposed_window_gqa_parallel_merge[(batch, 16)](
        partial, lse, lengths, out,
        LEN_STRIDE=lengths.stride(0), SPLITS=splits, WINDOW=cfg["window"],
        BLOCK_S=triton.next_power_of_2(splits), num_warps=4, num_stages=2,
    )
    _own_window_compile_diagnostic(first, second, batch, sm_count, splits, capacity)
    return out


# Specialized dense projection paths.
# Vendor Hopper GEMV/TinyGEMM are pinned SGLang/FlashInfer Apache-2.0 APIs.
# General shapes use the native _gemm fallback.
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


def _dense_project(x, w, layer, state):
    """Selected verified BF16 paths; no router or shared backend-state change."""
    m, n = x.shape[0], w.shape[0]
    if m == 1 and layer in ("qkv_gate", "pooler"):
        return hopper_bf16_gemv(x, w)
    if layer == "unpooler" and 1 < m <= 8:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        tinygemm_bf16(x, w, out, use_pdl=False)
        return out
    if (layer == "unpooler" and m > 8) or (layer == "pooler" and m > 128):
        return torch.mm(x, state["_dense_transposed"][layer][:, :n])
    if m == 1 and layer in ("o", "unpooler"):
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
    return _gemm(x, w)


# Parallel integer route alignment for 1,024 < routes <= 2,048 and BM16.
# Fresh relative expert IDs address the physical local expert bank.
@triton.jit
def _parallel_alignment_map_counts(
    GLOBAL_IDS, LOCAL_IDS, COUNTS,
    ROUTES: tl.constexpr, LOCAL_START: tl.constexpr,
    EXPERTS: tl.constexpr, ROUTE_TILE: tl.constexpr,
):
    tl.static_assert(EXPERTS == 32 and ROUTE_TILE == 256)
    tile = tl.program_id(0)
    route = tile * ROUTE_TILE + tl.arange(0, ROUTE_TILE)
    g = tl.load(GLOBAL_IDS + route, mask=route < ROUTES, other=-1).to(tl.int32)
    local = g - LOCAL_START
    live = (route < ROUTES) & (local >= 0) & (local < EXPERTS)
    tl.store(LOCAL_IDS + route, tl.where(live, local, -1), mask=route < ROUTES)
    hist = tl.histogram(tl.where(live, local, EXPERTS), 64)
    expert = tl.arange(0, EXPERTS)
    count = tl.gather(hist, expert, axis=0)
    tl.store(COUNTS + tile * EXPERTS + expert, count)


@triton.jit
def _parallel_alignment_stable_scatter(
    LOCAL_IDS, COUNTS, SORTED_IDS, BLOCK_EXPERTS, N_PADDED,
    ROUTES: tl.constexpr, EXPERTS: tl.constexpr,
    ROUTE_TILE: tl.constexpr, TILES: tl.constexpr,
    BM: tl.constexpr, CAPACITY: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_D: tl.constexpr,
):
    tl.static_assert(EXPERTS == 32 and ROUTE_TILE == 256 and BM == 16)
    expert = tl.program_id(0)
    tile = tl.program_id(1)
    tile_count = tl.load(COUNTS + tile * EXPERTS + expert)
    # Tile0 owns expert metadata even if its own tile has no matches.
    if (tile == 0) | (tile_count > 0):
        tiles = tl.arange(0, BLOCK_T)
        experts = tl.arange(0, EXPERTS)
        h = tl.load(COUNTS + tiles[:, None] * EXPERTS + experts[None, :],
                    mask=tiles[:, None] < TILES, other=0)
        counts = tl.sum(h, axis=0)
        padded = tl.cdiv(counts, BM) * BM
        bases = tl.cumsum(padded, axis=0) - padded
        prior = tl.sum(tl.where(tiles[:, None] < tile, h, 0), axis=0)
        base = tl.sum(tl.where(experts == expert, bases, 0), axis=0)
        count = tl.sum(tl.where(experts == expert, counts, 0), axis=0)
        pad = tl.sum(tl.where(experts == expert, padded, 0), axis=0)
        carry = tl.sum(tl.where(experts == expert, prior, 0), axis=0)
        route = tile * ROUTE_TILE + tl.arange(0, ROUTE_TILE)
        local = tl.load(LOCAL_IDS + route, mask=route < ROUTES, other=-1)
        match = (route < ROUTES) & (local == expert)
        rank = tl.cumsum(match.to(tl.int32), axis=0) - match.to(tl.int32)
        tl.store(SORTED_IDS + base + carry + rank, route, mask=match)
        if tile == 0:
            # Padding, active descriptors and real routes have disjoint owners.
            p = tl.arange(0, BM)
            tl.store(SORTED_IDS + base + count + p, ROUTES, mask=p < pad - count)
            d = tl.arange(0, BLOCK_D)
            tl.store(BLOCK_EXPERTS + base // BM + d, expert, mask=d < pad // BM)
            if expert == 0:
                total = tl.sum(padded, axis=0)
                tl.store(N_PADDED, total)
                tail = tl.arange(0, BLOCK_P)
                tl.store(SORTED_IDS + tail, ROUTES,
                         mask=(tail >= total) & (tail < CAPACITY))
                tl.store(BLOCK_EXPERTS + d, -1,
                         mask=(d >= total // BM) & (d < CAPACITY // BM))


_OWN_PARALLEL_ALIGNMENT_COMPILES = set()


def _parallel_alignment_diagnostic(first, second, routes, local_start, block_size_m, capacity, tiles):
    """Read only own normal compiled producer objects, once per compiled pair."""
    key = (first.hash, second.hash)
    if key in _OWN_PARALLEL_ALIGNMENT_COMPILES:
        return
    def resources(kernel):
        return dict(registers_per_thread=getattr(kernel, "n_regs", None),
                    shared_bytes=getattr(kernel.metadata, "shared", None),
                    num_warps=getattr(kernel.metadata, "num_warps", None),
                    num_stages=getattr(kernel.metadata, "num_stages", None))
    print("OWN_MOE_PARALLEL_ALIGNMENT_COMPILE", dict(
        triton_version=triton.__version__, routes=routes, local_start=local_start,
        experts=32, route_tile=256, block_size_m=block_size_m, capacity=capacity,
        count_shape=(tiles, 32), count_bytes=tiles * 32 * 4,
        stage_a_grid=(tiles,), stage_b_grid=(32, tiles),
        block_t=triton.next_power_of_2(tiles),
        block_p=triton.next_power_of_2(capacity),
        block_d=triton.next_power_of_2(capacity // block_size_m),
        stage_a=resources(first), stage_b=resources(second),
        ordering="ordinary_same_stream_launch_dependency", stable_original_route_indices=True,
    ))
    _OWN_PARALLEL_ALIGNMENT_COMPILES.add(key)


def _launch_parallel_alignment(ids, local_ids, sorted_ids, block_experts, n_padded,
                               local_start, local_count, block_size_m, capacity):
    # Only shape/config-dependent allocation; scratch and input-derived metadata
    # are fresh for each timed call, with no host read of routing values.
    routes = ids.numel()
    tiles = triton.cdiv(routes, 256)
    counts = torch.empty((tiles, local_count), device=ids.device, dtype=torch.int32)
    first = _parallel_alignment_map_counts[(tiles,)](
        ids, local_ids, counts,
        ROUTES=routes, LOCAL_START=local_start, EXPERTS=local_count, ROUTE_TILE=256,
        num_warps=4, num_stages=1,
    )
    second = _parallel_alignment_stable_scatter[(local_count, tiles)](
        local_ids, counts, sorted_ids, block_experts, n_padded,
        ROUTES=routes, EXPERTS=local_count, ROUTE_TILE=256, TILES=tiles,
        BM=block_size_m, CAPACITY=capacity,
        BLOCK_T=triton.next_power_of_2(tiles),
        BLOCK_P=triton.next_power_of_2(capacity),
        BLOCK_D=triton.next_power_of_2(capacity // block_size_m),
        num_warps=4, num_stages=1,
    )
    _parallel_alignment_diagnostic(first, second, routes, local_start, block_size_m, capacity, tiles)
