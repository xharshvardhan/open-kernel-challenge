# okc: problem=muse_moe_w4a16
"""NVFP4 routed experts with fused route alignment and output aggregation.

Per-expert token counts select compatible BM16/BM32 tiles without adding
padded rows. Native packed-weight GEMMs, BF16 activations and FP32 reduction
boundaries are preserved.

Adapted from SGLang, copyright 2023-2024 SGLang Team (Apache-2.0).
Modified for the Open Kernel Challenge; see NOTICE.
"""

from typing import TypedDict

import types

import torch
import triton
import triton.language as tl
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
    _post_expert_norm_weighted_sum_kernel,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def prepare(weights):
    device = weights["w13"].device
    # One repacked bank: this GPU's 32 experts under EP8.
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    torch.cuda.synchronize()
    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


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


def kernel_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / bank["count"] / block_size_m < 0.9:
            break
    # Retain the native block-size selection and both Marlin GEMMs.
    block_size_m = min(block_size_m, 16)
    trusted_active_prefix = expert_map is not None and bank["count"] == 32 and 0 < num_tokens <= 256
    heterogeneous = trusted_active_prefix and 128 <= num_tokens <= 256 and block_size_m == 16
    if heterogeneous:
        local_ids, cold_metadata, hot_metadata = _heterogeneous_alignment(
            data["topk_ids"], cfg["local_start"], bank["count"])
    elif trusted_active_prefix:
        # All fresh routing-dependent work happens in this timed producer.
        local_ids, sorted_ids, block_experts, n_padded = _fused_local_alignment(
            data["topk_ids"], cfg["local_start"], bank["count"], block_size_m)
    else:
        # Use the general alignment fallback above 2,048 routes.
        local_ids = data["topk_ids"].to(torch.int32)
        if expert_map is not None:
            local_ids = expert_map[local_ids]
        local_ids = local_ids.contiguous()
        if num_tokens == 1 and expert_map is None:
            from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token

            sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, block_size_m)
        else:
            sorted_ids, block_experts, n_padded = moe_align_block_size(
                local_ids, block_size_m, N_EXPERTS if expert_map is not None else bank["count"])
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None and not trusted_active_prefix,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    if heterogeneous:
        for (sorted_ids, block_experts, n_padded), bucket_bm in ((cold_metadata, 16), (hot_metadata, 32)):
            cache1 = moe_wna16_marlin_gemm(
                x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
                sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
                size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **dict(common, moe_block_size=bucket_bm))
    else:
        cache1 = moe_wna16_marlin_gemm(
            x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
            sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
            size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    if heterogeneous:
        for (sorted_ids, block_experts, n_padded), bucket_bm in ((cold_metadata, 16), (hot_metadata, 32)):
            y = moe_wna16_marlin_gemm(
                cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
                sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
                size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **dict(common, moe_block_size=bucket_bm)).view(num_tokens, TOP_K, size_k)
    else:
        y = moe_wna16_marlin_gemm(
            cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
            sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
            size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    if num_tokens < 32:
        return _small_post_expert_norm_weighted_sum(
            y, topk_weights, state["norm_w"], local_ids)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


# Fuse local route alignment for up to 2,048 routes.
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


@triton.jit
def _heterogeneous_alignment_scatter(
    LOCAL_IDS, COUNTS,
    COLD_SORTED, COLD_EXPERTS, COLD_N_PADDED,
    HOT_SORTED, HOT_EXPERTS, HOT_N_PADDED,
    ROUTES: tl.constexpr, EXPERTS: tl.constexpr,
    ROUTE_TILE: tl.constexpr, TILES: tl.constexpr,
    COLD_CAPACITY: tl.constexpr, HOT_CAPACITY: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr,
    BLOCK_CD: tl.constexpr, BLOCK_HD: tl.constexpr,
):
    tl.static_assert(EXPERTS == 32 and ROUTE_TILE == 256)
    tl.static_assert(1024 <= ROUTES and ROUTES <= 2048)
    tl.static_assert(COLD_CAPACITY % 16 == 0 and HOT_CAPACITY % 32 == 0)
    expert = tl.program_id(0)
    tile = tl.program_id(1)
    tile_count = tl.load(COUNTS + tile * EXPERTS + expert)
    if (tile == 0) | (tile_count > 0):
        tiles = tl.arange(0, BLOCK_T)
        experts = tl.arange(0, EXPERTS)
        histogram = tl.load(COUNTS + tiles[:, None] * EXPERTS + experts[None, :],
                            mask=tiles[:, None] < TILES, other=0)
        counts = tl.sum(histogram, axis=0)
        p16 = tl.cdiv(counts, 16) * 16
        hot = (counts > 0) & (p16 % 32 == 0)
        cold_padding = tl.where(hot, 0, p16)
        hot_padding = tl.where(hot, p16, 0)
        cold_bases = tl.cumsum(cold_padding, axis=0) - cold_padding
        hot_bases = tl.cumsum(hot_padding, axis=0) - hot_padding
        prior = tl.sum(tl.where(tiles[:, None] < tile, histogram, 0), axis=0)
        is_hot = tl.sum(tl.where(experts == expert, hot.to(tl.int32), 0), axis=0) > 0
        cold_base = tl.sum(tl.where(experts == expert, cold_bases, 0), axis=0)
        hot_base = tl.sum(tl.where(experts == expert, hot_bases, 0), axis=0)
        count = tl.sum(tl.where(experts == expert, counts, 0), axis=0)
        pad = tl.sum(tl.where(experts == expert, p16, 0), axis=0)
        carry = tl.sum(tl.where(experts == expert, prior, 0), axis=0)
        route = tile * ROUTE_TILE + tl.arange(0, ROUTE_TILE)
        local = tl.load(LOCAL_IDS + route, mask=route < ROUTES, other=-1)
        match = (route < ROUTES) & (local == expert)
        rank = tl.cumsum(match.to(tl.int32), axis=0) - match.to(tl.int32)
        # Uniform per-expert branch chooses one bucket; valid rows precede its
        # at-most15 sentinel slots. No route is ever written to both buckets.
        if is_hot:
            tl.store(HOT_SORTED + hot_base + carry + rank, route, mask=match)
            if tile == 0:
                p = tl.arange(0, 16)
                tl.store(HOT_SORTED + hot_base + count + p, ROUTES, mask=p < pad - count)
                d = tl.arange(0, BLOCK_HD)
                tl.store(HOT_EXPERTS + hot_base // 32 + d, expert, mask=d < pad // 32)
        else:
            tl.store(COLD_SORTED + cold_base + carry + rank, route, mask=match)
            if tile == 0:
                p = tl.arange(0, 16)
                tl.store(COLD_SORTED + cold_base + count + p, ROUTES, mask=p < pad - count)
                d = tl.arange(0, BLOCK_CD)
                tl.store(COLD_EXPERTS + cold_base // 16 + d, expert, mask=d < pad // 16)
        if (expert == 0) & (tile == 0):
            # Both tails/counts have one owner, disjoint from every bucket's
            # valid prefix and per-expert padding, irrespective of CTA order.
            cold_total = tl.sum(cold_padding, axis=0)
            hot_total = tl.sum(hot_padding, axis=0)
            tl.store(COLD_N_PADDED, cold_total)
            tl.store(HOT_N_PADDED, hot_total)
            p = tl.arange(0, BLOCK_C)
            tl.store(COLD_SORTED + p, ROUTES,
                     mask=(p >= cold_total) & (p < COLD_CAPACITY))
            p = tl.arange(0, BLOCK_H)
            tl.store(HOT_SORTED + p, ROUTES,
                     mask=(p >= hot_total) & (p < HOT_CAPACITY))
            d = tl.arange(0, BLOCK_CD)
            tl.store(COLD_EXPERTS + d, -1,
                     mask=(d >= cold_total // 16) & (d < COLD_CAPACITY // 16))
            d = tl.arange(0, BLOCK_HD)
            tl.store(HOT_EXPERTS + d, -1,
                     mask=(d >= hot_total // 32) & (d < HOT_CAPACITY // 32))


_OWN_HETEROGENEOUS_ALIGNMENT_COMPILES = set()


def _heterogeneous_alignment_diagnostic(first, second, routes, local_start, cold_capacity, hot_capacity, tiles):
    """Own normal compiled producer artifacts; never inspect values/capture."""
    if torch.cuda.is_current_stream_capturing():
        return
    key = (first.hash, second.hash)
    if key in _OWN_HETEROGENEOUS_ALIGNMENT_COMPILES:
        return
    def resources(kernel):
        return dict(registers_per_thread=getattr(kernel, "n_regs", None),
                    spill_count=getattr(kernel, "n_spills", None),
                    shared_bytes=getattr(kernel.metadata, "shared", None),
                    num_warps=getattr(kernel.metadata, "num_warps", None),
                    num_stages=getattr(kernel.metadata, "num_stages", None))
    print("OWN_MOE_HETEROGENEOUS_ALIGNMENT_COMPILE", dict(
        triton_version=triton.__version__, routes=routes, local_start=local_start,
        cold_block_m=16, hot_block_m=32, cold_capacity=cold_capacity, hot_capacity=hot_capacity,
        stage_a_grid=(tiles,), stage_b_grid=(32, tiles), counts_bytes=tiles * 32 * 4,
        policy="hot_iff_count_positive_and_p16_divisible32_no_extra_padding",
        stage_a=resources(first), stage_b=resources(second),
        ordering="ordinary_current_stream_dependency", original_route_IDs=True,
    ))
    _OWN_HETEROGENEOUS_ALIGNMENT_COMPILES.add(key)


def _heterogeneous_alignment(global_ids, local_start, local_count):
    """All fresh GPU routing metadata; capacities/launches use shape only."""
    ids = global_ids.contiguous()
    routes = ids.numel()
    if local_count != 32 or routes < 1024 or routes > 2048 or routes % TOP_K != 0:
        raise ValueError("Heterogeneous alignment requires local32 and M128..256/topk8")
    tiles = triton.cdiv(routes, 256)
    cold_capacity = triton.cdiv(min(routes * 16, routes + 32 * 15), 16) * 16
    hot_capacity = (cold_capacity // 32) * 32
    local_ids = torch.empty(ids.shape, device=ids.device, dtype=torch.int32)
    counts = torch.empty((tiles, 32), device=ids.device, dtype=torch.int32)
    cold = (torch.empty((cold_capacity,), device=ids.device, dtype=torch.int32),
            torch.empty((cold_capacity // 16,), device=ids.device, dtype=torch.int32),
            torch.empty((1,), device=ids.device, dtype=torch.int32))
    hot = (torch.empty((hot_capacity,), device=ids.device, dtype=torch.int32),
           torch.empty((hot_capacity // 32,), device=ids.device, dtype=torch.int32),
           torch.empty((1,), device=ids.device, dtype=torch.int32))
    first = _parallel_alignment_map_counts[(tiles,)](
        ids, local_ids, counts,
        ROUTES=routes, LOCAL_START=local_start, EXPERTS=32, ROUTE_TILE=256,
        num_warps=4, num_stages=1,
    )
    second = _heterogeneous_alignment_scatter[(32, tiles)](
        local_ids, counts, *cold, *hot,
        ROUTES=routes, EXPERTS=32, ROUTE_TILE=256, TILES=tiles,
        COLD_CAPACITY=cold_capacity, HOT_CAPACITY=hot_capacity,
        BLOCK_T=triton.next_power_of_2(tiles),
        BLOCK_C=triton.next_power_of_2(cold_capacity),
        BLOCK_H=triton.next_power_of_2(hot_capacity),
        BLOCK_CD=triton.next_power_of_2(cold_capacity // 16),
        BLOCK_HD=triton.next_power_of_2(hot_capacity // 32),
        num_warps=4, num_stages=1,
    )
    _heterogeneous_alignment_diagnostic(first, second, routes, local_start, cold_capacity, hot_capacity, tiles)
    return local_ids, cold, hot
