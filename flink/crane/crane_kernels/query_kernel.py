import torch
import triton
import triton.language as tl


@triton.jit
def _fused_query_min_fwd_kernel(
    source_emb_ptr,
    dest_emb_ptr,
    memory_ptr,
    ratio_min_ptr,
    argmin_ptr,
    activated_memory_dim,
    L: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    RC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)

    if pid_l > activated_memory_dim:
        out_offset = pid_b * L + pid_l
        tl.store(ratio_min_ptr + out_offset, 0.0)
        tl.store(argmin_ptr + out_offset, 0)
        return

    src_base = pid_b * L * R + pid_l * R
    dst_base = pid_b * L * C + pid_l * C
    mem_base = pid_l * R * C

    # Prefetch source and dest vectors into L1 cache
    src = tl.load(source_emb_ptr + src_base + tl.arange(0, R))
    dst = tl.load(dest_emb_ptr + dst_base + tl.arange(0, C))

    current_min = float('inf')
    current_argmin = tl.zeros([], dtype=tl.int32)
    has_valid = tl.zeros([], dtype=tl.int32)

    for block_start in tl.static_range(0, RC, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < RC
        r_idx = offsets // C
        c_idx = offsets % C

        s_vals = tl.load(source_emb_ptr + src_base + r_idx, mask=mask, other=0.0)
        d_vals = tl.load(dest_emb_ptr + dst_base + c_idx, mask=mask, other=0.0)
        outer = s_vals * d_vals

        mem_vals = tl.load(memory_ptr + mem_base + offsets, mask=mask, other=0.0)

        nz = (outer != 0.0) & mask
        ratio = tl.where(nz, mem_vals / outer, float('inf'))

        block_min = tl.min(ratio)
        if block_min < current_min:
            current_min = block_min
            current_argmin = tl.argmin(ratio, axis=0).to(tl.int32) + block_start

        has_valid = has_valid | tl.where(tl.sum(nz.to(tl.int32)) > 0, 1, 0)

    out_offset = pid_b * L + pid_l
    result = tl.where(has_valid > 0, current_min, 0.0)
    tl.store(ratio_min_ptr + out_offset, result)
    tl.store(argmin_ptr + out_offset, current_argmin)


@triton.jit
def _fused_query_min_bwd_kernel(
    grad_out_ptr,
    source_emb_ptr,
    dest_emb_ptr,
    memory_ptr,
    argmin_ptr,
    grad_source_ptr,
    grad_dest_ptr,
    activated_memory_dim,
    L: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)

    out_offset = pid_b * L + pid_l

    if pid_l > activated_memory_dim:
        return

    grad_val = tl.load(grad_out_ptr + out_offset)
    flat_idx = tl.load(argmin_ptr + out_offset)

    r_star = flat_idx // C
    c_star = flat_idx % C

    src_base = pid_b * L * R + pid_l * R
    dst_base = pid_b * L * C + pid_l * C
    mem_base = pid_l * R * C

    s_val = tl.load(source_emb_ptr + src_base + r_star)
    d_val = tl.load(dest_emb_ptr + dst_base + c_star)
    m_val = tl.load(memory_ptr + mem_base + flat_idx)

    outer = s_val * d_val
    is_nz = outer != 0.0

    # d(m / (s*d)) / ds = -m / (s^2 * d)
    grad_s = tl.where(is_nz, grad_val * (-m_val) / (s_val * s_val * d_val), 0.0)
    # d(m / (s*d)) / dd = -m / (s * d^2)
    grad_d = tl.where(is_nz, grad_val * (-m_val) / (s_val * d_val * d_val), 0.0)

    tl.atomic_add(grad_source_ptr + src_base + r_star, grad_s)
    tl.atomic_add(grad_dest_ptr + dst_base + c_star, grad_d)


@triton.jit
def _fused_query_min_inference_kernel(
    source_emb_ptr,
    dest_emb_ptr,
    memory_ptr,
    ratio_min_ptr,
    activated_memory_dim,
    L: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    RC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Inference-only kernel: no argmin, no backward support."""
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)

    if pid_l > activated_memory_dim:
        tl.store(ratio_min_ptr + pid_b * L + pid_l, 0.0)
        return

    src_base = pid_b * L * R + pid_l * R
    dst_base = pid_b * L * C + pid_l * C
    mem_base = pid_l * R * C

    current_min = float('inf')
    has_valid = tl.zeros([], dtype=tl.int32)

    for block_start in tl.static_range(0, RC, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < RC
        r_idx = offsets // C
        c_idx = offsets % C

        s_vals = tl.load(source_emb_ptr + src_base + r_idx, mask=mask, other=0.0)
        d_vals = tl.load(dest_emb_ptr + dst_base + c_idx, mask=mask, other=0.0)
        outer = s_vals * d_vals

        mem_vals = tl.load(memory_ptr + mem_base + offsets, mask=mask, other=0.0)

        nz = (outer != 0.0) & mask
        ratio = tl.where(nz, mem_vals / outer, float('inf'))

        block_min = tl.min(ratio)
        current_min = tl.minimum(current_min, block_min)
        has_valid = has_valid | tl.where(tl.sum(nz.to(tl.int32)) > 0, 1, 0)

    out_offset = pid_b * L + pid_l
    result = tl.where(has_valid > 0, current_min, 0.0)
    tl.store(ratio_min_ptr + out_offset, result)


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class FusedQueryMin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source_emb, dest_emb, memory_matrix, activated_memory_dim):
        B, L, R = source_emb.shape
        C = dest_emb.shape[2]
        RC = R * C

        ratio_min = torch.empty(B, L, device=source_emb.device, dtype=source_emb.dtype)
        argmin_indices = torch.empty(B, L, device=source_emb.device, dtype=torch.int32)

        BLOCK_SIZE = min(_next_power_of_2(RC), 4096)

        grid = (B, L)
        _fused_query_min_fwd_kernel[grid](
            source_emb, dest_emb, memory_matrix,
            ratio_min, argmin_indices,
            activated_memory_dim,
            L=L, R=R, C=C, RC=RC,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(source_emb, dest_emb, memory_matrix, argmin_indices)
        ctx.activated_memory_dim = activated_memory_dim
        ctx.B = B
        ctx.L = L
        ctx.R = R
        ctx.C = C
        return ratio_min

    @staticmethod
    def backward(ctx, grad_output):
        source_emb, dest_emb, memory_matrix, argmin_indices = ctx.saved_tensors
        B, L, R, C = ctx.B, ctx.L, ctx.R, ctx.C

        grad_source = torch.zeros_like(source_emb)
        grad_dest = torch.zeros_like(dest_emb)

        grid = (B, L)
        _fused_query_min_bwd_kernel[grid](
            grad_output.contiguous(), source_emb, dest_emb, memory_matrix,
            argmin_indices, grad_source, grad_dest,
            ctx.activated_memory_dim,
            L=L, R=R, C=C,
        )

        return grad_source, grad_dest, None, None


def fused_query_min(source_emb, dest_emb, memory_matrix, activated_memory_dim):
    return FusedQueryMin.apply(source_emb, dest_emb, memory_matrix, activated_memory_dim)


def fused_query_min_inference(source_emb, dest_emb, memory_matrix, activated_memory_dim):
    """Inference-only: skips argmin computation, no backward support."""
    B, L, R = source_emb.shape
    C = dest_emb.shape[2]
    RC = R * C

    ratio_min = torch.empty(B, L, device=source_emb.device, dtype=source_emb.dtype)

    BLOCK_SIZE = min(_next_power_of_2(RC), 4096)

    grid = (B, L)
    _fused_query_min_inference_kernel[grid](
        source_emb, dest_emb, memory_matrix,
        ratio_min,
        activated_memory_dim,
        L=L, R=R, C=C, RC=RC,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return ratio_min


# ── LBR-layout inference kernel ──────────────────────────────
# Accepts source_emb/dest_emb in [L, B, R/C] layout (contiguous
# from BMM output) and any dtype (FP16/FP32).  Widens to FP32 at
# register level before division so memory_matrix stays FP32.
# This eliminates the expensive permute+contiguous+float() step
# that the BLR kernel requires.

@triton.jit
def _fused_query_min_inference_lbr_kernel(
    source_emb_ptr,
    dest_emb_ptr,
    memory_ptr,
    ratio_min_ptr,
    activated_memory_dim,
    stride_src_l,  # B * R (runtime, avoids recompile per batch size)
    stride_dst_l,  # B * C
    L: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    RC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Inference kernel for [L, B, R] layout embeddings."""
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)

    if pid_l > activated_memory_dim:
        tl.store(ratio_min_ptr + pid_b * L + pid_l, 0.0)
        return

    # [L, B, R] layout: stride_src_l = B*R, stride_dst_l = B*C
    src_base = pid_l * stride_src_l + pid_b * R
    dst_base = pid_l * stride_dst_l + pid_b * C
    mem_base = pid_l * R * C

    current_min = float('inf')
    has_valid = tl.zeros([], dtype=tl.int32)

    for block_start in tl.static_range(0, RC, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < RC
        r_idx = offsets // C
        c_idx = offsets % C

        # Load embeddings (may be FP16) and widen to FP32 for division
        s_vals = tl.load(source_emb_ptr + src_base + r_idx, mask=mask, other=0.0).to(tl.float32)
        d_vals = tl.load(dest_emb_ptr + dst_base + c_idx, mask=mask, other=0.0).to(tl.float32)
        outer = s_vals * d_vals

        mem_vals = tl.load(memory_ptr + mem_base + offsets, mask=mask, other=0.0)

        nz = (outer != 0.0) & mask
        ratio = tl.where(nz, mem_vals / outer, float('inf'))

        block_min = tl.min(ratio)
        current_min = tl.minimum(current_min, block_min)
        has_valid = has_valid | tl.where(tl.sum(nz.to(tl.int32)) > 0, 1, 0)

    out_offset = pid_b * L + pid_l
    result = tl.where(has_valid > 0, current_min, 0.0)
    tl.store(ratio_min_ptr + out_offset, result)


def fused_query_min_inference_lbr(source_emb, dest_emb, memory_matrix,
                                  activated_memory_dim, out=None):
    """Inference-only query for [L, B, R/C] layout embeddings (any dtype).

    Args:
        source_emb: [L, B, R] tensor (contiguous, FP16 or FP32)
        dest_emb:   [L, B, C] tensor (contiguous, FP16 or FP32)
        memory_matrix: [L, R, C] tensor (FP32)
        activated_memory_dim: int
        out: optional pre-allocated [B, L] FP32 output tensor
    Returns:
        ratio_min: [B, L] FP32 tensor
    """
    L, B, R = source_emb.shape
    C = dest_emb.shape[2]
    RC = R * C

    if out is not None:
        ratio_min = out[:B]
    else:
        ratio_min = torch.empty(B, L, device=source_emb.device, dtype=torch.float32)

    BLOCK_SIZE = min(_next_power_of_2(RC), 4096)

    grid = (B, L)
    _fused_query_min_inference_lbr_kernel[grid](
        source_emb, dest_emb, memory_matrix,
        ratio_min,
        activated_memory_dim,
        B * R,  # stride_src_l
        B * C,  # stride_dst_l
        L=L, R=R, C=C, RC=RC,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return ratio_min
