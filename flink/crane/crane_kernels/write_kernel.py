import torch
import triton
import triton.language as tl


def fused_write_sum(source_emb, dest_emb, input_y, L, R, C,
                    base_out=None, base_sum_out=None):
    """Compute outer product sums using batched matrix multiply (cuBLAS).

    Equivalent to:
        base_per_layer[l,r,c] = sum_b source_emb[b,l,r] * dest_emb[b,l,c]
        base_sum_per_layer[l,r,c] = sum_b source_emb[b,l,r] * dest_emb[b,l,c] * y[b]

    Uses batched GEMM: [L, R, B] @ [L, B, C] = [L, R, C]
    Pass pre-allocated base_out/base_sum_out to avoid repeated allocation.
    """
    s_t = source_emb.permute(1, 2, 0).contiguous()  # [L, R, B]
    d_p = dest_emb.permute(1, 0, 2).contiguous()    # [L, B, C]

    if base_out is not None:
        torch.bmm(s_t, d_p, out=base_out)
    else:
        base_out = torch.bmm(s_t, d_p)

    sy = source_emb * input_y.view(-1, 1, 1)         # [B, L, R]
    sy_t = sy.permute(1, 2, 0).contiguous()           # [L, R, B]

    if base_sum_out is not None:
        torch.bmm(sy_t, d_p, out=base_sum_out)
    else:
        base_sum_out = torch.bmm(sy_t, d_p)

    return base_out, base_sum_out


# ═══════════════════════════════════════════════════════════════
# Fused Triton write_sum — eliminates permute + contiguous + bmm
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _fused_write_sum_triton_kernel(
    source_ptr, dest_ptr, y_ptr,
    base_ptr, base_sum_ptr,
    B,
    stride_src_b: tl.constexpr,
    stride_src_l: tl.constexpr,
    stride_dst_b: tl.constexpr,
    stride_dst_l: tl.constexpr,
    L: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Compute base[l]=src[:,l]^T @ dst[:,l] and base_sum[l]=(src*y)[:,l]^T @ dst[:,l].

    Reads source[B,L,R] and dest[B,L,C] directly — no permute or contiguous.
    Grid: (L,). Each program handles one layer.
    """
    pid_l = tl.program_id(0)

    acc_base = tl.zeros([R, C], dtype=tl.float32)
    acc_sum = tl.zeros([R, C], dtype=tl.float32)

    for b_start in range(0, B, BLOCK_B):
        b_offs = b_start + tl.arange(0, BLOCK_B)
        b_mask = b_offs < B

        # Load source[b, l, :]: [BLOCK_B, R]
        s_ptrs = source_ptr + b_offs[:, None] * stride_src_b + pid_l * stride_src_l + tl.arange(0, R)[None, :]
        s_vals = tl.load(s_ptrs, mask=b_mask[:, None], other=0.0).to(tl.float32)

        # Load dest[b, l, :]: [BLOCK_B, C]
        d_ptrs = dest_ptr + b_offs[:, None] * stride_dst_b + pid_l * stride_dst_l + tl.arange(0, C)[None, :]
        d_vals = tl.load(d_ptrs, mask=b_mask[:, None], other=0.0).to(tl.float32)

        # Load y[b]: [BLOCK_B]
        y_vals = tl.load(y_ptr + b_offs, mask=b_mask, other=0.0).to(tl.float32)

        # base += src^T @ dst: [R, BLOCK_B] @ [BLOCK_B, C] → [R, C]
        acc_base += tl.dot(tl.trans(s_vals), d_vals)

        # base_sum += (src*y)^T @ dst
        sy = s_vals * y_vals[:, None]
        acc_sum += tl.dot(tl.trans(sy), d_vals)

    # Store [R, C] results
    rc_offs = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
    out_offset = pid_l * R * C
    tl.store(base_ptr + out_offset + rc_offs, acc_base)
    tl.store(base_sum_ptr + out_offset + rc_offs, acc_sum)


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def fused_write_sum_triton(source_emb, dest_emb, input_y, L, R, C,
                           base_out, base_sum_out):
    """Fused write_sum using Triton — no permute/contiguous/bmm overhead."""
    B = source_emb.shape[0]
    BLOCK_B = max(min(_next_power_of_2(B), 64), 16)  # tl.dot requires K >= 16

    _fused_write_sum_triton_kernel[(L,)](
        source_emb, dest_emb, input_y,
        base_out, base_sum_out,
        B,
        stride_src_b=source_emb.stride(0),
        stride_src_l=source_emb.stride(1),
        stride_dst_b=dest_emb.stride(0),
        stride_dst_l=dest_emb.stride(1),
        L=L, R=R, C=C,
        BLOCK_B=BLOCK_B,
        num_stages=1,
    )
    return base_out, base_sum_out


@triton.jit
def _fused_carry_kernel(
    memory_ptr,
    base_ptr,
    base_sum_ptr,
    carry_threshold: tl.constexpr,
    L: tl.constexpr,
    RC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused memory update + carry for all layers in a single kernel.

    Eliminates all CPU sync points. Branching happens on GPU.
    Assumes base_per_layer > 0 everywhere (guaranteed by Softplus).
    """
    # Step 1: memory[0] += base_sum[0]
    for start in tl.static_range(0, RC, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < RC
        m = tl.load(memory_ptr + offs, mask=mask, other=0.0)
        bs = tl.load(base_sum_ptr + offs, mask=mask, other=0.0)
        tl.store(memory_ptr + offs, m + bs, mask=mask)

    # Step 2: Carry for each layer j → j+1
    for j in tl.static_range(0, L - 1):
        j_off = j * RC
        j1_off = (j + 1) * RC

        # Compute min(memory[j] / (threshold * base[j]))
        min_ratio = float('inf')
        for start in tl.static_range(0, RC, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            mask = offs < RC
            m = tl.load(memory_ptr + j_off + offs, mask=mask, other=0.0)
            b = tl.load(base_ptr + j_off + offs, mask=mask, other=1.0)
            ratio = m / (carry_threshold * b)
            block_min = tl.min(tl.where(mask, ratio, float('inf')))
            min_ratio = tl.minimum(min_ratio, block_min)

        # Uniform branch: all threads see the same min_ratio
        if min_ratio >= 1.0:
            for start in tl.static_range(0, RC, BLOCK_SIZE):
                offs = start + tl.arange(0, BLOCK_SIZE)
                mask = offs < RC

                # memory[j] -= threshold * min_ratio * base[j]
                m_j = tl.load(memory_ptr + j_off + offs, mask=mask, other=0.0)
                b_j = tl.load(base_ptr + j_off + offs, mask=mask, other=0.0)
                tl.store(memory_ptr + j_off + offs,
                         m_j - carry_threshold * min_ratio * b_j, mask=mask)

                # memory[j+1] += min_ratio * base[j+1]
                m_j1 = tl.load(memory_ptr + j1_off + offs, mask=mask, other=0.0)
                b_j1 = tl.load(base_ptr + j1_off + offs, mask=mask, other=0.0)
                tl.store(memory_ptr + j1_off + offs,
                         m_j1 + min_ratio * b_j1, mask=mask)


def fused_carry(memory_matrix, base_per_layer, base_sum_per_layer, carry_threshold):
    """Fused memory[0] update + carry across all layers. Single kernel launch, no CPU sync."""
    L, R, C = memory_matrix.shape
    RC = R * C
    BLOCK_SIZE = min(_next_power_of_2(RC), 4096)

    _fused_carry_kernel[(1,)](
        memory_matrix, base_per_layer, base_sum_per_layer,
        carry_threshold=carry_threshold,
        L=L, RC=RC,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# ═══════════════════════════════════════════════════════════════
# Fully fused write_sum + carry — single kernel, no intermediate
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _fused_write_carry_kernel(
    memory_ptr, source_ptr, dest_ptr, y_ptr,
    carry_threshold: tl.constexpr,
    B,
    stride_src_b: tl.constexpr,
    stride_src_l: tl.constexpr,
    stride_dst_b: tl.constexpr,
    stride_dst_l: tl.constexpr,
    L: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Fused write_sum + carry in a single kernel. Grid=(1,).

    Computes outer-product sums per layer and immediately applies carry,
    keeping base/base_sum in registers — no intermediate global buffers.
    """
    RC = R * C
    rc_offs = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]

    # ── Layer 0: compute base[0] and base_sum[0] ──
    acc_base = tl.zeros([R, C], dtype=tl.float32)
    acc_sum = tl.zeros([R, C], dtype=tl.float32)
    for b_start in range(0, B, BLOCK_B):
        b_offs = b_start + tl.arange(0, BLOCK_B)
        b_mask = b_offs < B
        s = tl.load(source_ptr + b_offs[:, None] * stride_src_b + 0 * stride_src_l
                     + tl.arange(0, R)[None, :],
                     mask=b_mask[:, None], other=0.0).to(tl.float32)
        d = tl.load(dest_ptr + b_offs[:, None] * stride_dst_b + 0 * stride_dst_l
                     + tl.arange(0, C)[None, :],
                     mask=b_mask[:, None], other=0.0).to(tl.float32)
        y_val = tl.load(y_ptr + b_offs, mask=b_mask, other=0.0).to(tl.float32)
        acc_base += tl.dot(tl.trans(s), d)
        acc_sum += tl.dot(tl.trans(s * y_val[:, None]), d)

    # memory[0] += base_sum[0]
    m0 = tl.load(memory_ptr + rc_offs)
    tl.store(memory_ptr + rc_offs, m0 + acc_sum)

    # ── Carry: layers 0 → 1 → ... → L-1 ──
    for j in tl.static_range(0, L - 1):
        # Compute base[j+1] (needed for carry addition)
        acc_base_next = tl.zeros([R, C], dtype=tl.float32)
        for b_start in range(0, B, BLOCK_B):
            b_offs = b_start + tl.arange(0, BLOCK_B)
            b_mask = b_offs < B
            s = tl.load(source_ptr + b_offs[:, None] * stride_src_b
                         + (j + 1) * stride_src_l + tl.arange(0, R)[None, :],
                         mask=b_mask[:, None], other=0.0).to(tl.float32)
            d = tl.load(dest_ptr + b_offs[:, None] * stride_dst_b
                         + (j + 1) * stride_dst_l + tl.arange(0, C)[None, :],
                         mask=b_mask[:, None], other=0.0).to(tl.float32)
            acc_base_next += tl.dot(tl.trans(s), d)

        # min_ratio = min(memory[j] / (threshold * base[j]))
        j_off = j * RC
        m_j = tl.load(memory_ptr + j_off + rc_offs)
        ratio = m_j / (carry_threshold * acc_base)
        min_ratio = tl.min(ratio)

        if min_ratio >= 1.0:
            tl.store(memory_ptr + j_off + rc_offs,
                     m_j - carry_threshold * min_ratio * acc_base)
            j1_off = (j + 1) * RC
            m_j1 = tl.load(memory_ptr + j1_off + rc_offs)
            tl.store(memory_ptr + j1_off + rc_offs,
                     m_j1 + min_ratio * acc_base_next)

        acc_base = acc_base_next


def fused_write_carry(memory_matrix, source_emb, dest_emb, input_y, carry_threshold):
    """Fully fused write_sum + carry. Single kernel, no intermediate buffers."""
    B = source_emb.shape[0]
    L, R, C = memory_matrix.shape
    BLOCK_B = max(min(_next_power_of_2(B), 64), 16)

    _fused_write_carry_kernel[(1,)](
        memory_matrix, source_emb, dest_emb, input_y,
        carry_threshold=carry_threshold,
        B=B,
        stride_src_b=source_emb.stride(0),
        stride_src_l=source_emb.stride(1),
        stride_dst_b=dest_emb.stride(0),
        stride_dst_l=dest_emb.stride(1),
        L=L, R=R, C=C,
        BLOCK_B=BLOCK_B,
        num_stages=1,
    )
