import torch
import time
from crane.crane_kernels.query_kernel import fused_query_min, fused_query_min_inference
from crane.crane_kernels.write_kernel import fused_write_sum, fused_write_sum_triton, fused_carry, fused_write_carry


def _reference_query_min(source_emb, dest_emb, memory_matrix, activated_memory_dim):
    B, L, R = source_emb.shape
    C = dest_emb.shape[2]
    base_per_layer = torch.einsum('blr,blc->blrc', source_emb, dest_emb)
    memory_now = memory_matrix.unsqueeze(0).expand(B, -1, -1, -1)
    nz = base_per_layer != 0
    ratio_map = torch.zeros_like(base_per_layer)
    ratio_map[nz] = memory_now[nz] / base_per_layer[nz]
    inf = torch.full_like(base_per_layer, float('inf'))
    masked = torch.where(nz, ratio_map, inf)
    layer_min, _ = masked.view(B, L, -1).min(dim=-1)
    has_valid = nz.view(B, L, -1).any(dim=-1)
    ratio_min = torch.zeros(B, L, device=base_per_layer.device, dtype=base_per_layer.dtype)
    ratio_min[has_valid] = layer_min[has_valid]
    if activated_memory_dim < (L - 1):
        mask = torch.zeros(L, dtype=torch.bool, device=ratio_min.device)
        mask[:activated_memory_dim + 1] = True
        ratio_min = ratio_min * mask.unsqueeze(0)
    return ratio_min


def _reference_write_sum(source_emb, dest_emb, input_y, L, R, C):
    B = source_emb.shape[0]
    input_y_expanded = input_y.view(B, 1, 1, 1).expand(B, L, 1, 1)
    base_all = torch.einsum('blr,blc->blrc', source_emb, dest_emb)
    base_per_layer = base_all.sum(dim=0)
    base_sum_per_layer = (base_all * input_y_expanded).sum(dim=0)
    return base_per_layer, base_sum_per_layer


def test_query_correctness():
    torch.manual_seed(42)
    B, L, R, C = 32, 4, 64, 64
    source_emb = torch.rand(B, L, R, device='cuda') + 0.1
    dest_emb = torch.rand(B, L, C, device='cuda') + 0.1
    memory_matrix = torch.rand(L, R, C, device='cuda') * 10

    for activated_dim in [0, 1, 2, 3]:
        ref = _reference_query_min(source_emb, dest_emb, memory_matrix, activated_dim)
        fused = fused_query_min(source_emb, dest_emb, memory_matrix, activated_dim)
        assert torch.allclose(ref, fused, atol=1e-5, rtol=1e-5), \
            f"Query mismatch at activated_dim={activated_dim}: max diff={torch.max(torch.abs(ref - fused))}"
    print("PASS: test_query_correctness")


def test_query_inference_correctness():
    torch.manual_seed(42)
    B, L, R, C = 32, 4, 64, 64
    source_emb = torch.rand(B, L, R, device='cuda') + 0.1
    dest_emb = torch.rand(B, L, C, device='cuda') + 0.1
    memory_matrix = torch.rand(L, R, C, device='cuda') * 10

    for activated_dim in [0, 1, 2, 3]:
        ref = _reference_query_min(source_emb, dest_emb, memory_matrix, activated_dim)
        with torch.no_grad():
            infer = fused_query_min_inference(source_emb, dest_emb, memory_matrix, activated_dim)
        assert torch.allclose(ref, infer, atol=1e-5, rtol=1e-5), \
            f"Inference query mismatch at activated_dim={activated_dim}: max diff={torch.max(torch.abs(ref - infer))}"
    print("PASS: test_query_inference_correctness")


def test_query_edge_cases():
    torch.manual_seed(123)

    # B=1
    source_emb = torch.rand(1, 4, 64, device='cuda') + 0.1
    dest_emb = torch.rand(1, 4, 64, device='cuda') + 0.1
    memory_matrix = torch.rand(4, 64, 64, device='cuda') * 5
    ref = _reference_query_min(source_emb, dest_emb, memory_matrix, 3)
    fused = fused_query_min(source_emb, dest_emb, memory_matrix, 3)
    assert torch.allclose(ref, fused, atol=1e-5, rtol=1e-5), "B=1 mismatch"

    # All zeros in memory
    memory_matrix = torch.zeros(4, 64, 64, device='cuda')
    ref = _reference_query_min(source_emb, dest_emb, memory_matrix, 3)
    fused = fused_query_min(source_emb, dest_emb, memory_matrix, 3)
    assert torch.allclose(ref, fused, atol=1e-5, rtol=1e-5), "Zero memory mismatch"

    # activated_memory_dim=0
    memory_matrix = torch.rand(4, 64, 64, device='cuda')
    ref = _reference_query_min(source_emb, dest_emb, memory_matrix, 0)
    fused = fused_query_min(source_emb, dest_emb, memory_matrix, 0)
    assert torch.allclose(ref, fused, atol=1e-5, rtol=1e-5), "activated_dim=0 mismatch"

    print("PASS: test_query_edge_cases")


def test_query_gradient():
    torch.manual_seed(42)
    B, L, R, C = 4, 2, 8, 8

    src_data = torch.rand(B, L, R, device='cuda') + 0.5
    dst_data = torch.rand(B, L, C, device='cuda') + 0.5
    memory_matrix = torch.rand(L, R, C, device='cuda') * 5

    source_emb_f = src_data.clone().requires_grad_(True)
    dest_emb_f = dst_data.clone().requires_grad_(True)
    fused_out = fused_query_min(source_emb_f, dest_emb_f, memory_matrix, L - 1)
    loss_fused = fused_out.sum()
    loss_fused.backward()
    grad_src_fused = source_emb_f.grad.clone()
    grad_dst_fused = dest_emb_f.grad.clone()

    source_emb_r = src_data.clone().requires_grad_(True)
    dest_emb_r = dst_data.clone().requires_grad_(True)
    ref_out = _reference_query_min(source_emb_r, dest_emb_r, memory_matrix, L - 1)
    loss_ref = ref_out.sum()
    loss_ref.backward()
    grad_src_ref = source_emb_r.grad.clone()
    grad_dst_ref = dest_emb_r.grad.clone()

    assert torch.allclose(grad_src_fused, grad_src_ref, atol=1e-4, rtol=1e-4), \
        f"Source grad mismatch: max diff={torch.max(torch.abs(grad_src_fused - grad_src_ref))}"
    assert torch.allclose(grad_dst_fused, grad_dst_ref, atol=1e-4, rtol=1e-4), \
        f"Dest grad mismatch: max diff={torch.max(torch.abs(grad_dst_fused - grad_dst_ref))}"
    print("PASS: test_query_gradient")


def test_write_correctness():
    torch.manual_seed(42)
    B, L, R, C = 32, 4, 64, 64
    source_emb = torch.rand(B, L, R, device='cuda') + 0.1
    dest_emb = torch.rand(B, L, C, device='cuda') + 0.1
    input_y = torch.rand(B, device='cuda') * 10

    ref_base, ref_sum = _reference_write_sum(source_emb, dest_emb, input_y, L, R, C)
    fused_base, fused_sum = fused_write_sum(source_emb, dest_emb, input_y, L, R, C)

    assert torch.allclose(ref_base, fused_base, atol=1e-4, rtol=1e-4), \
        f"Write base mismatch: max diff={torch.max(torch.abs(ref_base - fused_base))}"
    assert torch.allclose(ref_sum, fused_sum, atol=1e-4, rtol=1e-4), \
        f"Write sum mismatch: max diff={torch.max(torch.abs(ref_sum - fused_sum))}"
    print("PASS: test_write_correctness")


def test_write_triton_correctness():
    torch.manual_seed(42)
    B, L, R, C = 32, 4, 64, 64
    source_emb = torch.rand(B, L, R, device='cuda') + 0.1
    dest_emb = torch.rand(B, L, C, device='cuda') + 0.1
    input_y = torch.rand(B, device='cuda') * 10

    ref_base, ref_sum = _reference_write_sum(source_emb, dest_emb, input_y, L, R, C)
    base_out = torch.empty(L, R, C, device='cuda')
    sum_out = torch.empty(L, R, C, device='cuda')
    triton_base, triton_sum = fused_write_sum_triton(
        source_emb, dest_emb, input_y, L, R, C, base_out, sum_out)

    # Triton tl.dot accumulation order differs from einsum — relax tolerance
    assert torch.allclose(ref_base, triton_base, atol=0.02, rtol=1e-3), \
        f"Triton write base mismatch: max diff={torch.max(torch.abs(ref_base - triton_base))}"
    assert torch.allclose(ref_sum, triton_sum, atol=0.02, rtol=1e-3), \
        f"Triton write sum mismatch: max diff={torch.max(torch.abs(ref_sum - triton_sum))}"
    print("PASS: test_write_triton_correctness")


def test_write_edge_cases():
    torch.manual_seed(123)

    # B=1
    source_emb = torch.rand(1, 4, 64, device='cuda') + 0.1
    dest_emb = torch.rand(1, 4, 64, device='cuda') + 0.1
    input_y = torch.tensor([5.0], device='cuda')
    ref_base, ref_sum = _reference_write_sum(source_emb, dest_emb, input_y, 4, 64, 64)
    fused_base, fused_sum = fused_write_sum(source_emb, dest_emb, input_y, 4, 64, 64)
    assert torch.allclose(ref_base, fused_base, atol=1e-4, rtol=1e-4), "B=1 write base mismatch"
    assert torch.allclose(ref_sum, fused_sum, atol=1e-4, rtol=1e-4), "B=1 write sum mismatch"

    print("PASS: test_write_edge_cases")


def test_write_carry_correctness():
    """fused_write_carry must produce identical memory state as write_sum + carry."""
    torch.manual_seed(42)
    B, L, R, C = 32, 4, 64, 64
    carry_threshold = 4
    source_emb = torch.rand(B, L, R, device='cuda') + 0.1
    dest_emb = torch.rand(B, L, C, device='cuda') + 0.1
    input_y = torch.rand(B, device='cuda') * 10

    # Reference: write_sum_triton + carry (two separate kernels)
    mem_ref = torch.rand(L, R, C, device='cuda') * 5
    base_buf = torch.empty(L, R, C, device='cuda')
    sum_buf = torch.empty(L, R, C, device='cuda')
    fused_write_sum_triton(source_emb, dest_emb, input_y, L, R, C, base_buf, sum_buf)
    fused_carry(mem_ref, base_buf, sum_buf, carry_threshold)

    # Fused: single kernel
    mem_fused = mem_ref.clone()
    mem_fused.copy_(torch.rand(L, R, C, device='cuda') * 5)
    # Reset to same initial state
    torch.manual_seed(42)
    _ = torch.rand(B, L, R, device='cuda')
    _ = torch.rand(B, L, C, device='cuda')
    _ = torch.rand(B, device='cuda')
    mem_init = torch.rand(L, R, C, device='cuda') * 5
    mem_ref.copy_(mem_init)
    mem_fused.copy_(mem_init)

    fused_write_sum_triton(source_emb, dest_emb, input_y, L, R, C, base_buf, sum_buf)
    fused_carry(mem_ref, base_buf, sum_buf, carry_threshold)
    fused_write_carry(mem_fused, source_emb, dest_emb, input_y, carry_threshold)

    assert torch.allclose(mem_ref, mem_fused, atol=0.02, rtol=1e-3), \
        f"Write+carry mismatch: max diff={torch.max(torch.abs(mem_ref - mem_fused))}"
    print("PASS: test_write_carry_correctness")


def test_performance():
    torch.manual_seed(42)
    B, L, R, C = 2048, 4, 64, 64
    source_emb = torch.rand(B, L, R, device='cuda') + 0.1
    dest_emb = torch.rand(B, L, C, device='cuda') + 0.1
    memory_matrix = torch.rand(L, R, C, device='cuda') * 10
    input_y = torch.rand(B, device='cuda') * 10

    # Warmup
    for _ in range(3):
        fused_query_min(source_emb, dest_emb, memory_matrix, L - 1)
        fused_query_min_inference(source_emb, dest_emb, memory_matrix, L - 1)
        fused_write_sum(source_emb, dest_emb, input_y, L, R, C)
    torch.cuda.synchronize()

    # Benchmark fused query (training, with backward support)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(100):
        fused_query_min(source_emb, dest_emb, memory_matrix, L - 1)
    torch.cuda.synchronize()
    fused_query_time = (time.perf_counter() - start) / 100
    fused_query_mem = torch.cuda.max_memory_allocated() / 1e6

    # Benchmark inference query (no backward)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(100):
        fused_query_min_inference(source_emb, dest_emb, memory_matrix, L - 1)
    torch.cuda.synchronize()
    infer_query_time = (time.perf_counter() - start) / 100
    infer_query_mem = torch.cuda.max_memory_allocated() / 1e6

    # Benchmark reference query
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(100):
        _reference_query_min(source_emb, dest_emb, memory_matrix, L - 1)
    torch.cuda.synchronize()
    ref_query_time = (time.perf_counter() - start) / 100
    ref_query_mem = torch.cuda.max_memory_allocated() / 1e6

    print(f"Query  - Train:  {fused_query_time*1000:.3f}ms, {fused_query_mem:.1f}MB")
    print(f"Query  - Infer:  {infer_query_time*1000:.3f}ms, {infer_query_mem:.1f}MB")
    print(f"Query  - Ref:    {ref_query_time*1000:.3f}ms, {ref_query_mem:.1f}MB")

    # Benchmark fused write (bmm)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(100):
        fused_write_sum(source_emb, dest_emb, input_y, L, R, C)
    torch.cuda.synchronize()
    fused_write_time = (time.perf_counter() - start) / 100
    fused_write_mem = torch.cuda.max_memory_allocated() / 1e6

    # Benchmark Triton write
    base_out = torch.empty(L, R, C, device='cuda')
    sum_out = torch.empty(L, R, C, device='cuda')
    for _ in range(3):
        fused_write_sum_triton(source_emb, dest_emb, input_y, L, R, C, base_out, sum_out)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(100):
        fused_write_sum_triton(source_emb, dest_emb, input_y, L, R, C, base_out, sum_out)
    torch.cuda.synchronize()
    triton_write_time = (time.perf_counter() - start) / 100
    triton_write_mem = torch.cuda.max_memory_allocated() / 1e6

    # Benchmark reference write
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(100):
        _reference_write_sum(source_emb, dest_emb, input_y, L, R, C)
    torch.cuda.synchronize()
    ref_write_time = (time.perf_counter() - start) / 100
    ref_write_mem = torch.cuda.max_memory_allocated() / 1e6

    # Benchmark fused write+carry (single kernel)
    mem_bench = torch.rand(L, R, C, device='cuda') * 10
    for _ in range(3):
        fused_write_carry(mem_bench, source_emb, dest_emb, input_y, 4)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(100):
        fused_write_carry(mem_bench, source_emb, dest_emb, input_y, 4)
    torch.cuda.synchronize()
    fused_wc_time = (time.perf_counter() - start) / 100
    fused_wc_mem = torch.cuda.max_memory_allocated() / 1e6

    print(f"Write  - cuBLAS: {fused_write_time*1000:.3f}ms, {fused_write_mem:.1f}MB")
    print(f"Write  - Triton: {triton_write_time*1000:.3f}ms, {triton_write_mem:.1f}MB")
    print(f"Write  - Fused:  {fused_wc_time*1000:.3f}ms, {fused_wc_mem:.1f}MB")
    print(f"Write  - Ref:    {ref_write_time*1000:.3f}ms, {ref_write_mem:.1f}MB")
    print("PASS: test_performance")


def test_all():
    test_query_correctness()
    test_query_inference_correctness()
    test_query_edge_cases()
    test_query_gradient()
    test_write_correctness()
    test_write_triton_correctness()
    test_write_carry_correctness()
    test_write_edge_cases()
    test_performance()
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_all()
