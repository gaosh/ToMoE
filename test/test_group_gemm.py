import time
import torch
import grouped_gemm


def sync():
    torch.cuda.synchronize()


@torch.no_grad()
def benchmark_forward(fn, warmup=20, iters=100):
    for _ in range(warmup):
        _ = fn()

    sync()
    start = time.time()

    for _ in range(iters):
        _ = fn()

    sync()
    elapsed = time.time() - start
    return elapsed / iters


def benchmark_fwd_bwd(fn, params, warmup=10, iters=50):
    for _ in range(warmup):
        y = fn()
        loss = y.float().pow(2).mean()
        loss.backward()

        for p in params:
            p.grad = None

    sync()
    start = time.time()

    for _ in range(iters):
        y = fn()
        loss = y.float().pow(2).mean()
        loss.backward()

        for p in params:
            p.grad = None

    sync()
    elapsed = time.time() - start
    return elapsed / iters


def loop_gemm(x, w, splits_cpu):
    outs = []
    start = 0

    for e in range(w.shape[0]):
        n = int(splits_cpu[e].item())
        if n == 0:
            continue

        xe = x[start:start + n]
        we = w[e]
        outs.append(xe @ we)
        start += n

    return torch.cat(outs, dim=0)


def print_env():
    print("=" * 80, flush=True)
    print("Environment", flush=True)
    print("=" * 80, flush=True)
    print("torch:", torch.__version__, flush=True)
    print("torch cuda:", torch.version.cuda, flush=True)
    print("device:", torch.cuda.get_device_name(), flush=True)
    print("grouped_gemm:", grouped_gemm.__file__, flush=True)
    print("=" * 80, flush=True)


def run_case(
    name,
    num_experts,
    hidden,
    intermediate,
    splits_list,
    dtype=torch.bfloat16,
    device="cuda",
):
    print("\n" + "=" * 80, flush=True)
    print(name, flush=True)
    print("=" * 80, flush=True)

    splits_cpu = torch.tensor(
        splits_list,
        device="cpu",
        dtype=torch.int64,
    )

    total_tokens = int(splits_cpu.sum().item())

    print(f"num_experts: {num_experts}", flush=True)
    print(f"total_tokens: {total_tokens}", flush=True)
    print(f"hidden: {hidden}", flush=True)
    print(f"intermediate: {intermediate}", flush=True)
    print(f"dtype: {dtype}", flush=True)
    print(f"splits_cpu: {splits_cpu.tolist()}", flush=True)

    x = torch.randn(
        total_tokens,
        hidden,
        device=device,
        dtype=dtype,
    )

    w = torch.randn(
        num_experts,
        hidden,
        intermediate,
        device=device,
        dtype=dtype,
    )

    def grouped_forward():
        return grouped_gemm.ops.gmm(x, w, splits_cpu)

    def loop_forward():
        return loop_gemm(x, w, splits_cpu)

    print("\nCorrectness check", flush=True)

    y_grouped = grouped_forward()
    y_loop = loop_forward()
    sync()

    print("y_grouped:", tuple(y_grouped.shape), y_grouped.dtype, flush=True)
    print("y_loop:   ", tuple(y_loop.shape), y_loop.dtype, flush=True)

    max_diff = (y_grouped - y_loop).float().abs().max().item()
    mean_diff = (y_grouped - y_loop).float().abs().mean().item()

    print(f"max diff:  {max_diff:.6f}", flush=True)
    print(f"mean diff: {mean_diff:.6f}", flush=True)

    # Forward benchmark
    print("\nForward only benchmark", flush=True)

    t_grouped = benchmark_forward(grouped_forward)
    t_loop = benchmark_forward(loop_forward)

    flops = 2 * total_tokens * hidden * intermediate

    print(
        f"grouped_gemm: {t_grouped * 1000:.3f} ms, "
        f"{flops / t_grouped / 1e12:.2f} TFLOPs",
        flush=True,
    )
    print(
        f"for loop:     {t_loop * 1000:.3f} ms, "
        f"{flops / t_loop / 1e12:.2f} TFLOPs",
        flush=True,
    )
    print(f"speedup:      {t_loop / t_grouped:.2f}x", flush=True)

    # Forward + backward benchmark
    print("\nForward + backward benchmark", flush=True)

    x_g = torch.randn(
        total_tokens,
        hidden,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    w_g = torch.randn(
        num_experts,
        hidden,
        intermediate,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    x_l = x_g.detach().clone().requires_grad_(True)
    w_l = w_g.detach().clone().requires_grad_(True)

    def grouped_fwd_bwd():
        return grouped_gemm.ops.gmm(x_g, w_g, splits_cpu)

    def loop_fwd_bwd():
        return loop_gemm(x_l, w_l, splits_cpu)

    t_grouped_bwd = benchmark_fwd_bwd(
        grouped_fwd_bwd,
        params=[x_g, w_g],
    )

    t_loop_bwd = benchmark_fwd_bwd(
        loop_fwd_bwd,
        params=[x_l, w_l],
    )

    # approx: forward + dX + dW = 3 GEMMs
    train_flops = 3 * flops

    print(
        f"grouped_gemm: {t_grouped_bwd * 1000:.3f} ms, "
        f"{train_flops / t_grouped_bwd / 1e12:.2f} TFLOPs",
        flush=True,
    )
    print(
        f"for loop:     {t_loop_bwd * 1000:.3f} ms, "
        f"{train_flops / t_loop_bwd / 1e12:.2f} TFLOPs",
        flush=True,
    )
    print(f"speedup:      {t_loop_bwd / t_grouped_bwd:.2f}x", flush=True)


def main():
    print("script started", flush=True)

    assert torch.cuda.is_available(), "CUDA is not available"

    print_env()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    num_experts = 8
    hidden = 4096
    intermediate = 14336

    # Case 1: balanced routing
    # Simulates per-GPU micro-step = 8192 tokens, evenly split over 8 experts.
    run_case(
        name="Balanced routing: 8 experts × 1024 tokens = 8192 tokens",
        num_experts=num_experts,
        hidden=hidden,
        intermediate=intermediate,
        splits_list=[1024] * 8,
    )

    # Case 2: uneven routing
    # Same total tokens = 8192, but uneven expert assignment.
    run_case(
        name="Uneven routing: total 8192 tokens",
        num_experts=num_experts,
        hidden=hidden,
        intermediate=intermediate,
        splits_list=[1600, 1200, 900, 800, 1000, 700, 1100, 892],
    )


if __name__ == "__main__":
    main()