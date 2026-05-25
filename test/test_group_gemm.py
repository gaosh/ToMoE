import time
import torch
import grouped_gemm


@torch.no_grad()
def benchmark_forward(fn, warmup=20, iters=100):
    for _ in range(warmup):
        y = fn()
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        y = fn()
    torch.cuda.synchronize()

    return (time.time() - start) / iters


def benchmark_fwd_bwd(fn, params, warmup=10, iters=50):
    for _ in range(warmup):
        y = fn()
        loss = y.float().pow(2).mean()
        loss.backward()
        for p in params:
            p.grad = None
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        y = fn()
        loss = y.float().pow(2).mean()
        loss.backward()
        for p in params:
            p.grad = None
    torch.cuda.synchronize()

    return (time.time() - start) / iters


def main():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    num_experts = 8
    hidden = 4096
    intermediate = 14336

    # 模拟每张 GPU 一个 micro-step 处理 8192 tokens
    tokens_per_expert = 1024
    total_tokens = num_experts * tokens_per_expert

    splits = torch.full(
        (num_experts,),
        tokens_per_expert,
        device=device,
        dtype=torch.int32,
    )

    print("grouped_gemm:", grouped_gemm.__file__)
    print(f"experts={num_experts}, total_tokens={total_tokens}")
    print(f"hidden={hidden}, intermediate={intermediate}, dtype={dtype}")

    # ------------------------------------------------------------
    # Forward only
    # ------------------------------------------------------------
    x = torch.randn(total_tokens, hidden, device=device, dtype=dtype)
    w = torch.randn(num_experts, hidden, intermediate, device=device, dtype=dtype)

    def grouped_forward():
        return grouped_gemm.ops.gmm(x, w, splits)

    def loop_forward():
        outs = []
        start = 0
        for e in range(num_experts):
            n = int(splits[e].item())
            xe = x[start:start + n]
            we = w[e]
            outs.append(xe @ we)
            start += n
        return torch.cat(outs, dim=0)

    # correctness
    y_grouped = grouped_forward()
    y_loop = loop_forward()
    max_diff = (y_grouped - y_loop).float().abs().max().item()
    mean_diff = (y_grouped - y_loop).float().abs().mean().item()

    print("\nCorrectness:")
    print("y_grouped:", y_grouped.shape)
    print("y_loop:   ", y_loop.shape)
    print("max diff: ", max_diff)
    print("mean diff:", mean_diff)

    t_grouped = benchmark_forward(grouped_forward)
    t_loop = benchmark_forward(loop_forward)

    flops = 2 * total_tokens * hidden * intermediate

    print("\nForward only:")
    print(f"grouped_gemm: {t_grouped * 1000:.3f} ms, {flops / t_grouped / 1e12:.2f} TFLOPs")
    print(f"for loop:     {t_loop * 1000:.3f} ms, {flops / t_loop / 1e12:.2f} TFLOPs")
    print(f"speedup:      {t_loop / t_grouped:.2f}x")

    # ------------------------------------------------------------
    # Forward + backward
    # ------------------------------------------------------------
    x_g = torch.randn(total_tokens, hidden, device=device, dtype=dtype, requires_grad=True)
    w_g = torch.randn(num_experts, hidden, intermediate, device=device, dtype=dtype, requires_grad=True)

    x_l = x_g.detach().clone().requires_grad_(True)
    w_l = w_g.detach().clone().requires_grad_(True)

    def grouped_fwd_bwd():
        return grouped_gemm.ops.gmm(x_g, w_g, splits)

    def loop_fwd_bwd():
        outs = []
        start = 0
        for e in range(num_experts):
            n = int(splits[e].item())
            xe = x_l[start:start + n]
            we = w_l[e]
            outs.append(xe @ we)
            start += n
        return torch.cat(outs, dim=0)

    t_grouped_bwd = benchmark_fwd_bwd(grouped_fwd_bwd, [x_g, w_g])
    t_loop_bwd = benchmark_fwd_bwd(loop_fwd_bwd, [x_l, w_l])

    # rough: forward + dX + dW = 3 GEMMs
    train_flops = 3 * flops

    print("\nForward + backward:")
    print(f"grouped_gemm: {t_grouped_bwd * 1000:.3f} ms, {train_flops / t_grouped_bwd / 1e12:.2f} TFLOPs")
    print(f"for loop:     {t_loop_bwd * 1000:.3f} ms, {train_flops / t_loop_bwd / 1e12:.2f} TFLOPs")
    print(f"speedup:      {t_loop_bwd / t_grouped_bwd:.2f}x")

    # ------------------------------------------------------------
    # Uneven routing test
    # ------------------------------------------------------------
    uneven_splits = torch.tensor(
        [1600, 1200, 900, 800, 1000, 700, 1100, 892],
        device=device,
        dtype=torch.int32,
    )
    uneven_total = int(uneven_splits.sum().item())

    x = torch.randn(uneven_total, hidden, device=device, dtype=dtype)
    w = torch.randn(num_experts, hidden, intermediate, device=device, dtype=dtype)

    def grouped_forward_uneven():
        return grouped_gemm.ops.gmm(x, w, uneven_splits)

    def loop_forward_uneven():
        outs = []
        start = 0
        for e in range(num_experts):
            n = int(uneven_splits[e].item())
            xe = x[start:start + n]
            we = w[e]
            outs.append(xe @ we)
            start += n
        return torch.cat(outs, dim=0)

    y_grouped = grouped_forward_uneven()
    y_loop = loop_forward_uneven()

    max_diff = (y_grouped - y_loop).float().abs().max().item()
    mean_diff = (y_grouped - y_loop).float().abs().mean().item()

    t_grouped_uneven = benchmark_forward(grouped_forward_uneven)
    t_loop_uneven = benchmark_forward(loop_forward_uneven)

    flops_uneven = 2 * uneven_total * hidden * intermediate

    print("\nUneven forward only:")
    print("splits:", uneven_splits.tolist())
    print("max diff: ", max_diff)
    print("mean diff:", mean_diff)
    print(f"grouped_gemm: {t_grouped_uneven * 1000:.3f} ms, {flops_uneven / t_grouped_uneven / 1e12:.2f} TFLOPs")
    print(f"for loop:     {t_loop_uneven * 1000:.3f} ms, {flops_uneven / t_loop_uneven / 1e12:.2f} TFLOPs")
    print(f"speedup:      {t_loop_uneven / t_grouped_uneven:.2f}x")


if __name__ == "__main__":
    main()