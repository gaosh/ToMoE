import argparse
import copy
import time
from types import SimpleNamespace

import torch
from torch import nn

from models.modeling_llama_tomoe_gated_actual_moe import (
    GroupedSwiGLUExperts,
    LlamaMLP,
    LlamaMlpExpert,
    SingleMlpRouter,
    _HAS_GROUPED_GEMM,
)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_mlp(num_experts, hidden_size, intermediate_size, dtype, device):
    config = SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        pretraining_tp=1,
        tomoe_moe_cfgs=[intermediate_size, num_experts],
        tomoe_moe_impl="naive",
    )
    mlp = LlamaMLP(config)
    mlp.gate_proj = None
    mlp.up_proj = None
    mlp.down_proj = None
    mlp.router = SingleMlpRouter(hidden_size, experts=num_experts)
    mlp.experts = nn.ModuleList(
        [LlamaMlpExpert(hidden_size, intermediate_size, config.hidden_act) for _ in range(num_experts)]
    )
    mlp.num_experts = num_experts
    mlp.actual_moe = True
    return mlp.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def correctness():
    if not torch.cuda.is_available() or not _HAS_GROUPED_GEMM:
        print("SKIP: grouped_gemm correctness requires CUDA and grouped_gemm.")
        return

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    naive = make_mlp(
        num_experts=4,
        hidden_size=64,
        intermediate_size=128,
        dtype=dtype,
        device=device,
    )
    grouped = copy.deepcopy(naive)
    grouped.set_moe_impl("grouped_gemm")

    x = torch.randn(2, 32, 64, device=device, dtype=dtype)
    y_naive, router_naive = naive(x)
    y_grouped, router_grouped = grouped(x)
    sync()

    max_diff = (y_naive - y_grouped).float().abs().max().item()
    router_diff = (router_naive - router_grouped).float().abs().max().item()
    print(f"correctness max_diff={max_diff:.6f} router_diff={router_diff:.6f}")
    assert max_diff <= 2e-2
    assert router_diff == 0.0


def benchmark_forward(fn, warmup, iters):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        sync()
        start = time.time()
        for _ in range(iters):
            fn()
    sync()
    return (time.time() - start) / iters


def benchmark_fwd_bwd(fn, params, warmup, iters):
    for _ in range(warmup):
        loss = fn().float().pow(2).mean()
        loss.backward()
        for param in params:
            param.grad = None
    sync()
    start = time.time()
    for _ in range(iters):
        loss = fn().float().pow(2).mean()
        loss.backward()
        for param in params:
            param.grad = None
    sync()
    return (time.time() - start) / iters


def benchmark(args):
    if not torch.cuda.is_available() or not _HAS_GROUPED_GEMM:
        print("SKIP: benchmark requires CUDA and grouped_gemm.")
        return

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    splits_cpu = torch.full((args.num_experts,), args.total_tokens // args.num_experts, dtype=torch.int64)
    splits_cpu[-1] += args.total_tokens - int(splits_cpu.sum().item())
    sorted_tokens = torch.randn(args.total_tokens, args.hidden_size, device=device, dtype=dtype)

    naive = make_mlp(args.num_experts, args.hidden_size, args.intermediate_size, dtype, device)
    grouped = GroupedSwiGLUExperts.from_experts(naive.experts, "silu").to(device=device, dtype=dtype)

    def naive_fn():
        outs = []
        start = 0
        for expert_idx, expert in enumerate(naive.experts):
            num_tokens = int(splits_cpu[expert_idx].item())
            end = start + num_tokens
            if num_tokens > 0:
                outs.append(expert(sorted_tokens[start:end]))
            start = end
        return torch.cat(outs, dim=0)

    def grouped_fn():
        return grouped(sorted_tokens, splits_cpu)

    print("B200-like ToMoE SwiGLU expert benchmark")
    print(f"experts={args.num_experts} hidden={args.hidden_size} intermediate={args.intermediate_size}")
    print(f"tokens={args.total_tokens} dtype={dtype} splits={splits_cpu.tolist()}")

    t_naive = benchmark_forward(naive_fn, args.warmup, args.iters)
    t_grouped = benchmark_forward(grouped_fn, args.warmup, args.iters)
    print(f"forward naive={t_naive * 1000:.3f}ms grouped={t_grouped * 1000:.3f}ms speedup={t_naive / t_grouped:.2f}x")

    sorted_tokens_bwd = sorted_tokens.detach().clone().requires_grad_(True)

    def grouped_bwd_fn():
        return grouped(sorted_tokens_bwd, splits_cpu)

    grouped_params = [sorted_tokens_bwd] + list(grouped.parameters())
    t_grouped_bwd = benchmark_fwd_bwd(grouped_bwd_fn, grouped_params, args.warmup, args.iters)

    sorted_tokens_naive_bwd = sorted_tokens.detach().clone().requires_grad_(True)

    def naive_bwd_fn():
        outs = []
        start = 0
        for expert_idx, expert in enumerate(naive.experts):
            num_tokens = int(splits_cpu[expert_idx].item())
            end = start + num_tokens
            if num_tokens > 0:
                outs.append(expert(sorted_tokens_naive_bwd[start:end]))
            start = end
        return torch.cat(outs, dim=0)

    naive_params = [sorted_tokens_naive_bwd] + list(naive.parameters())
    t_naive_bwd = benchmark_fwd_bwd(naive_bwd_fn, naive_params, args.warmup, args.iters)
    print(
        f"fwd+bwd naive={t_naive_bwd * 1000:.3f}ms "
        f"grouped={t_grouped_bwd * 1000:.3f}ms speedup={t_naive_bwd / t_grouped_bwd:.2f}x"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--intermediate_size", type=int, default=14336)
    parser.add_argument("--total_tokens", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    correctness()
    if args.benchmark:
        benchmark(args)


if __name__ == "__main__":
    main()
