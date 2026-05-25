import argparse
import os
import sys
import time

print(f"[dataset-test] python entry pid={os.getpid()} argv={sys.argv}", flush=True)

np = None
torch = None
dist = None
ConcatDataset = None
DataLoader = None
DistributedSampler = None
PackedTokenDataset = None
DistributedEnv = None


def lazy_imports():
    global np, torch, dist, ConcatDataset, DataLoader, DistributedSampler, PackedTokenDataset, DistributedEnv
    tic = time.time()
    print("[dataset-test] importing numpy/torch/dataset modules", flush=True)
    import numpy as _np
    import torch as _torch
    import torch.distributed as _dist
    from torch.utils.data import ConcatDataset as _ConcatDataset
    from torch.utils.data import DataLoader as _DataLoader
    from torch.utils.data import DistributedSampler as _DistributedSampler

    from data.dataloader_packed import PackedTokenDataset as _PackedTokenDataset
    from utils import DistributedEnv as _DistributedEnv

    np = _np
    torch = _torch
    dist = _dist
    ConcatDataset = _ConcatDataset
    DataLoader = _DataLoader
    DistributedSampler = _DistributedSampler
    PackedTokenDataset = _PackedTokenDataset
    DistributedEnv = _DistributedEnv
    print(f"[dataset-test] imports done in {time.time() - tic:.2f}s", flush=True)


def dtype_from_name(name):
    return {
        "uint16": np.uint16,
        "uint32": np.uint32,
        "int32": np.int32,
        "int64": np.int64,
    }[name]


def build_dataset(data_dirs, seq_len, file_pattern, data_dtype):
    datasets = []
    for data_dir in data_dirs:
        tic = time.time()
        print(f"[dataset-build] loading {data_dir}", flush=True)
        dataset = PackedTokenDataset(
            data_dir=data_dir,
            seq_len=seq_len,
            file_pattern=file_pattern,
            dtype=dtype_from_name(data_dtype),
            shuffle_shards=False,
        )
        print(
            f"[dataset-build] loaded {data_dir}: len={len(dataset)} time={time.time() - tic:.2f}s",
            flush=True,
        )
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", nargs="+", required=True)
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--file_pattern", type=str, default="*.npy")
    parser.add_argument("--data_dtype", type=str, default="uint32", choices=["uint16", "uint32", "int32", "int64"])
    parser.add_argument("--num_batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("[dataset-test] parsed args", flush=True)
    lazy_imports()
    env = DistributedEnv()
    print(f"[rank:{env.global_rank}] before init_process_group", flush=True)
    dist.init_process_group("nccl", rank=env.global_rank, world_size=env.world_size)
    print(f"[rank:{env.global_rank}] after init_process_group", flush=True)
    torch.cuda.set_device(env.local_rank)
    torch.manual_seed(args.seed + env.global_rank)

    tic = time.time()
    dataset = build_dataset(args.data_dirs, args.seq_len, args.file_pattern, args.data_dtype)
    env.print(f"dataset built in {time.time() - tic:.2f}s")
    sampler = DistributedSampler(
        dataset,
        num_replicas=env.world_size,
        rank=env.global_rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    env.print_master("[dataset-test]")
    env.print_master(f"data_dirs: {args.data_dirs}")
    env.print_master(f"dataset_len: {len(dataset)}")
    env.print_master(f"world_size: {env.world_size}")
    env.print_master(f"per_rank_sampler_len: {len(sampler)}")
    env.print_master(f"batch_size_per_rank: {args.batch_size}")

    sampler.set_epoch(0)
    env.print("before dataloader iteration")
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.num_batches:
            break
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        first_tokens = input_ids[0, : min(8, input_ids.shape[1])].tolist()
        first_labels = labels[0, : min(8, labels.shape[1])].tolist()
        env.print(
            f"batch={batch_idx} input_shape={tuple(input_ids.shape)} "
            f"labels_shape={tuple(labels.shape)} first_tokens={first_tokens} "
            f"first_labels={first_labels}"
        )

    dist.barrier()
    env.print_master("[dataset-test] done")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
