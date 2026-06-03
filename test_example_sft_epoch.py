import argparse
import math
import os

import torch
import torch.distributed as dist
from datasets import load_dataset, load_dataset_builder
try:
    from datasets.distributed import split_dataset_by_node
except Exception:
    split_dataset_by_node = None
from torch.utils.data import DataLoader, IterableDataset


class RawStreamingDataset(IterableDataset):
    def __init__(self, dataset, max_train_samples=None):
        self.dataset = dataset
        self.max_train_samples = max_train_samples

    def __iter__(self):
        yielded = 0
        for example in self.dataset:
            yield example
            yielded += 1
            if self.max_train_samples is not None and yielded >= self.max_train_samples:
                return


def collate_examples(features):
    return features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count one SFT streaming epoch after rank sharding and DataLoader workers."
    )
    parser.add_argument("--dataset_name", type=str, default="allenai/tulu-3-sft-mixture")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument(
        "--limit_batches",
        type=int,
        default=None,
        help="Optional early stop for smoke tests; omit to count the full epoch.",
    )
    return parser.parse_args()


def maybe_init_distributed(args):
    env_rank = os.environ.get("RANK")
    env_world_size = os.environ.get("WORLD_SIZE")
    rank = int(env_rank) if env_rank is not None else (args.rank or 0)
    world_size = int(env_world_size) if env_world_size is not None else (args.world_size or 1)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("gloo")
    return rank, world_size


def get_split_num_examples(args):
    try:
        builder = load_dataset_builder(
            args.dataset_name,
            revision=args.dataset_revision,
            cache_dir=args.dataset_cache_dir,
        )
        split_info = builder.info.splits.get(args.dataset_split)
        if split_info is None:
            return None
        num_examples = int(split_info.num_examples)
        if args.max_train_samples is not None:
            num_examples = min(num_examples, int(args.max_train_samples))
        return num_examples
    except Exception:
        return None


def build_loader(args, rank, world_size):
    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        revision=args.dataset_revision,
        cache_dir=args.dataset_cache_dir,
        streaming=True,
    )
    if args.shuffle_buffer_size and args.shuffle_buffer_size > 0:
        dataset = dataset.shuffle(buffer_size=args.shuffle_buffer_size, seed=args.seed)
    if split_dataset_by_node is not None:
        dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    else:
        dataset = dataset.shard(num_shards=world_size, index=rank)

    return DataLoader(
        RawStreamingDataset(dataset, max_train_samples=args.max_train_samples),
        batch_size=args.per_device_train_batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_examples,
    )


def reduce_counts(rows, batches):
    if not dist.is_available() or not dist.is_initialized():
        return {
            "min_rows": rows,
            "max_rows": rows,
            "sum_rows": rows,
            "min_batches": batches,
            "max_batches": batches,
            "sum_batches": batches,
        }

    values = torch.tensor([rows, batches], dtype=torch.long)
    min_values = values.clone()
    max_values = values.clone()
    sum_values = values.clone()
    dist.all_reduce(min_values, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_values, op=dist.ReduceOp.MAX)
    dist.all_reduce(sum_values, op=dist.ReduceOp.SUM)
    return {
        "min_rows": int(min_values[0].item()),
        "max_rows": int(max_values[0].item()),
        "sum_rows": int(sum_values[0].item()),
        "min_batches": int(min_values[1].item()),
        "max_batches": int(max_values[1].item()),
        "sum_batches": int(sum_values[1].item()),
    }


def main():
    args = parse_args()
    rank, world_size = maybe_init_distributed(args)
    loader = build_loader(args, rank, world_size)

    rows = 0
    batches = 0
    for batch in loader:
        rows += len(batch)
        batches += 1
        if args.limit_batches is not None and batches >= args.limit_batches:
            break

    counts = reduce_counts(rows, batches)
    local_optimizer_steps = batches // args.gradient_accumulation_steps
    synced_optimizer_steps = counts["min_batches"] // args.gradient_accumulation_steps

    print(
        f"[rank {rank}] rows={rows} batches={batches} "
        f"local_optimizer_steps={local_optimizer_steps}",
        flush=True,
    )
    if rank == 0:
        expected_rows = get_split_num_examples(args)
        if expected_rows is not None:
            expected_examples_per_step = (
                world_size
                * args.per_device_train_batch_size
                * args.gradient_accumulation_steps
            )
            expected_steps = math.ceil(expected_rows / expected_examples_per_step)
            print(
                f"[summary] builder_rows={expected_rows} "
                f"expected_optimizer_steps_per_epoch={expected_steps} "
                f"examples_per_optimizer_step={expected_examples_per_step}",
                flush=True,
            )
        print(
            f"[summary] world_size={world_size} min_rows={counts['min_rows']} "
            f"max_rows={counts['max_rows']} sum_rows={counts['sum_rows']} "
            f"min_batches={counts['min_batches']} max_batches={counts['max_batches']} "
            f"sum_batches={counts['sum_batches']} "
            f"synced_optimizer_steps_per_epoch={synced_optimizer_steps}",
            flush=True,
        )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
