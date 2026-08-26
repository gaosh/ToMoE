import argparse
import os
from datasets import load_dataset

DATASETS = {
    "fineweb_edu": {
        "name": "fineweb_edu",
        "repo": "HuggingFaceFW/fineweb-edu",
        "config": "sample-100BT",
        "split": "train",
    },
    "openwebmath": {
        "name": "openwebmath",
        "repo": "open-web-math/open-web-math",
        "config": None,
        "split": "train",
    },
}


def save_dataset_to_parquet(
    repo,
    config,
    split,
    output_dir,
    cache_dir,
    shard_size=100_000,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\nLoading {repo} config={config}")

    if config is None:
        ds = load_dataset(
            repo,
            split=split,
            cache_dir=cache_dir,
        )
    else:
        ds = load_dataset(
            repo,
            name=config,
            split=split,
            cache_dir=cache_dir,
        )

    print(ds)

    num_rows = len(ds)
    print(f"Total rows: {num_rows}")

    for start in range(0, num_rows, shard_size):
        end = min(start + shard_size, num_rows)
        shard_id = start // shard_size

        out_path = os.path.join(
            output_dir,
            f"shard_{shard_id:05d}.parquet",
        )

        if os.path.exists(out_path):
            print(f"Skip existing {out_path}")
            continue

        shard = ds.select(range(start, end))
        shard.to_parquet(out_path)

        print(f"Saved {out_path} rows={start}:{end}")


def main():
    parser = argparse.ArgumentParser(description="Download continual-pretraining datasets as Parquet shards.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
        help="Dataset presets to download.",
    )
    parser.add_argument("--output-root", required=True, help="Root directory for Parquet datasets.")
    parser.add_argument("--cache-root", default=None, help="Hugging Face cache root (defaults under output root).")
    parser.add_argument("--shard-size", type=int, default=100_000, help="Rows per Parquet shard.")
    args = parser.parse_args()

    output_root = os.path.abspath(os.path.expanduser(args.output_root))
    cache_root = os.path.abspath(os.path.expanduser(args.cache_root or os.path.join(output_root, ".cache")))

    for dataset_name in args.datasets:
        item = DATASETS[dataset_name]
        save_dataset_to_parquet(
            repo=item["repo"],
            config=item["config"],
            split=item["split"],
            output_dir=os.path.join(output_root, item["name"]),
            cache_dir=os.path.join(cache_root, item["name"]),
            shard_size=args.shard_size,
        )


if __name__ == "__main__":
    main()
