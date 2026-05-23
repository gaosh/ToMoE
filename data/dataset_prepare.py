# download_pretrain_datasets.py

import os
from datasets import load_dataset

DATASETS = [
    {
        "name": "fineweb_edu",
        "repo": "HuggingFaceFW/fineweb-edu",
        "config": "sample-100BT",
        "split": "train",
        "output_dir": "/orange/sgao1/sgao1/data/fineweb_edu_100bt",
        "cache_dir": "/orange/sgao1/sgao1/cache/fineweb_edu",
    },
    {
        "name": "openwebmath",
        "repo": "open-web-math/open-web-math",
        "config": None,
        "split": "train",
        "output_dir": "/orange/sgao1/sgao1/openwebmath",
        "cache_dir": "/orange/sgao1/sgao1/cache/openwebmath",
    },
    # {
    #     "name": "dclm_baseline",
    #     "repo": "mlfoundations/dclm-baseline-1.0",
    #     "config": None,
    #     "split": "train",
    #     "output_dir": "/orange/sgao1/sgao1/dclm_baseline",
    #     "cache_dir": "/orange/sgao1/sgao1/cache/dclm_baseline",
    # },
]


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
    for item in DATASETS:
        save_dataset_to_parquet(
            repo=item["repo"],
            config=item["config"],
            split=item["split"],
            output_dir=item["output_dir"],
            cache_dir=item["cache_dir"],
        )


if __name__ == "__main__":
    main()