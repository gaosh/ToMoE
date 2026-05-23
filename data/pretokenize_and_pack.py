# data/pretokenize_and_pack.py

import argparse
import glob
import json
import os
import random
from typing import Iterable, List, Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def list_parquet_files(data_dir: str, shuffle_files: bool, seed: int) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))

    if len(files) == 0:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    if shuffle_files:
        rng = random.Random(seed)
        rng.shuffle(files)

    return files


def iter_text_from_parquet(files: List[str], text_column: str = "text") -> Iterable[str]:
    dataset = load_dataset(
        "parquet",
        data_files=files,
        split="train",
        streaming=True,
    )

    for sample in dataset:
        text = sample.get(text_column, None)

        if text is None:
            continue

        if not isinstance(text, str):
            continue

        text = text.strip()

        if len(text) == 0:
            continue

        yield text


def save_packed_shard(
    sequences: List[List[int]],
    output_dir: str,
    shard_id: int,
    seq_len: int,
) -> str:
    arr = np.asarray(sequences, dtype=np.uint32)

    if arr.ndim != 2 or arr.shape[1] != seq_len:
        raise ValueError(
            f"Invalid packed shard shape: {arr.shape}, expected [N, {seq_len}]"
        )

    output_path = os.path.join(output_dir, f"packed_{seq_len}_{shard_id:06d}.npy")
    np.save(output_path, arr)

    return output_path


def write_metadata(
    output_dir: str,
    metadata: dict,
) -> None:
    path = os.path.join(output_dir, "metadata.json")

    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def pretokenize_and_pack(
    model_name: str,
    data_dir: str,
    output_dir: str,
    seq_len: int,
    target_tokens: Optional[int],
    shard_sequences: int,
    max_tokens_per_doc: Optional[int],
    text_column: str,
    add_eos: bool,
    shuffle_files: bool,
    seed: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    eos_token_id = tokenizer.eos_token_id

    files = list_parquet_files(
        data_dir=data_dir,
        shuffle_files=shuffle_files,
        seed=seed,
    )

    print(f"[pretokenize] model_name={model_name}")
    print(f"[pretokenize] data_dir={data_dir}")
    print(f"[pretokenize] output_dir={output_dir}")
    print(f"[pretokenize] num_input_files={len(files)}")
    print(f"[pretokenize] seq_len={seq_len}")
    print(f"[pretokenize] target_tokens={target_tokens}")
    print(f"[pretokenize] shard_sequences={shard_sequences}")
    print(f"[pretokenize] max_tokens_per_doc={max_tokens_per_doc}")
    print(f"[pretokenize] shuffle_files={shuffle_files}")
    print(f"[pretokenize] seed={seed}")

    token_buffer: List[int] = []
    packed_sequences: List[List[int]] = []

    total_raw_tokens = 0
    total_packed_tokens = 0
    total_docs = 0
    shard_id = 0

    progress = tqdm(desc="tokenizing", unit="doc")

    for text in iter_text_from_parquet(files, text_column=text_column):
        total_docs += 1

        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if max_tokens_per_doc is not None and len(ids) > max_tokens_per_doc:
            ids = ids[:max_tokens_per_doc]

        if add_eos and eos_token_id is not None:
            ids.append(eos_token_id)

        if len(ids) == 0:
            progress.update(1)
            continue

        if target_tokens is not None:
            remaining = target_tokens - total_raw_tokens
            if remaining <= 0:
                break
            if len(ids) > remaining:
                ids = ids[:remaining]

        token_buffer.extend(ids)
        total_raw_tokens += len(ids)

        while len(token_buffer) >= seq_len:
            packed_sequences.append(token_buffer[:seq_len])
            token_buffer = token_buffer[seq_len:]
            total_packed_tokens += seq_len

            if len(packed_sequences) >= shard_sequences:
                output_path = save_packed_shard(
                    sequences=packed_sequences,
                    output_dir=output_dir,
                    shard_id=shard_id,
                    seq_len=seq_len,
                )

                print(
                    f"\n[save] {output_path} "
                    f"sequences={len(packed_sequences):,} "
                    f"tokens={len(packed_sequences) * seq_len:,}"
                )

                shard_id += 1
                packed_sequences = []

        progress.set_postfix(
            docs=total_docs,
            raw_tokens=total_raw_tokens,
            packed_tokens=total_packed_tokens,
            shards=shard_id,
        )
        progress.update(1)

        if target_tokens is not None and total_raw_tokens >= target_tokens:
            break

    progress.close()

    if len(packed_sequences) > 0:
        output_path = save_packed_shard(
            sequences=packed_sequences,
            output_dir=output_dir,
            shard_id=shard_id,
            seq_len=seq_len,
        )

        print(
            f"[save] {output_path} "
            f"sequences={len(packed_sequences):,} "
            f"tokens={len(packed_sequences) * seq_len:,}"
        )

        shard_id += 1

    metadata = {
        "model_name": model_name,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "seq_len": seq_len,
        "target_tokens": target_tokens,
        "shard_sequences": shard_sequences,
        "max_tokens_per_doc": max_tokens_per_doc,
        "text_column": text_column,
        "add_eos": add_eos,
        "shuffle_files": shuffle_files,
        "seed": seed,
        "num_input_files": len(files),
        "total_docs": total_docs,
        "total_raw_tokens": total_raw_tokens,
        "total_packed_tokens": total_packed_tokens,
        "num_output_shards": shard_id,
        "dtype": "uint32",
        "format": "npy",
        "shape_per_file": "[num_sequences, seq_len]",
    }

    write_metadata(output_dir, metadata)

    print("[done]")
    print(json.dumps(metadata, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--target_tokens",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--shard_sequences",
        type=int,
        default=8192,
        help="Number of packed sequences per output shard.",
    )

    parser.add_argument(
        "--max_tokens_per_doc",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
    )

    parser.add_argument(
        "--add_eos",
        action="store_true",
    )

    parser.add_argument(
        "--shuffle_files",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pretokenize_and_pack(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        target_tokens=args.target_tokens,
        shard_sequences=args.shard_sequences,
        max_tokens_per_doc=args.max_tokens_per_doc,
        text_column=args.text_column,
        add_eos=args.add_eos,
        shuffle_files=args.shuffle_files,
        seed=args.seed,
    )