# data/pretokenize_and_pack.py

import argparse
import glob
import json
import os
import random
from typing import Iterable, Iterator, List, Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def list_parquet_files(
    data_dir: str,
    shuffle_files: bool,
    seed: int,
) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))

    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    if shuffle_files:
        rng = random.Random(seed)
        rng.shuffle(files)

    return files


def iter_text_from_parquet(
    files: List[str],
    text_column: str = "text",
) -> Iterator[str]:
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

        if not text:
            continue

        yield text


def iter_text_batches(
    text_iter: Iterable[str],
    batch_size: int,
) -> Iterator[List[str]]:
    batch = []

    for text in text_iter:
        batch.append(text)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def save_packed_shard(
    sequences: List[List[int]],
    output_dir: str,
    shard_id: int,
    seq_len: int,
) -> str:
    arr = np.asarray(sequences, dtype=np.uint32)

    if arr.ndim != 2 or arr.shape[1] != seq_len:
        raise ValueError(
            f"Invalid shard shape {arr.shape}; expected [N, {seq_len}]"
        )

    output_path = os.path.join(
        output_dir,
        f"packed_{seq_len}_{shard_id:06d}.npy",
    )

    np.save(output_path, arr)

    return output_path


def write_metadata(
    output_dir: str,
    metadata: dict,
) -> None:
    metadata_path = os.path.join(output_dir, "metadata.json")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def maybe_trim_to_target(
    ids: List[int],
    total_raw_tokens: int,
    target_tokens: Optional[int],
) -> List[int]:
    if target_tokens is None:
        return ids

    remaining = target_tokens - total_raw_tokens

    if remaining <= 0:
        return []

    if len(ids) > remaining:
        return ids[:remaining]

    return ids


def pack_tokens_from_ids(
    ids: List[int],
    token_buffer: List[int],
    packed_sequences: List[List[int]],
    seq_len: int,
) -> int:
    token_buffer.extend(ids)

    newly_packed_tokens = 0

    while len(token_buffer) >= seq_len:
        packed_sequences.append(token_buffer[:seq_len])
        del token_buffer[:seq_len]
        newly_packed_tokens += seq_len

    return newly_packed_tokens


def pretokenize_and_pack(
    model_name: str,
    data_dir: str,
    output_dir: str,
    seq_len: int,
    target_tokens: Optional[int],
    shard_sequences: int,
    batch_size: int,
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

    if not getattr(tokenizer, "is_fast", False):
        print("[warning] tokenizer is not a fast tokenizer; batch tokenization may be slow.")

    eos_token_id = tokenizer.eos_token_id

    files = list_parquet_files(
        data_dir=data_dir,
        shuffle_files=shuffle_files,
        seed=seed,
    )

    print("[pretokenize] configuration")
    print(f"  model_name          : {model_name}")
    print(f"  data_dir            : {data_dir}")
    print(f"  output_dir          : {output_dir}")
    print(f"  num_input_files     : {len(files)}")
    print(f"  seq_len             : {seq_len}")
    print(f"  target_tokens       : {target_tokens}")
    print(f"  shard_sequences     : {shard_sequences}")
    print(f"  batch_size          : {batch_size}")
    print(f"  max_tokens_per_doc  : {max_tokens_per_doc}")
    print(f"  text_column         : {text_column}")
    print(f"  add_eos             : {add_eos}")
    print(f"  shuffle_files       : {shuffle_files}")
    print(f"  seed                : {seed}")
    print(f"  eos_token_id        : {eos_token_id}")

    token_buffer: List[int] = []
    packed_sequences: List[List[int]] = []

    total_docs = 0
    total_raw_tokens = 0
    total_packed_tokens = 0
    shard_id = 0

    text_iter = iter_text_from_parquet(
        files=files,
        text_column=text_column,
    )

    batch_iter = iter_text_batches(
        text_iter=text_iter,
        batch_size=batch_size,
    )

    progress = tqdm(
        batch_iter,
        desc="tokenizing",
        unit="batch",
    )

    should_stop = False

    for texts in progress:
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        for ids in encoded["input_ids"]:
            if max_tokens_per_doc is not None and len(ids) > max_tokens_per_doc:
                ids = ids[:max_tokens_per_doc]

            if add_eos and eos_token_id is not None:
                ids = ids + [eos_token_id]

            ids = maybe_trim_to_target(
                ids=ids,
                total_raw_tokens=total_raw_tokens,
                target_tokens=target_tokens,
            )

            if not ids:
                should_stop = True
                break

            total_docs += 1
            total_raw_tokens += len(ids)

            newly_packed = pack_tokens_from_ids(
                ids=ids,
                token_buffer=token_buffer,
                packed_sequences=packed_sequences,
                seq_len=seq_len,
            )

            total_packed_tokens += newly_packed

            while len(packed_sequences) >= shard_sequences:
                shard_to_save = packed_sequences[:shard_sequences]
                del packed_sequences[:shard_sequences]

                output_path = save_packed_shard(
                    sequences=shard_to_save,
                    output_dir=output_dir,
                    shard_id=shard_id,
                    seq_len=seq_len,
                )

                print(
                    f"\n[save] {output_path} "
                    f"sequences={len(shard_to_save):,} "
                    f"tokens={len(shard_to_save) * seq_len:,}"
                )

                shard_id += 1

            progress.set_postfix(
                docs=total_docs,
                raw_tokens=f"{total_raw_tokens:.3e}",
                packed_tokens=f"{total_packed_tokens:.3e}",
                shards=shard_id,
            )

            if target_tokens is not None and total_raw_tokens >= target_tokens:
                should_stop = True
                break

        if should_stop:
            break

    progress.close()

    if packed_sequences:
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

    dropped_tail_tokens = len(token_buffer)

    metadata = {
        "model_name": model_name,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "seq_len": seq_len,
        "target_tokens": target_tokens,
        "shard_sequences": shard_sequences,
        "batch_size": batch_size,
        "max_tokens_per_doc": max_tokens_per_doc,
        "text_column": text_column,
        "add_eos": add_eos,
        "shuffle_files": shuffle_files,
        "seed": seed,
        "num_input_files": len(files),
        "total_docs": total_docs,
        "total_raw_tokens": total_raw_tokens,
        "total_packed_tokens": total_packed_tokens,
        "dropped_tail_tokens": dropped_tail_tokens,
        "num_output_shards": shard_id,
        "dtype": "uint32",
        "format": "npy",
        "shape_per_file": "[num_sequences, seq_len]",
    }

    write_metadata(
        output_dir=output_dir,
        metadata=metadata,
    )

    print("[done]")
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretokenize parquet text datasets and pack into fixed-length npy shards."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Tokenizer/model name, e.g. meta-llama/Meta-Llama-3-8B.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing parquet files.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save packed npy shards.",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=8192,
        help="Packed sequence length.",
    )

    parser.add_argument(
        "--target_tokens",
        type=int,
        default=None,
        help="Stop after this many raw tokens. Use None for full dataset.",
    )

    parser.add_argument(
        "--shard_sequences",
        type=int,
        default=4096,
        help="Number of packed sequences per output shard.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Number of documents per tokenizer batch.",
    )

    parser.add_argument(
        "--max_tokens_per_doc",
        type=int,
        default=None,
        help="Optionally truncate each document to this many tokens.",
    )

    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Text column name in parquet files.",
    )

    parser.add_argument(
        "--add_eos",
        action="store_true",
        help="Append EOS token after each document.",
    )

    parser.add_argument(
        "--shuffle_files",
        action="store_true",
        help="Shuffle parquet files before streaming.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for file shuffling.",
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
        batch_size=args.batch_size,
        max_tokens_per_doc=args.max_tokens_per_doc,
        text_column=args.text_column,
        add_eos=args.add_eos,
        shuffle_files=args.shuffle_files,
        seed=args.seed,
    )