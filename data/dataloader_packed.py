# dataloader.py
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PackedTokenDataset(Dataset):
    """
    Load pre-tokenized packed shards for causal LM pretraining.

    Supports:
      - .npy: numpy array of token ids
      - .bin: raw uint16/int32/int64 token ids
      - .pt: torch tensor or dict containing input_ids/tokens
    """

    def __init__(
        self,
        data_dir,
        seq_len=8192,
        file_pattern="*.npy",
        dtype=np.uint32,
        shuffle_shards=True,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.dtype = dtype

        self.files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        if len(self.files) == 0:
            raise ValueError(f"No shard files found in {data_dir} with pattern {file_pattern}")

        if shuffle_shards:
            random.shuffle(self.files)

        self.shard_lens = []
        self.shard_shapes = []
        self.shard_formats = []
        self.num_sequences_per_shard = []

        for f in self.files:
            shape, shard_format, n_seq = self._inspect_shard(f)
            self.shard_shapes.append(shape)
            self.shard_formats.append(shard_format)
            self.shard_lens.append(int(np.prod(shape)))
            self.num_sequences_per_shard.append(n_seq)

        self.cum_sequences = np.cumsum(self.num_sequences_per_shard)
        self.total_sequences = int(self.cum_sequences[-1])
        if self.total_sequences == 0:
            raise RuntimeError(
                "Dataset contains zero sequences. "
                "Check shard format and seq_len."
            )
        example_shape = self.shard_shapes[0]
        estimated_tokens = self.total_sequences * self.seq_len
        print("[dataset]")
        print(f"num_shards={len(self.files)}")
        print(f"example_shape={example_shape}")
        print(f"total_sequences={self.total_sequences}")
        print(f"estimated_tokens={estimated_tokens}")

        self._cache_file = None
        self._cache_data = None

    def _inspect_shard(self, path):
        if path.endswith(".npy"):
            arr = np.load(path, mmap_mode="r")
            return self._shape_to_sequence_info(path, arr.shape)

        if path.endswith(".bin"):
            n_tokens = os.path.getsize(path) // np.dtype(self.dtype).itemsize
            return (n_tokens,), "stream_1d", (n_tokens - 1) // self.seq_len

        if path.endswith(".pt"):
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                if "input_ids" in obj:
                    obj = obj["input_ids"]
                elif "tokens" in obj:
                    obj = obj["tokens"]
                else:
                    raise ValueError(f"Unsupported .pt dict keys in {path}")
            shape = tuple(obj.shape) if torch.is_tensor(obj) else np.asarray(obj).shape
            return self._shape_to_sequence_info(path, shape)

        raise ValueError(f"Unsupported file type: {path}")

    def _shape_to_sequence_info(self, path, shape):
        shape = tuple(shape)
        if len(shape) == 1:
            n_tokens = shape[0]
            return shape, "stream_1d", (n_tokens - 1) // self.seq_len
        if len(shape) == 2:
            if shape[1] != self.seq_len:
                raise ValueError(
                    f"Packed shard {path} has shape {shape}; expected second dimension "
                    f"to match seq_len={self.seq_len}"
                )
            return shape, "packed_2d", shape[0]
        raise ValueError(f"Unsupported shard shape for {path}: {shape}")

    def _load_shard(self, path):
        if self._cache_file == path:
            return self._cache_data

        if path.endswith(".npy"):
            data = np.load(path, mmap_mode="r")

        elif path.endswith(".bin"):
            data = np.memmap(path, dtype=self.dtype, mode="r")

        elif path.endswith(".pt"):
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                obj = obj.get("input_ids", obj.get("tokens"))
            data = obj.numpy() if torch.is_tensor(obj) else np.asarray(obj)

        else:
            raise ValueError(f"Unsupported file type: {path}")

        self._cache_file = path
        self._cache_data = data
        return data

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        shard_idx = int(np.searchsorted(self.cum_sequences, idx, side="right"))
        prev = 0 if shard_idx == 0 else self.cum_sequences[shard_idx - 1]

        local_idx = idx - prev

        shard = self._load_shard(self.files[shard_idx])
        shard_format = self.shard_formats[shard_idx]

        if shard_format == "stream_1d":
            start = local_idx * self.seq_len
            end = start + self.seq_len + 1
            tokens = np.asarray(shard[start:end], dtype=np.int64)
            x = torch.from_numpy(tokens[:-1].copy()).long()
            y = torch.from_numpy(tokens[1:].copy()).long()
        elif shard_format == "packed_2d":
            tokens = np.asarray(shard[local_idx], dtype=np.int64)
            x = torch.from_numpy(tokens.copy()).long()
            y = x.clone()
        else:
            raise ValueError(f"Unsupported shard format: {shard_format}")


        return {
            "input_ids": x,
            "labels": y,
        }


def create_dataloader(
    data_dir,
    seq_len=8192,
    batch_size=1,
    num_workers=4,
    file_pattern="*.npy",
    dtype=np.uint32,
    shuffle=True,
):
    dataset = PackedTokenDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        file_pattern=file_pattern,
        dtype=dtype,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    return loader
