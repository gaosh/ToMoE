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
        self.num_sequences_per_shard = []

        for f in self.files:
            n_tokens = self._get_num_tokens(f)
            n_seq = (n_tokens - 1) // seq_len
            self.shard_lens.append(n_tokens)
            self.num_sequences_per_shard.append(n_seq)

        self.cum_sequences = np.cumsum(self.num_sequences_per_shard)
        self.total_sequences = int(self.cum_sequences[-1])

        self._cache_file = None
        self._cache_data = None

    def _get_num_tokens(self, path):
        if path.endswith(".npy"):
            arr = np.load(path, mmap_mode="r")
            return len(arr)

        if path.endswith(".bin"):
            return os.path.getsize(path) // np.dtype(self.dtype).itemsize

        if path.endswith(".pt"):
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                if "input_ids" in obj:
                    return len(obj["input_ids"])
                if "tokens" in obj:
                    return len(obj["tokens"])
            return len(obj)

        raise ValueError(f"Unsupported file type: {path}")

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
        start = local_idx * self.seq_len
        end = start + self.seq_len + 1

        shard = self._load_shard(self.files[shard_idx])
        tokens = np.asarray(shard[start:end], dtype=np.int64)

        x = torch.from_numpy(tokens[:-1].copy()).long()
        y = torch.from_numpy(tokens[1:].copy()).long()

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