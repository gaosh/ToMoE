from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

class Wrap(IterableDataset):
    def __init__(self, ds):
        self.ds = ds

    def __iter__(self):
        ds = self.ds
        wi = get_worker_info()
        if wi is not None:
            ds = ds.shard(num_shards=wi.num_workers, index=wi.id)
        for ex in ds:
            yield ex

ds = load_dataset(
    "allenai/tulu-3-sft-mixture",
    split="train",
    revision="b14afda60f1bbebe55d5d2fa1e4df5042f97f8be",
    streaming=True,
)

ds = ds.shuffle(buffer_size=10000, seed=42)
ds = split_dataset_by_node(ds, rank=1, world_size=8)

loader = DataLoader(Wrap(ds), batch_size=1, num_workers=4)

for i, ex in enumerate(loader):
    print(i, ex["id"], flush=True)
    if i >= 1000:
        break