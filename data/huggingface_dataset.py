import datasets
from datasets import load_dataset, load_from_disk, concatenate_datasets, interleave_datasets
import os, errno
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

# <<How to save huggingface datasets>>
# Step 1: Use huggingface to download a dataset. 
#   Ex: 
#   from datasets import load_dataset
#   dataset = load_dataset("EleutherAI/the_pile_deduplicated")
# Step 2: Find the cache and upload it to object storage.
#   Ex: scp -r ~/.cache/huggingface/datasets/EleutherAI___parquet [cloud storage path]
# Step 3: Create the function that overlay the cache stored in object storage.
#   Ex: load_hf_dataset_pile_dedup()

def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
            # Change the symlink will cause race condition - so avoid to do it
            # os.remove(link_name)
            # os.symlink(target, link_name)
        else:
            raise e

def load_hf_dataset(hf_path: str, data_cache_dir: Optional[str]=None, default_cache_dir: Optional[str]='~/.cache/huggingface/datasets', **kwargs):
    # This function load a specific cached hf dataset from a different storage
    # This function is useful to load the cached hf dataset from a read-only storage like object storage
    # Huggingface datasets requires the dataset cache folder to be writable, but the object storage is read-only. So we created this function.
    # Example data_cache_dir = '/data-sets/sdp-llm/huggingface/datasets/EleutherAI___the_pile'
    if data_cache_dir is not None:
        sym_name = os.path.basename(data_cache_dir)
        os.makedirs(os.path.expanduser(default_cache_dir), exist_ok=True)
        symlink_force(data_cache_dir, os.path.join(os.path.expanduser(default_cache_dir),sym_name))
    return load_dataset(hf_path, **kwargs)

def load_hf_dataset_pile_dedup(split, n_shards=None):
    # Off-the-shelf function to get the HF pile deduplicated dataset
    if split=='train':
        ds = load_dataset('/data-sets/sdp-text2/the_pile_deduplicated', streaming=True)
        ds = ds['train']
        # ds = load_from_disk('/data-sets/sdp-llm/huggingface/datasets/pile/train')
        ds = ds.select_columns("text")
        # ds = ds.to_iterable_dataset(num_shards=n_shards)
    if split=='validation':
        ds = load_from_disk('/data-sets/sdp-llm/huggingface/datasets/pile/val')
        #ds = ds.select_columns("text")
        ds = ds.to_iterable_dataset(num_shards=n_shards)
    if split=='test':
        ds = load_from_disk('/data-sets/sdp-llm/huggingface/datasets/pile/test')
        ds = ds.select_columns("text")
        ds = ds.to_iterable_dataset(num_shards=n_shards)
    return ds

def load_hf_dataset_slimpajama(split=None, n_shards=None):
    ds = load_dataset('/data-sets/sdp-text2/SlimPajama-627B/'+split, streaming=True)
    ds = ds['train']
    ds = ds.select_columns("text")
    return ds

def load_hf_dataset_korean(split=None, n_shards=None):
    ds = load_dataset('/group-volume/datasets/corpus_sr/corpus_pt_korean_split', streaming=True)
    ds = ds['train']
    ds = ds.select_columns("text")
    return ds

def load_hf_dataset_refinedweb(split=None, n_shards=None):
    ds = load_dataset('/data-sets/sdp-text2/falcon-refinedweb', streaming=True)
    ds = ds['train']
    ds = ds.rename_column('content', 'text')
    ds = ds.select_columns("text")
    return ds

def load_hf_dataset_minipile(split='train', n_shards=None):
    # SJC MLP group dataset
    # train split number of token: 1690681344 (1.7B or ~0.6% of original Pile of ~0.7% or pile_dedup)
    ds = load_dataset("/group-volume/Bixby-Compression/yenchang.hsu/minipile")
    ds = ds[split]
    ds = ds.to_iterable_dataset(num_shards=n_shards)
    return ds

def load_hf_dataset_wiki(split='train', n_shards=None, seed=777 ):
    import os
    
    if os.path.isdir("/group-volume/users/s.gao1/dataset"):
        cache_dir = "/group-volume/users/s.gao1/dataset"
    elif os.path.isdir("/group-volume/Bixby-Compression/s.gao1/datasets"):
        cache_dir = "/group-volume/Bixby-Compression/s.gao1/datasets"
    elif os.path.isdir("/blue/yonghui.wu/sgao1/datasets"):
        cache_dir = "/blue/yonghui.wu/sgao1/datasets"

    wikitext = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir, split="train")
    wikitext = wikitext.remove_columns([col for col in wikitext.column_names if col != "text"])
    raw_datasets = wikitext
    raw_datasets.shuffle(seed=seed)
    # if return_raw:
    #     return raw_datasets
    # else:
    return raw_datasets.to_iterable_dataset(num_shards=n_shards)

def load_hf_dataset_alpaca(n_shards=None, seed=777):
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.select_columns("text").shuffle(seed=seed)
    return ds.to_iterable_dataset(num_shards=n_shards)

def load_hf_dataset_orca_dpo(n_shards=None, seed=777):
    #/orange/yonghui.wu/sgao1/datasets/orca_dpo_pairs.hf
    ds = load_from_disk("/orange/yonghui.wu/sgao1/datasets/orca_dpo_pairs.hf")
    ds = ds.select_columns("text").shuffle(seed=seed)
    return ds.to_iterable_dataset(num_shards=n_shards)

def load_hf_dataset_wizardlMv2(n_shards=None, seed=777):
    ds = load_from_disk("/group-volume/users/s.gao1/saved_model/local_datasets/WizardLM_evol_instruct_V2_196k.hf")
    ds = ds.select_columns("text").shuffle(seed=seed)
    return ds.to_iterable_dataset(num_shards=n_shards)

def format_codealpaca_prompt_batch(batch: Dict[str, List[str]]) -> Dict[str, List[str]]:
    instructions = batch.get("instruction", [])
    inputs = batch.get("input", [])
    outputs = batch.get("output", [])

    texts = []
    for inst, inp, outp in zip(instructions, inputs, outputs):
        inst = inst.strip() if isinstance(inst, str) else ""
        inp = inp.strip() if isinstance(inp, str) else ""
        outp = outp.strip() if isinstance(outp, str) else ""
        if inp:
            txt = (
                "Below is an instruction that describes a task, paired with an input that "
                "provides further context. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{inst}\n\n"
                f"### Input:\n{inp}\n\n"
                f"### Response:\n{outp}"
            )
        else:
            txt = (
                "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{inst}\n\n"
                f"### Response:\n{outp}"
            )
        texts.append(txt)

    return {"text": texts}

def format_prompt_style_batch(batch):
    out = []
    for conv in batch["conversations"]:
        parts = []
        for turn in conv:
            if isinstance(turn, dict):
                role = turn.get("from", "")
                content = turn.get("value", "").strip()
            elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
                role, content = turn[0], turn[1]
            else:
                continue
            if role == "human":
                parts.append(f"### Instruction:\n{content}")
            else:
                parts.append(f"### Response:\n{content}")
        out.append("\n\n".join(parts))
    return {"text": out}

def load_hf_dataset_mixed(n_shards=None, seed=777, splits=[0.25,0.25,0.25,0.25], root_path = "/orange/yonghui.wu/sgao1"):
    os.makedirs(root_path, exist_ok=True)
    cache_dir = os.path.join(root_path, "datasets")
    os.makedirs(cache_dir, exist_ok=True)

    ds1 = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=cache_dir)
    ds1 = ds1.select_columns("text").shuffle(seed=seed)

    ds2 = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir, split="train")
    ds2 = ds2.select_columns("text").shuffle(seed=seed)

    ds3 = load_dataset("sahil2801/CodeAlpaca-20k", cache_dir=cache_dir, split="train")
    ds3 = ds3.map(
    format_codealpaca_prompt_batch,
    batched=True,   # usually fine; use batched=True for speed if examples are large
    desc="Adding text column from conversations")
    # 
    
    print(ds3["text"][0])
    ds3 = ds3.select_columns("text").shuffle(seed=seed)

    ds4 = load_dataset("WizardLMTeam/WizardLM_evol_instruct_V2_196k", cache_dir=cache_dir, split="train")
    # ds4 = load_from_disk("/orange/yonghui.wu/sgao1/datasets/WizardLM_evol_instruct_V2_196k.hf")
    ds4 = ds4.map(format_prompt_style_batch,
    batched=True,   # usually fine; use batched=True for speed if examples are large
    desc="Adding text column from conversations")
    print(ds3["text"][0])
    ds4 = ds4.select_columns("text").shuffle(seed=seed)

    dsc = interleave_datasets([ds1, ds2, ds3, ds4], probabilities=splits, seed=seed)

    return dsc.to_iterable_dataset(num_shards=n_shards)