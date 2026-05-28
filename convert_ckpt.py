# fix_export_checkpoint_v2.py

import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

SRC_DIR = Path("/orange/sgao1/sgao1/continual_pretrain_outputs/tomoe_gated_llama3_8b/checkpoint-20000")
DST_DIR = Path("/orange/sgao1/sgao1/continual_pretrain_outputs/tomoe_gated_llama3_8b/checkpoint-20000-fixed-v2")


def normalize_key(k: str) -> str:
    prefixes = [
        "model._orig_mod.",
        "_orig_mod.",
    ]
    for p in prefixes:
        if k.startswith(p):
            return k[len(p):]
    return k


def copy_non_weight_files():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    for path in SRC_DIR.iterdir():
        if path.suffix == ".safetensors":
            continue
        if path.name.endswith(".bin"):
            continue
        if path.name in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}:
            continue

        dst = DST_DIR / path.name
        if path.is_file():
            shutil.copy2(path, dst)


def convert_safetensors():
    for src_file in sorted(SRC_DIR.glob("*.safetensors")):
        print(f"[convert] {src_file.name}")

        sd = load_file(str(src_file))
        new_sd = {}

        for k, v in sd.items():
            new_k = normalize_key(k)
            if new_k in new_sd:
                raise RuntimeError(f"duplicate key after normalize: {new_k}")
            new_sd[new_k] = v

        dst_file = DST_DIR / src_file.name
        save_file(new_sd, str(dst_file))
        print(f"[save] {dst_file}")


def fix_index_json():
    for index_name in ["model.safetensors.index.json", "pytorch_model.bin.index.json"]:
        src_index = SRC_DIR / index_name
        if not src_index.exists():
            continue

        with open(src_index, "r") as f:
            index = json.load(f)

        if "weight_map" in index:
            new_weight_map = {}
            for k, v in index["weight_map"].items():
                new_k = normalize_key(k)
                if new_k in new_weight_map:
                    raise RuntimeError(f"duplicate index key after normalize: {new_k}")
                new_weight_map[new_k] = v
            index["weight_map"] = new_weight_map

        dst_index = DST_DIR / index_name
        with open(dst_index, "w") as f:
            json.dump(index, f, indent=2)

        print(f"[fix-index] {dst_index}")


def inspect_fixed_keys():
    files = sorted(DST_DIR.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {DST_DIR}")

    sd = load_file(str(files[0]))
    keys = list(sd.keys())

    bad = [k for k in keys if k.startswith("model._orig_mod.") or k.startswith("_orig_mod.")]

    print("[inspect] first 20 keys:")
    for k in keys[:20]:
        print("  ", k)

    if bad:
        raise RuntimeError(f"still found bad keys, examples: {bad[:10]}")

    print("[inspect] no model._orig_mod prefix found")


def test_load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[test] loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(DST_DIR), trust_remote_code=True)

    print("[test] loading model")
    model = AutoModelForCausalLM.from_pretrained(
        str(DST_DIR),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print("[test] loaded model:", type(model))

    inputs = tokenizer("Hello world", return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)

    print("[test] forward ok:", out.logits.shape)


def main():
    print(f"SRC_DIR={SRC_DIR}")
    print(f"DST_DIR={DST_DIR}")

    copy_non_weight_files()
    convert_safetensors()
    fix_index_json()
    inspect_fixed_keys()
    test_load()

    print("[done]")


if __name__ == "__main__":
    main()