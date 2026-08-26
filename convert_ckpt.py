import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

def normalize_key(k: str) -> str:
    prefixes = [
        "model._orig_mod.",
        "_orig_mod.",
    ]
    for p in prefixes:
        if k.startswith(p):
            return k[len(p):]
    return k


def copy_non_weight_files(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)

    for path in src_dir.iterdir():
        if path.suffix == ".safetensors":
            continue
        if path.name.endswith(".bin"):
            continue
        if path.name in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}:
            continue

        dst = dst_dir / path.name
        if path.is_file():
            shutil.copy2(path, dst)


def convert_safetensors(src_dir: Path, dst_dir: Path):
    src_files = sorted(src_dir.glob("*.safetensors"))
    if not src_files:
        raise FileNotFoundError(f"No safetensors files found in {src_dir}")

    for src_file in src_files:
        print(f"[convert] {src_file.name}")

        sd = load_file(str(src_file))
        new_sd = {}

        for k, v in sd.items():
            new_k = normalize_key(k)
            if new_k in new_sd:
                raise RuntimeError(f"duplicate key after normalize: {new_k}")
            new_sd[new_k] = v

        dst_file = dst_dir / src_file.name
        save_file(new_sd, str(dst_file))
        print(f"[save] {dst_file}")


def fix_index_json(src_dir: Path, dst_dir: Path):
    for index_name in ["model.safetensors.index.json", "pytorch_model.bin.index.json"]:
        src_index = src_dir / index_name
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

        dst_index = dst_dir / index_name
        with open(dst_index, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        print(f"[fix-index] {dst_index}")


def inspect_fixed_keys(dst_dir: Path):
    files = sorted(dst_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {dst_dir}")

    sd = load_file(str(files[0]))
    keys = list(sd.keys())

    bad = [k for k in keys if k.startswith("model._orig_mod.") or k.startswith("_orig_mod.")]

    print("[inspect] first 20 keys:")
    for k in keys[:20]:
        print("  ", k)

    if bad:
        raise RuntimeError(f"still found bad keys, examples: {bad[:10]}")

    print("[inspect] no model._orig_mod prefix found")


def test_load(dst_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[test] loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(dst_dir), trust_remote_code=True)

    print("[test] loading model")
    model = AutoModelForCausalLM.from_pretrained(
        str(dst_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print("[test] loaded model:", type(model))

    inputs = tokenizer("Hello world", return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)

    print("[test] forward ok:", out.logits.shape)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove torch.compile wrapper prefixes from a saved Hugging Face checkpoint."
    )
    parser.add_argument("--src-dir", type=Path, required=True, help="Source checkpoint directory.")
    parser.add_argument("--dst-dir", type=Path, required=True, help="Destination checkpoint directory.")
    parser.add_argument("--test-load", action="store_true", help="Load the converted model and run one forward pass.")
    return parser.parse_args()


def main():
    args = parse_args()
    src_dir = args.src_dir.expanduser().resolve()
    dst_dir = args.dst_dir.expanduser().resolve()

    if not src_dir.is_dir():
        raise NotADirectoryError(f"Source checkpoint directory does not exist: {src_dir}")
    if src_dir == dst_dir:
        raise ValueError("--src-dir and --dst-dir must be different directories")

    print(f"SRC_DIR={src_dir}")
    print(f"DST_DIR={dst_dir}")

    copy_non_weight_files(src_dir, dst_dir)
    convert_safetensors(src_dir, dst_dir)
    fix_index_json(src_dir, dst_dir)
    inspect_fixed_keys(dst_dir)
    if args.test_load:
        test_load(dst_dir)

    print("[done]")


if __name__ == "__main__":
    main()
