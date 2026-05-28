# fix_export_checkpoint.py

import os
import shutil
import json
from pathlib import Path

from safetensors.torch import load_file, save_file


SRC_DIR = Path("/orange/sgao1/sgao1/continual_pretrain_outputs/tomoe_gated_llama3_8b/checkpoint-20000")
DST_DIR = Path("/orange/sgao1/sgao1/continual_pretrain_outputs/tomoe_gated_llama3_8b/checkpoint-20000-fixed")


def copy_metadata_files(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)

    keep_suffixes = {
        ".json",
        ".txt",
        ".model",
        ".py",
    }

    keep_names = {
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "config.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    }

    for path in src_dir.iterdir():
        if path.name.startswith("model") and path.suffix in {".safetensors", ".bin"}:
            continue

        if path.name in keep_names or path.suffix in keep_suffixes:
            target = dst_dir / path.name
            print(f"[copy] {path} -> {target}")
            shutil.copy2(path, target)


def fix_config(dst_dir: Path):
    config_path = dst_dir / "config.json"

    if not config_path.exists():
        print("[warn] no config.json found")
        return

    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Important for custom model loading
    cfg.setdefault("trust_remote_code", True)

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[update] {config_path}")


def strip_state_dict_prefix(src_dir: Path, dst_dir: Path):
    safetensors_files = sorted(src_dir.glob("*.safetensors"))

    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {src_dir}")

    for sf in safetensors_files:
        if sf.name.startswith("optimizer"):
            continue

        print(f"[load] {sf}")
        state_dict = load_file(str(sf))

        new_state_dict = {}

        for k, v in state_dict.items():
            new_k = k

            if new_k.startswith("model._orig_mod."):
                new_k = new_k.replace("model._orig_mod.", "", 1)
            elif new_k.startswith("_orig_mod."):
                new_k = new_k.replace("_orig_mod.", "", 1)

            new_state_dict[new_k] = v

        out_path = dst_dir / sf.name
        print(f"[save] {out_path}")
        save_file(new_state_dict, str(out_path))


def main():
    print(f"SRC_DIR={SRC_DIR}")
    print(f"DST_DIR={DST_DIR}")

    copy_metadata_files(SRC_DIR, DST_DIR)
    strip_state_dict_prefix(SRC_DIR, DST_DIR)
    fix_config(DST_DIR)

    print("[done]")


if __name__ == "__main__":
    main()