import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM


def normalize_tomoe_layer_widths(cfgs):
    if cfgs is None:
        return None, []
    if isinstance(cfgs, dict):
        num_experts = int(cfgs["num_experts"])
        widths = []
        for layer_cfg in cfgs["layers"]:
            if "width" in layer_cfg:
                widths.append(int(layer_cfg["width"]))
            elif "expert_widths" in layer_cfg:
                widths.append(max(int(width) for width in layer_cfg["expert_widths"]))
            elif "union_width" in layer_cfg:
                widths.append(int(layer_cfg["union_width"]))
            else:
                raise ValueError(f"Unsupported tomoe_moe_cfgs layer entry: {layer_cfg}")
        return num_experts, widths
    return int(cfgs[-1]), [int(width) for width in cfgs[:-1]]


def preflight(model_dir):
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    cfgs = config.get("tomoe_moe_cfgs")
    num_experts, widths = normalize_tomoe_layer_widths(cfgs)
    print(f"[config] num_layers={len(widths)} num_experts={num_experts}")
    print(f"[config] first_widths={widths[:8]}")

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print("[preflight] no model.safetensors.index.json found; skipping shard shape check")
        return

    from safetensors import safe_open

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    for layer_idx, width in enumerate(widths[:2]):
        for expert_idx in range(min(num_experts, 2)):
            prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            for proj, dim in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
                key = f"{prefix}.{proj}.weight"
                shard = weight_map[key]
                with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as handle:
                    shape = tuple(handle.get_slice(key).get_shape())
                print(f"[shape] {key}: {shape}")
                if shape[dim] != width:
                    raise RuntimeError(f"{key} shape {shape} does not match config width {width}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.torch_dtype]

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")

    preflight(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": args.device} if args.device.startswith("cuda") else None,
    )
    if not args.device.startswith("cuda"):
        model.to(args.device)
    model.eval()

    first_mlp = model.model.layers[0].mlp
    print(f"[loaded] class={model.__class__.__name__}")
    print(f"[loaded] layer0 num_experts={len(first_mlp.experts)}")
    for idx, expert in enumerate(first_mlp.experts[:2]):
        print(
            f"[loaded] layer0 expert{idx}: "
            f"gate={tuple(expert.gate_proj.weight.shape)} "
            f"up={tuple(expert.up_proj.weight.shape)} "
            f"down={tuple(expert.down_proj.weight.shape)}"
        )


if __name__ == "__main__":
    main()
