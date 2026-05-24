import os
import shutil
import gc
import json
import re
from collections import OrderedDict

import torch
from transformers import AutoConfig, AutoTokenizer

from tomoe.hypernetwork import experts_module_list, hn_module_list, hypernetwork
from tomoe.pruning_helper import collect_info_reg_llama, help_functions_hn
from utils import unwrap_model


def infer_attention_metadata(model, config, param_reg=None):
    head_dim = getattr(param_reg, "head_dim", None) or getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads

    num_kv_heads = getattr(param_reg, "num_kv_heads", None) or getattr(config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = config.num_attention_heads
    return head_dim, num_kv_heads


def infer_model_dim(model, config, param_reg=None):
    model_dim = getattr(param_reg, "model_dim", None) or getattr(config, "hidden_size", None)
    if model_dim is None:
        for module in unwrap_model(model).modules():
            if hasattr(module, "hidden_size"):
                return module.hidden_size
    return model_dim


def normalize_hn_state_dict(checkpoint, target_hn):
    if isinstance(checkpoint, dict) and "hn" in checkpoint:
        checkpoint = checkpoint["hn"]
    target_uses_model_list = hasattr(target_hn, "model_list")
    state_dict = OrderedDict()
    for key, value in checkpoint.items():
        name = key
        if name.startswith("module."):
            name = name.replace("module.", "", 1)
        if name.startswith("model_list.") and not target_uses_model_list:
            name = name.replace("model_list.", "", 1)
        state_dict[name] = value
    return state_dict


def get_hn_parts(hn):
    if hasattr(hn, "model_list"):
        return hn.model_list[0], hn.model_list[1]
    return hn[0], hn[1]


def load_gated_attention_state(model, checkpoint):
    if not (isinstance(checkpoint, dict) and "gated_attention" in checkpoint):
        return
    current = {}
    current.update(dict(model.named_parameters()))
    current.update(dict(model.named_buffers()))
    for key, value in checkpoint["gated_attention"].items():
        if key in current:
            current[key].data.copy_(value.to(device=current[key].device, dtype=current[key].dtype))


def attach_gated_attention_modules(model, gate_rank=128, gate_init_bias=3.0):
    from models.modeling_llama_tomoe_gated_actual_moe import SingleGatedAttnModule

    for module in model.modules():
        if type(module).__name__ in ("LlamaAttention", "LlamaFlashAttention2", "LlamaSdpaAttention"):
            module.use_gated_attn = True
            module.virtual_gated_attn.gate_module = SingleGatedAttnModule(
                d_model=module.hidden_size,
                n_heads=module.num_heads,
                head_dim=module.head_dim,
                rank=gate_rank,
                init_bias=gate_init_bias,
            )


def build_hn_for_model(model, config, dynamic_experts):
    param_reg = collect_info_reg_llama(model, p=0.5, lam=1.0)
    head_dim, num_kv_heads = infer_attention_metadata(model, config, param_reg)
    model_dim = infer_model_dim(model, config, param_reg)
    rnn = hypernetwork(t_structures=param_reg.structures, experts=dynamic_experts)
    experts_list = experts_module_list(
        structures=param_reg.structures,
        model_dim=model_dim,
        experts=dynamic_experts,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
    )
    hn = hn_module_list(rnn, experts_list)
    hn_helper = help_functions_hn(param_reg.structures)
    return hn, hn_helper, param_reg


def parse_size_to_bytes(size):
    if isinstance(size, int):
        return size
    text = str(size).strip().upper()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]?B)", text)
    if match is None:
        raise ValueError(f"Invalid size string: {size}")
    value = float(match.group(1))
    unit = match.group(2)
    scale = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
    }[unit]
    return int(value * scale)


def cleanup_old_weight_files(output_dir):
    if not os.path.isdir(output_dir):
        return
    patterns = [
        "model*.safetensors",
        "pytorch_model*.bin",
        "tmp-export-shard-*",
        "*.index.json",
    ]
    for pattern in patterns:
        for filename in os.listdir(output_dir):
            if re.fullmatch(pattern.replace("*", ".*"), filename):
                path = os.path.join(output_dir, filename)
                if os.path.isfile(path):
                    os.remove(path)


def iter_final_tensors(model, width_union_list, hn, dynamic_experts):
    _, hn_experts = get_hn_parts(hn)
    mlp_unions = [item for item in width_union_list if not isinstance(item, int) and item.sum().item() != 0]
    mlp_prefixes = set()
    plans = []

    for layer_idx, layer in enumerate(model.model.layers):
        module = layer.mlp
        mid_vector = mlp_unions[layer_idx].to(module.gate_proj.weight.device)
        mid_index = (mid_vector > 0).nonzero(as_tuple=False).view(-1)
        if mid_index.numel() == 0:
            mid_index = torch.argmax(mid_vector).view(1)
        mid_dim = int(mid_index.numel())
        prefix = f"model.layers.{layer_idx}.mlp"
        mlp_prefixes.add(prefix)
        plans.append((layer_idx, module, mid_index, mid_dim, hn_experts.module_list[layer_idx]))

    for name, tensor in model.state_dict().items():
        skip = False
        for prefix in mlp_prefixes:
            if name in (
                f"{prefix}.gate_proj.weight",
                f"{prefix}.up_proj.weight",
                f"{prefix}.down_proj.weight",
            ):
                skip = True
                break
        if not skip:
            yield name, tensor

    for layer_idx, module, mid_index, mid_dim, source_expert in plans:
        prefix = f"model.layers.{layer_idx}.mlp"
        yield f"{prefix}.router.linear_router.weight", source_expert.linear_router.weight.detach()
        for expert_idx in range(dynamic_experts):
            expert_prefix = f"{prefix}.experts.{expert_idx}"
            yield f"{expert_prefix}.gate_proj.weight", module.gate_proj.weight.detach()[mid_index, :]
            yield f"{expert_prefix}.up_proj.weight", module.up_proj.weight.detach()[mid_index, :]
            yield f"{expert_prefix}.down_proj.weight", module.down_proj.weight.detach()[:, mid_index]

    cfgs = [mid_dim for _, _, _, mid_dim, _ in plans] + [int(dynamic_experts)]
    return cfgs


def save_streamed_pretrained(model, width_union_list, hn, dynamic_experts, output_dir, max_shard_size):
    os.makedirs(output_dir, exist_ok=True)
    cleanup_old_weight_files(output_dir)

    max_shard_bytes = parse_size_to_bytes(max_shard_size)
    shard_paths = []
    shard_keys = []
    shard = OrderedDict()
    shard_size = 0
    total_size = 0

    try:
        from safetensors.torch import save_file as save_safetensors_file

        safe_serialization = True
    except Exception:
        save_safetensors_file = None
        safe_serialization = False

    def tensor_nbytes(tensor):
        return tensor.numel() * tensor.element_size()

    def flush_shard():
        nonlocal shard, shard_size
        if not shard:
            return
        shard_id = len(shard_paths) + 1
        suffix = "safetensors" if safe_serialization else "bin"
        tmp_name = f"tmp-export-shard-{shard_id:05d}.{suffix}"
        tmp_path = os.path.join(output_dir, tmp_name)
        if safe_serialization:
            save_safetensors_file(shard, tmp_path, metadata={"format": "pt"})
        else:
            torch.save(shard, tmp_path)
        shard_paths.append(tmp_path)
        shard_keys.append(list(shard.keys()))
        shard = OrderedDict()
        shard_size = 0
        gc.collect()

    cfgs = None
    tensor_iter = iter_final_tensors(model, width_union_list, hn, dynamic_experts)
    while True:
        try:
            name, tensor = next(tensor_iter)
        except StopIteration as stop:
            cfgs = stop.value
            break
        tensor = tensor.detach().cpu().contiguous()
        nbytes = tensor_nbytes(tensor)
        if shard and shard_size + nbytes > max_shard_bytes:
            flush_shard()
        shard[name] = tensor
        shard_size += nbytes
        total_size += nbytes

    flush_shard()

    total_shards = len(shard_paths)
    final_weight_map = {}
    for shard_idx, tmp_path in enumerate(shard_paths, start=1):
        if safe_serialization:
            final_name = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
        else:
            final_name = f"pytorch_model-{shard_idx:05d}-of-{total_shards:05d}.bin"
        final_path = os.path.join(output_dir, final_name)
        os.replace(tmp_path, final_path)
        for key in shard_keys[shard_idx - 1]:
            final_weight_map[key] = final_name

    if safe_serialization:
        index_name = "model.safetensors.index.json"
    else:
        index_name = "pytorch_model.bin.index.json"
    with open(os.path.join(output_dir, index_name), "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {"total_size": total_size},
                "weight_map": final_weight_map,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    return cfgs


def main(
    hf_model: str = "meta-llama/Meta-Llama-3-8B",
    hn_path: str = "hn_path",
    output_dir: str = "output_path",
    dynamic_experts: int = 8,
    gate_rank: int = 128,
    gate_init_bias: float = 3.0,
    torch_dtype: str = "bfloat16",
    save_tokenizer: bool = True,
    low_cpu_mem_usage: bool = True,
    save_shard_size: str = "2GB",
):
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[torch_dtype]

    from models.modeling_llama_tomoe_gated_attn import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(
        hf_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    config = AutoConfig.from_pretrained(hf_model)
    attach_gated_attention_modules(model, gate_rank=gate_rank, gate_init_bias=gate_init_bias)

    checkpoint = torch.load(hn_path, map_location="cpu")
    load_gated_attention_state(model, checkpoint)

    hn, hn_helper, param_reg = build_hn_for_model(model, config, dynamic_experts)
    load_result = hn.load_state_dict(normalize_hn_state_dict(checkpoint, hn), strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print("[hn-load]")
        print(f"missing_keys: {len(load_result.missing_keys)}")
        print(f"unexpected_keys: {len(load_result.unexpected_keys)}")
        print(f"missing_keys_sample: {load_result.missing_keys[:5]}")
        print(f"unexpected_keys_sample: {load_result.unexpected_keys[:5]}")
    if len(load_result.missing_keys) > 0 and len(load_result.unexpected_keys) > 0:
        raise RuntimeError("HN checkpoint did not match the export HN structure; refusing to export with an untrained HN.")
    hn.eval()
    hn_rnn, hn_experts = get_hn_parts(hn)
    with torch.no_grad():
        vectors = hn_rnn()
        width_list, width_union_list = hn_helper.prepare_for_eval(
            hn_experts.module_list,
            vectors,
            non_uniform=True,
            return_vector_union=True,
        )
        param_reg.count_current_params(width_list)

    cfgs = save_streamed_pretrained(
        model=model,
        width_union_list=width_union_list,
        hn=hn,
        dynamic_experts=dynamic_experts,
        output_dir=output_dir,
        max_shard_size=save_shard_size,
    )
    del hn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.config.tomoe_moe_cfgs = cfgs
    model.config.tomoe_gated_attn_rank = gate_rank
    model.config.tomoe_gated_attn_init_bias = gate_init_bias
    model.config.architectures = ["LlamaForCausalLM"]
    model.config.auto_map = {
        "AutoModelForCausalLM": "modeling_llama_tomoe_gated_actual_moe.LlamaForCausalLM",
    }

    model.config.save_pretrained(output_dir)

    modeling_src = os.path.join(os.path.dirname(__file__), "models", "modeling_llama_tomoe_gated_actual_moe.py")
    shutil.copy2(modeling_src, os.path.join(output_dir, "modeling_llama_tomoe_gated_actual_moe.py"))

    if save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
