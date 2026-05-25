import os
import shutil
import gc
import json
import math
import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tomoe.hypernetwork import experts_module_list, hn_module_list, hypernetwork
from tomoe.pruning_helper import collect_info_reg_llama, help_functions_hn
from utils import unwrap_model


def load_eval_data(dataset_name: str) -> str:
    if dataset_name == "wikitext":
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(testdata["text"])
    if dataset_name == "ptb":
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test", trust_remote_code=True)
        return "\n\n".join(testdata["sentence"])
    if dataset_name == "c4":
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        return " ".join(testdata[:1100]["text"])
    raise ValueError("invalid dataset name (wikitext, ptb, c4 are allowed)")


@torch.inference_mode()
def evaluate_ppl(model, tokenizer, datasets="wikitext", block_size=2048, max_tokens=524288, device="cuda"):
    model.eval()
    model.to(device)
    if device.startswith("cuda"):
        model.bfloat16()

    for dsname in datasets.split(","):
        text = load_eval_data(dsname)
        encoded_text = tokenizer.encode(text, return_tensors="pt")
        if max_tokens is not None and max_tokens > 0:
            encoded_text = encoded_text[:, :max_tokens]

        nlls = 0.0
        toks = 0
        last_logits_shape = None
        for start in range(0, encoded_text.shape[1] - 1, block_size):
            inp = encoded_text[:, start : start + block_size].to(device=device, dtype=torch.long)
            if inp.shape[1] < 2:
                continue
            output = model(inp)
            logits = output.logits if hasattr(output, "logits") else output
            nll = F.cross_entropy(
                logits[0, :-1],
                inp[0, 1:],
                reduction="sum",
            )
            nlls += float(nll.item())
            toks += inp.shape[1] - 1
            last_logits_shape = tuple(logits.shape)

        if toks == 0:
            raise RuntimeError(f"No evaluation tokens for dataset={dsname}")
        ppl = math.exp(nlls / toks)
        print(f"[ppl] dataset={dsname} tokens={toks} logits_shape={last_logits_shape} ppl={ppl:.4f}")


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


def build_mlp_export_plans(model, width_union_list, hn, dynamic_experts):
    _, hn_experts = get_hn_parts(hn)
    mlp_unions = [item for item in width_union_list if not isinstance(item, int) and item.sum().item() != 0]
    plans = []

    for layer_idx, layer in enumerate(model.model.layers):
        module = layer.mlp
        mid_vector = mlp_unions[layer_idx].to(module.gate_proj.weight.device)
        mid_index = (mid_vector > 0).nonzero(as_tuple=False).view(-1)
        if mid_index.numel() == 0:
            mid_index = torch.argmax(mid_vector).view(1)
        prefix = f"model.layers.{layer_idx}.mlp"
        source_expert = hn_experts.module_list[layer_idx]
        source_eval = source_expert.experts_for_eval[:, mid_index].to(device=mid_index.device)
        source_scores = getattr(source_expert, "binary_approx_for_eval", None)
        if source_scores is None:
            raise RuntimeError(
                "HN expert does not expose binary_approx_for_eval. "
                "Run hn_helper.prepare_for_eval before building export plans."
            )
        source_scores = source_scores[:, mid_index].to(device=mid_index.device)
        selected_indices = []
        for expert_idx in range(dynamic_experts):
            expert_mask = source_eval[expert_idx] > 0
            expert_index = mid_index[expert_mask]
            if expert_index.numel() == 0:
                expert_index = mid_index[source_scores[expert_idx].argmax().view(1)]
            selected_indices.append(expert_index)

        layer_width = max(int(expert_index.numel()) for expert_index in selected_indices)
        expert_indices = []
        for expert_idx, expert_index in enumerate(selected_indices):
            if expert_index.numel() < layer_width:
                selected_mask = torch.zeros(mid_index.numel(), dtype=torch.bool, device=mid_index.device)
                selected_mask[(mid_index[:, None] == expert_index[None, :]).any(dim=1)] = True
                remaining_scores = source_scores[expert_idx].masked_fill(selected_mask, float("-inf"))
                pad_local_index = remaining_scores.topk(
                    k=layer_width - expert_index.numel(),
                    largest=True,
                    sorted=True,
                ).indices
                pad_index = mid_index[pad_local_index]
                expert_index = torch.cat([expert_index, pad_index], dim=0)
            expert_indices.append(expert_index)
        plans.append((layer_idx, prefix, module, expert_indices, source_expert))

    cfgs = [int(plans[layer_idx][3][0].numel()) for layer_idx in range(len(plans))] + [int(dynamic_experts)]
    return plans, cfgs


def count_exported_actual_moe_parameters(model, plans, dynamic_experts):
    mlp_weight_names = set()
    original_mlp_params = 0
    for _, prefix, module, _, _ in plans:
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            weight_name = f"{prefix}.{proj_name}.weight"
            mlp_weight_names.add(weight_name)
            original_mlp_params += module.get_submodule(proj_name).weight.numel()

    dense_with_gated_params = sum(param.numel() for param in model.parameters())
    non_mlp_params = sum(
        param.numel()
        for name, param in model.named_parameters()
        if name not in mlp_weight_names
    )

    moe_router_params = 0
    moe_expert_params = 0
    for _, _, module, expert_indices, source_expert in plans:
        moe_router_params += source_expert.linear_router.weight.numel()
        hidden_size = module.gate_proj.weight.shape[1]
        for expert_idx in range(dynamic_experts):
            width = int(expert_indices[expert_idx].numel())
            moe_expert_params += width * hidden_size
            moe_expert_params += width * hidden_size
            moe_expert_params += hidden_size * width

    final_total_params = non_mlp_params + moe_router_params + moe_expert_params
    gated_attention_params = sum(
        param.numel()
        for name, param in model.named_parameters()
        if "virtual_gated_attn" in name or "gate_module" in name
    )
    return {
        "dense_with_gated_params": dense_with_gated_params,
        "original_dense_mlp_params": original_mlp_params,
        "final_non_mlp_params": non_mlp_params,
        "actual_moe_router_params": moe_router_params,
        "actual_moe_expert_params": moe_expert_params,
        "actual_moe_total_params": final_total_params,
        "gated_attention_params": gated_attention_params,
    }


def print_parameter_report(counts):
    print("[export-parameter-count]")
    for key, value in counts.items():
        print(f"{key}: {value / 1_000_000:.3f}M")


def iter_final_tensors(model, plans, dynamic_experts):
    mlp_prefixes = {prefix for _, prefix, _, _, _ in plans}

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

    for _, prefix, module, expert_indices, source_expert in plans:
        yield f"{prefix}.router.linear_router.weight", source_expert.linear_router.weight.detach()
        for expert_idx in range(dynamic_experts):
            expert_prefix = f"{prefix}.experts.{expert_idx}"
            expert_index = expert_indices[expert_idx]
            yield f"{expert_prefix}.gate_proj.weight", module.gate_proj.weight.detach()[expert_index, :]
            yield f"{expert_prefix}.up_proj.weight", module.up_proj.weight.detach()[expert_index, :]
            yield f"{expert_prefix}.down_proj.weight", module.down_proj.weight.detach()[:, expert_index]


def save_streamed_pretrained(model, plans, dynamic_experts, output_dir, max_shard_size):
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

    for name, tensor in iter_final_tensors(model, plans, dynamic_experts):
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
    test_ppl: bool = False,
    ppl_datasets: str = "wikitext",
    ppl_block_size: int = 2048,
    ppl_max_tokens: int = 524288,
    ppl_device: str = "cuda",
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

    plans, cfgs = build_mlp_export_plans(
        model=model,
        width_union_list=width_union_list,
        hn=hn,
        dynamic_experts=dynamic_experts,
    )
    print_parameter_report(count_exported_actual_moe_parameters(model, plans, dynamic_experts))

    model.config.tomoe_moe_cfgs = cfgs
    model.config.tomoe_gated_attn_rank = gate_rank
    model.config.tomoe_gated_attn_init_bias = gate_init_bias
    model.config.architectures = ["LlamaForCausalLM"]
    model.config.auto_map = {
        "AutoModelForCausalLM": "modeling_llama_tomoe_gated_actual_moe.LlamaForCausalLM",
    }
    os.makedirs(output_dir, exist_ok=True)
    model.config.save_pretrained(output_dir)

    modeling_src = os.path.join(os.path.dirname(__file__), "models", "modeling_llama_tomoe_gated_actual_moe.py")
    shutil.copy2(modeling_src, os.path.join(output_dir, "modeling_llama_tomoe_gated_actual_moe.py"))

    save_streamed_pretrained(
        model=model,
        plans=plans,
        dynamic_experts=dynamic_experts,
        output_dir=output_dir,
        max_shard_size=save_shard_size,
    )

    if save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)

    if test_ppl:
        if ppl_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested ppl_device={ppl_device}, but CUDA is not available.")
        del model, hn, checkpoint, plans, vectors, width_list, width_union_list
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"[ppl] loading exported model from {output_dir}")
        exported_model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": ppl_device} if ppl_device.startswith("cuda") else None,
        )
        evaluate_ppl(
            exported_model,
            tokenizer,
            datasets=ppl_datasets,
            block_size=ppl_block_size,
            max_tokens=ppl_max_tokens,
            device=ppl_device,
        )
    else:
        del hn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
