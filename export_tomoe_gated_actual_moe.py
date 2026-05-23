import os
import shutil
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


def normalize_hn_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "hn" in checkpoint:
        checkpoint = checkpoint["hn"]
    state_dict = OrderedDict()
    for key, value in checkpoint.items():
        name = key
        if name.startswith("module."):
            name = name.replace("module.", "", 1)
        if name.startswith("model_list."):
            name = name.replace("model_list.", "", 1)
        state_dict[name] = value
    return state_dict


def load_gated_attention_state(model, checkpoint):
    if not (isinstance(checkpoint, dict) and "gated_attention" in checkpoint):
        return
    current = model.state_dict()
    updates = {}
    for key, value in checkpoint["gated_attention"].items():
        if key in current:
            updates[key] = value
    model.load_state_dict(updates, strict=False)


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


def convert_mlp_to_actual_moe(model, width_union_list, hn, dynamic_experts):
    from models.modeling_llama_tomoe_gated_actual_moe import LlamaMlpExpert, SingleMlpRouter

    device = next(model.parameters()).device
    mlp_unions = [item for item in width_union_list if not isinstance(item, int) and item.sum().item() != 0]
    cfgs = []
    mlp_index = 0
    moe_index = 0
    for module in model.modules():
        if type(module).__name__ != "LlamaMLP":
            continue
        mid_vector = mlp_unions[mlp_index].to(device)
        mid_index = (mid_vector > 0).nonzero(as_tuple=False).view(-1)
        if mid_index.numel() == 0:
            mid_index = torch.argmax(mid_vector).view(1)
        mid_dim = int(mid_index.numel())

        source_expert = hn[1].module_list[moe_index]
        router = SingleMlpRouter(module.config.hidden_size, experts=dynamic_experts).to(device)
        router.linear_router.weight.data.copy_(source_expert.linear_router.weight.data.to(device))

        expert_modules = torch.nn.ModuleList()
        source_eval = source_expert.experts_for_eval[:, mid_index].to(device=device)
        for expert_idx in range(dynamic_experts):
            expert_mask = source_eval[expert_idx] > 0
            expert_index = mid_index[expert_mask]
            if expert_index.numel() == 0:
                expert_index = mid_index[:1]
            expert = LlamaMlpExpert(
                module.config.hidden_size,
                int(expert_index.numel()),
                module.config.hidden_act,
            ).to(device)
            expert.gate_proj.weight.data.copy_(module.gate_proj.weight.data[expert_index, :])
            expert.up_proj.weight.data.copy_(module.up_proj.weight.data[expert_index, :])
            expert.down_proj.weight.data.copy_(module.down_proj.weight.data[:, expert_index])
            expert_modules.append(expert)

        dense_gate_proj = torch.nn.Linear(module.config.hidden_size, mid_dim, bias=False).to(device)
        dense_up_proj = torch.nn.Linear(module.config.hidden_size, mid_dim, bias=False).to(device)
        dense_down_proj = torch.nn.Linear(mid_dim, module.config.hidden_size, bias=False).to(device)
        dense_gate_proj.weight.data.copy_(module.gate_proj.weight.data[mid_index, :])
        dense_up_proj.weight.data.copy_(module.up_proj.weight.data[mid_index, :])
        dense_down_proj.weight.data.copy_(module.down_proj.weight.data[:, mid_index])

        module.intermediate_size = mid_dim
        module.gate_proj = dense_gate_proj
        module.up_proj = dense_up_proj
        module.down_proj = dense_down_proj
        module.router = router
        module.experts = expert_modules
        module.actual_moe = True
        cfgs.append(mid_dim)
        mlp_index += 1
        moe_index += 1
    return cfgs + [int(dynamic_experts)]


def main(
    hf_model: str = "meta-llama/Meta-Llama-3-8B",
    hn_path: str = "hn_path",
    output_dir: str = "output_path",
    dynamic_experts: int = 8,
    gate_rank: int = 128,
    gate_init_bias: float = 3.0,
    torch_dtype: str = "bfloat16",
    save_tokenizer: bool = True,
):
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[torch_dtype]

    from models.modeling_llama_tomoe_gated_attn import LlamaForCausalLM
    from models.modeling_llama_tomoe_gated_actual_moe import LlamaForCausalLM as FinalLlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(hf_model, torch_dtype=dtype)
    config = AutoConfig.from_pretrained(hf_model)
    attach_gated_attention_modules(model, gate_rank=gate_rank, gate_init_bias=gate_init_bias)

    checkpoint = torch.load(hn_path, map_location="cpu")
    load_gated_attention_state(model, checkpoint)

    hn, hn_helper, param_reg = build_hn_for_model(model, config, dynamic_experts)
    hn.load_state_dict(normalize_hn_state_dict(checkpoint), strict=False)
    hn.eval()
    with torch.no_grad():
        vectors = hn[0]()
        width_list, width_union_list = hn_helper.prepare_for_eval(
            hn[1].module_list,
            vectors,
            non_uniform=True,
            return_vector_union=True,
        )
        param_reg.count_current_params(width_list)

    cfgs = convert_mlp_to_actual_moe(model, width_union_list, hn, dynamic_experts)
    config.tomoe_moe_cfgs = cfgs
    config.tomoe_gated_attn_rank = gate_rank
    config.tomoe_gated_attn_init_bias = gate_init_bias
    config.architectures = ["LlamaForCausalLM"]
    config.auto_map = {
        "AutoModelForCausalLM": "modeling_llama_tomoe_gated_actual_moe.LlamaForCausalLM",
    }

    FinalLlamaForCausalLM.cfgs = cfgs
    final_model = FinalLlamaForCausalLM(config).to(dtype=dtype)
    final_model.load_state_dict(model.state_dict(), strict=False)
    final_model.register_for_auto_class("AutoModelForCausalLM")
    final_model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    modeling_src = os.path.join(os.path.dirname(__file__), "models", "modeling_llama_tomoe_gated_actual_moe.py")
    shutil.copy2(modeling_src, os.path.join(output_dir, "modeling_llama_tomoe_gated_actual_moe.py"))

    if save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
