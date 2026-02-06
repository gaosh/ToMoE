import math
from collections import OrderedDict

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from models.modeling_llama_dpmoe import LlamaForCausalLM
from models.modeling_llama_moe_final import (
    LlamaForCausalLM as LlamaMoEForCausalLM,
    single_experts_module,
)
from tomoe.hypernetwork import experts_module_list, hypernetwork
from tomoe.pruning_helper import collect_info_reg_llama, help_functions_hn
from utils.unwrap import unwrap_model


def evaluate(model, tokenizer, datasets="wiki", hn_helper=None):
    model.eval()
    model.cuda()
    model.bfloat16()

    total_toks = 0
    for dsname in datasets.split(","):
        test_string = load_eval_data(dsname)
        encoded_text = tokenizer.encode(test_string, return_tensors="pt")
        encoded_text = encoded_text[:, : 256 * 2048]

        nlls = 0
        toks = 0
        with torch.inference_mode():
            block_size = 2048
            for i in tqdm.tqdm(range(0, encoded_text.shape[1], block_size)):
                inp = encoded_text[:, i : i + block_size]

                model_output = model(inp.cuda().to(dtype=torch.long))
                logits = model_output.logits if hasattr(model_output, "logits") else model_output
                nll = torch.nn.functional.cross_entropy(
                    logits[0, :-1],
                    inp[0, 1:].to(dtype=torch.long).cuda(),
                    reduction="sum",
                )
                toks += inp.size(1) - 1
                nlls += nll.item()
                if hn_helper is not None:
                    hn_helper.accumlate_router_logits(model)

        print(encoded_text.shape, logits.shape)
        encoded_text = encoded_text[:, : logits.shape[0]]
        ppl = math.exp(nlls / toks)
        print(f"Perplexity on {dsname}: {ppl:.2f}")
        total_toks += toks


def load_eval_data(dataset_name: str) -> str:
    if dataset_name == "wikitext":
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testdata = "\n\n".join(testdata["text"])
    elif dataset_name == "ptb":
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test", trust_remote_code=True)
        testdata = "\n\n".join(testdata["sentence"])
    elif dataset_name == "c4":
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        testdata = " ".join(testdata[:1100]["text"])
    else:
        raise ValueError("invalid dataset name (wikitext, ptb, c4 are allowed)")
    return testdata


def convert_to_moe_llama(model, truncated_union_list, hn, dynamic_experts, attn_prune=False):
    cfgs = []
    device = next(model.parameters()).device

    mlp_index = 0
    moe_index = 0
    modules = list(model.modules())
    for m in modules:
        if type(m).__name__ in ("LlamaFlashAttention2", "LlamaAttention", "LlamaSdpaAttention"):
            attn_vector = hn[1].module_list[moe_index].experts_for_eval

            expert_module = single_experts_module(
                m.head_dim,
                m.hidden_size,
                head_dim=m.head_dim,
                attn_flag=True,
                experts=dynamic_experts,
            )

            # expert_weight = hn[1].module_list[moe_index].linear_router.weight.data
            # eval_experts = hn[1].module_list[moe_index].experts_for_eval

            expert_weight = hn[1].module_list[moe_index].linear_router.weight.data
            decoder_weight = hn[1].module_list[moe_index].linear_decoder.weight.data[:m.head_dim,:]
            ln_weight = hn[1].module_list[moe_index].ln.weight.data
            ln_bias = hn[1].module_list[moe_index].ln.bias.data

            rnn_state_buffer =  hn[1].module_list[moe_index].rnn_state.mean(dim=0)

            expert_module.linear_router.weight.data.copy_(expert_weight)
            expert_module.linear_decoder.weight.data.copy_(decoder_weight)
            expert_module.ln.weight.data.copy_(ln_weight)
            expert_module.ln.bias.data.copy_(ln_bias)
            expert_module.rnn_state.copy_(rnn_state_buffer)

            if attn_prune and attn_vector is not None:
                attn_vector = attn_vector.to(device)
                if attn_vector.ndim != 1 or attn_vector.numel() != m.head_dim:
                    attn_vector = torch.ones(m.head_dim, dtype=attn_vector.dtype, device=device)
                select_index = (attn_vector > 0).nonzero().squeeze()
                if select_index.numel() == 0:
                    select_index = torch.argmax(attn_vector).view(1)
                pruned_head_dim = int(select_index.numel())

                q_proj = torch.nn.Linear(m.hidden_size, m.num_heads * pruned_head_dim, bias=m.q_proj.bias is not None).to(device)
                k_proj = torch.nn.Linear(
                    m.hidden_size, m.num_key_value_heads * pruned_head_dim, bias=m.k_proj.bias is not None
                ).to(device)

                q_weight = m.q_proj.weight.data.view(m.num_heads, m.head_dim, m.hidden_size)
                k_weight = m.k_proj.weight.data.view(m.num_key_value_heads, m.head_dim, m.hidden_size)
                q_weight_pruned = q_weight[:, select_index, :]
                k_weight_pruned = k_weight[:, select_index, :]
                q_proj.weight.data.copy_(q_weight_pruned.contiguous().view(m.num_heads * pruned_head_dim, m.hidden_size))
                k_proj.weight.data.copy_(
                    k_weight_pruned.contiguous().view(m.num_key_value_heads * pruned_head_dim, m.hidden_size)
                )

                if m.q_proj.bias is not None:
                    q_bias = m.q_proj.bias.data.view(m.num_heads, m.head_dim)
                    q_bias_pruned = q_bias[:, select_index]
                    q_proj.bias.data.copy_(q_bias_pruned.contiguous().view(m.num_heads * pruned_head_dim))

                if m.k_proj.bias is not None:
                    k_bias = m.k_proj.bias.data.view(m.num_key_value_heads, m.head_dim)
                    k_bias_pruned = k_bias[:, select_index]
                    k_proj.bias.data.copy_(k_bias_pruned.contiguous().view(m.num_key_value_heads * pruned_head_dim))

                m.q_proj = q_proj
                m.k_proj = k_proj

                mask = torch.zeros(1, m.head_dim, dtype=torch.uint8, device=device)
                mask[:, select_index] = 1
                expert_module.experts_for_eval.copy_(mask)
                expert_module.register_buffer('qk_index', torch.zeros(int(mask.sum())).to(torch.int64))
                expert_module.qk_index.copy_(select_index.to(torch.int64))
                
                m.experts_module = expert_module
                cfgs.append(pruned_head_dim)
            # else:
            #     expert_module.experts_for_eval.copy_(
            #         torch.ones(dynamic_experts + 1, m.head_dim, dtype=torch.uint8, device=device)
            #     )
            #     expert_module.qk_index.copy_(torch.arange(m.head_dim, device=device))
            #     m.experts_module = expert_module
            #     cfgs.append(m.head_dim)
            moe_index += 1

        if type(m).__name__ == "LlamaMLP":
            mid_vector = truncated_union_list[mlp_index]
            mid_index = (mid_vector > 0).nonzero(as_tuple=False).view(-1)
            if mid_index.numel() == 0:
                mid_index = torch.argmax(mid_vector).view(1)
            mid_dim = mid_index.numel()
            

            gate_proj = torch.nn.Linear(in_features=m.config.hidden_size, out_features=mid_dim, bias=False).to(device)
            up_proj = torch.nn.Linear(in_features=m.config.hidden_size, out_features=mid_dim, bias=False).to(device)
            down_proj = torch.nn.Linear(in_features=mid_dim, out_features=m.config.hidden_size, bias=False).to(device)

            gate_proj.weight.data.copy_(m.gate_proj.weight.data[mid_index, :])
            up_proj.weight.data.copy_(m.up_proj.weight.data[mid_index, :])
            down_proj.weight.data.copy_(m.down_proj.weight.data[:, mid_index])

            m.gate_proj = gate_proj
            m.up_proj = up_proj
            m.down_proj = down_proj

            expert_module = single_experts_module(mid_dim, m.config.hidden_size, experts=dynamic_experts)
            expert_weight = hn[1].module_list[moe_index].linear_router.weight.data
            eval_experts = hn[1].module_list[moe_index].experts_for_eval

            expert_module.linear_router.weight.data.copy_(expert_weight)
            expert_module.experts_for_eval.copy_(eval_experts[:, mid_index].to(torch.uint8))

            m.experts_module = expert_module
            cfgs.append(mid_dim)
            mlp_index += 1
            moe_index += 1

    model.cfgs = cfgs
    return model


def main(
    hf_model: str = "meta-llama/Llama-2-7b-hf",
    hn_path: str = "hn_path",
    output_dir: str = "output_path",
    dynamic_experts: int = 8,
    attn_prune: bool = False,
    test_model: bool = False,
) -> None:
    if hf_model not in (
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Meta-Llama-3-8B",
    ):
        raise ValueError(f"Unsupported hf_model for this repo: {hf_model}")

    device_id = "cuda:0"
    model = LlamaForCausalLM.from_pretrained(hf_model, torch_dtype=torch.bfloat16, device_map=device_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    param_reg = collect_info_reg_llama(model, p=0.5, lam=1.0)

    rnn = hypernetwork(t_structures=param_reg.structures, experts=dynamic_experts)
    experts_list = experts_module_list(
        structures=param_reg.structures,
        model_dim=param_reg.model_dim,
        head_dim=param_reg.head_dim,
        experts=dynamic_experts,
        num_kv_heads=param_reg.num_kv_heads,
    )
    hn_helper = help_functions_hn(param_reg.structures)

    hn = torch.nn.ModuleList([rnn, experts_list])
    print(hn)

    hn_state_dict = torch.load(hn_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in hn_state_dict.items():
        name = k
        if name.startswith("module."):
            name = name.replace("module.", "", 1)
        if name.startswith("model_list."):
            name = name.replace("model_list.", "", 1)
        new_state_dict[name] = v

    hn.load_state_dict(new_state_dict, strict=False)
    hn.eval()
    hn.cuda()
    hn = hn.bfloat16()

    if test_model:
        print(model)
        hn.train()
        hn_helper.set_expert_modules(model, experts_list.module_list)
        vectors = hn[0]()
        hn_helper.set_gate_vectors(unwrap_model(model), vectors)
        evaluate(model, tokenizer, datasets="wikitext")
        return

    vectors = hn[0]()
    width_list, width_union_list = hn_helper.prepare_for_eval(
        hn[1].module_list, vectors, non_uniform=True, return_vector_union=True
    )
    param_reg.count_current_params(width_list)
    width_list_tensor = [torch.tensor(item) for item in width_list]
    print(width_list_tensor)
    print(param_reg(width_list_tensor))

    model.cuda()

    truncated_union_list = [i for i in width_union_list if not isinstance(i, int) and i.sum().item() != 0]

    model = convert_to_moe_llama(model, truncated_union_list, hn, dynamic_experts, attn_prune=attn_prune)
    width_union_cfgs = model.cfgs + [int(dynamic_experts)]
    print(width_union_cfgs)
    state_dict = model.state_dict()
    config = AutoConfig.from_pretrained(hf_model)

    torch.cuda.empty_cache()

    LlamaMoEForCausalLM.cfgs = width_union_cfgs
    moe_model = LlamaMoEForCausalLM(config).to("cuda", dtype=torch.bfloat16)
    moe_model.load_state_dict(state_dict)
    del state_dict
    torch.cuda.empty_cache()

    del model

    torch.cuda.empty_cache()
    _ = sum(p.numel() for p in moe_model.parameters())

    evaluate(moe_model, tokenizer, datasets="wikitext", hn_helper=hn_helper)

    dynamic_width_list = hn_helper.get_dynamic_width_list()
    print(dynamic_width_list)
    moe_modules = list(moe_model.modules())
    index = 0
    for layer_id in range(len(moe_modules)):
        m = moe_modules[layer_id]
        if type(m).__name__ == 'single_experts_module' and hasattr(m, 'top_k'):
            m.top_k.copy_(int(dynamic_width_list[index][0]))

            width_list[index][0] = int(dynamic_width_list[index][0])
            #dynamic_width_list[index][1] = 2*int(dynamic_width_list[index][1])
            index += 1
    param_reg.count_current_params(width_list)

    evaluate(moe_model, tokenizer, datasets="wikitext")

    moe_model.register_for_auto_class("AutoModelForCausalLM")

    print(output_dir)
    print(width_union_cfgs)
    print(width_list)

    moe_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
