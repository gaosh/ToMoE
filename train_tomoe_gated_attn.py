import datetime
import os
import time
from collections import OrderedDict
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import IterableDataset
from transformers import AutoTokenizer

from data import dataloader_creator, load_hf_dataset_alpaca, load_hf_dataset_mixed, load_hf_dataset_wiki
from utils import DistributedEnv, log_softmax_fp32, softmax_fp32, unwrap_model

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def kl_div_loss_with_ignore_index(predictions, targets, labels, ignore_index=-100):
    device = predictions.device
    mask = (labels != ignore_index).to(device).view(-1)
    if mask.sum() == 0:
        return torch.zeros((), device=device, dtype=predictions.dtype)

    student_logprob = log_softmax_fp32(predictions, dim=-1)
    teacher_logprob = log_softmax_fp32(targets, dim=-1).detach()
    kl_per_token = F.kl_div(
        student_logprob,
        teacher_logprob,
        log_target=True,
        reduction="none",
    ).sum(-1)
    kl_per_token = kl_per_token.view(-1)
    return kl_per_token[mask].mean()


class ForwardKLLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, student_logits, teacher_logits, labels) -> torch.Tensor:
        teacher_prob = softmax_fp32(teacher_logits, dim=-1).detach()
        student_logprob = log_softmax_fp32(student_logits, dim=-1)
        prod_probs = teacher_prob * student_logprob
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (labels != self.ignore_index).int()
        return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)


def attach_gated_attention_modules(model, gate_rank=128, gate_init_bias=3.0):
    from tomoe.hypernetwork import SingleGatedAttnModule

    attached = 0
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
            attached += 1
    if attached == 0:
        raise RuntimeError("No LLaMA attention modules were found for gated-attention training.")
    return attached


def _parse_moe_layers(moe_layers, num_layers):
    if moe_layers is None or moe_layers == "all":
        return set(range(num_layers))
    if isinstance(moe_layers, str):
        return {int(item) for item in moe_layers.split(",") if item.strip()}
    return {int(item) for item in moe_layers}


def construct_tomoe_mlp_layers(model, num_experts=8, moe_layers="all", expert_init="balanced", top_k=1):
    if top_k != 1:
        raise ValueError("The existing ToMoE MLP router supports top_k=1 only.")
    if expert_init != "balanced":
        raise ValueError("Only expert_init='balanced' is implemented by the local ToMoE construction path.")

    from models.modeling_llama_moe_final import single_experts_module

    layers = unwrap_model(model).model.layers
    target_layers = _parse_moe_layers(moe_layers, len(layers))
    converted = []
    for layer_idx, layer in enumerate(layers):
        if layer_idx not in target_layers:
            continue

        mlp = layer.mlp
        expert_module = single_experts_module(
            mlp.gate_proj.out_features,
            mlp.config.hidden_size,
            experts=num_experts,
        )
        expert_module.experts_for_eval.zero_()
        expert_indices = torch.arange(mlp.gate_proj.out_features)
        chunks = torch.chunk(expert_indices, num_experts)
        for expert_idx, chunk in enumerate(chunks):
            if chunk.numel() == 0:
                chunk = expert_indices[-1:].clone()
            expert_module.experts_for_eval[expert_idx, chunk] = 1

        mlp.experts_module = expert_module.to(mlp.gate_proj.weight.device)
        mlp.actual_moe = True
        mlp.intermediate_size = mlp.gate_proj.out_features
        converted.append(layer_idx)

    if not converted:
        raise RuntimeError("No MLP layers were converted to ToMoE.")
    unwrap_model(model).tomoe_moe_config = {
        "num_experts": num_experts,
        "top_k": top_k,
        "expert_init": expert_init,
        "moe_layers": sorted(converted),
    }
    return converted


def construct_tomoe_from_hypernetwork(model, hn_path, num_experts=8):
    from prune_tomoe import convert_to_moe_llama
    from tomoe.hypernetwork import experts_module_list, hypernetwork
    from tomoe.pruning_helper import help_functions_hn

    layers = unwrap_model(model).model.layers
    structures = []
    for layer in layers:
        structures.append(layer.self_attn.head_dim)
        structures.append(layer.mlp.gate_proj.out_features)
    first_attn = layers[0].self_attn
    model_dim = first_attn.hidden_size
    head_dim = first_attn.head_dim
    num_kv_heads = first_attn.num_key_value_heads

    rnn = hypernetwork(t_structures=structures, experts=num_experts)
    experts_list = experts_module_list(
        structures=structures,
        model_dim=model_dim,
        head_dim=head_dim,
        experts=num_experts,
        num_kv_heads=num_kv_heads,
    )
    hn = torch.nn.ModuleList([rnn, experts_list])

    hn_state_dict = torch.load(hn_path, map_location="cpu")
    cleaned_state_dict = OrderedDict()
    for key, value in hn_state_dict.items():
        name = key
        if name.startswith("module."):
            name = name.replace("module.", "", 1)
        if name.startswith("model_list."):
            name = name.replace("model_list.", "", 1)
        cleaned_state_dict[name] = value
    hn.load_state_dict(cleaned_state_dict, strict=False)
    hn.eval()

    hn_helper = help_functions_hn(structures)
    with torch.no_grad():
        vectors = hn[0]()
        _, width_union_list = hn_helper.prepare_for_eval(
            hn[1].module_list,
            vectors,
            non_uniform=True,
            return_vector_union=True,
        )
    truncated_union_list = [
        item for item in width_union_list
        if not isinstance(item, int) and item.sum().item() != 0
    ]
    model = convert_to_moe_llama(
        model,
        truncated_union_list,
        hn,
        num_experts,
        attn_prune=False,
    )
    converted = [
        idx for idx, layer in enumerate(unwrap_model(model).model.layers)
        if getattr(layer.mlp, "experts_module", None) is not None
    ]
    unwrap_model(model).tomoe_moe_config = {
        "num_experts": num_experts,
        "top_k": 1,
        "expert_init": "hypernetwork",
        "hn_path": hn_path,
        "moe_layers": converted,
    }
    return model, converted


def set_gated_attention_status(model, enabled=True):
    for module in model.modules():
        if hasattr(module, "use_gated_attn"):
            module.use_gated_attn = enabled


def freeze_for_joint_training(model, freeze_base_model=True, train_gate_only=False):
    if freeze_base_model or train_gate_only:
        for param in model.parameters():
            param.requires_grad = False

    gate_param_count = 0
    for module in model.modules():
        if type(module).__name__ == "virtual_gate_module" and module.gate_module is not None:
            for param in module.gate_module.parameters():
                param.requires_grad = True
                gate_param_count += param.numel()

    if gate_param_count == 0:
        raise RuntimeError("No gated-attention parameters were enabled for training.")

    moe_param_count = 0
    if not train_gate_only:
        for module in model.modules():
            if type(module).__name__ == "single_experts_module":
                for param in module.parameters():
                    param.requires_grad = True
                    moe_param_count += param.numel()

    return gate_param_count, moe_param_count


def iter_gated_attention_modules(model):
    for module in model.modules():
        if type(module).__name__ == "virtual_gate_module" and module.gate_module is not None:
            yield module.gate_module


def gated_attention_reg_loss(model, reg_type="l1", target=1.0):
    gates = []
    device = None
    dtype = None
    for gate_module in iter_gated_attention_modules(model):
        gate = getattr(gate_module, "last_gate", None)
        if gate is None:
            continue
        gates.append(gate.float())
        device = gate.device
        dtype = gate.dtype

    if not gates:
        first_param = next(model.parameters())
        return torch.zeros((), device=first_param.device, dtype=first_param.dtype)

    gate_values = torch.cat([gate.reshape(-1) for gate in gates])
    target_tensor = torch.as_tensor(target, device=gate_values.device, dtype=gate_values.dtype)
    if reg_type == "l1":
        loss = gate_values.mean()
    elif reg_type == "target_mean":
        loss = (gate_values.mean() - target_tensor).pow(2)
    elif reg_type == "binary":
        loss = (gate_values * (1.0 - gate_values)).mean()
    elif reg_type == "none":
        loss = torch.zeros((), device=gate_values.device, dtype=gate_values.dtype)
    else:
        raise ValueError(f"Unknown gate_reg_type: {reg_type}")

    return loss.to(device=device, dtype=dtype)


def save_gated_attention_checkpoint(model, out_dir, filename, env):
    if env.global_rank != 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, filename)
    torch.save(unwrap_model(model).state_dict(), ckpt_path)
    env.print_master(f"Saving ToMoE gated-attention checkpoint to {ckpt_path}")


def load_gated_attn_model(hf_model, data_type):
    if hf_model in (
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Meta-Llama-3-8B",
    ):
        from models.modeling_llama_tomoe_gated_attn import LlamaDecoderLayer, LlamaForCausalLM

        attn_impl = "flash_attention_2" if data_type in (torch.float16, torch.bfloat16) else "sdpa"
        model = LlamaForCausalLM.from_pretrained(
            hf_model,
            attn_implementation=attn_impl,
            torch_dtype=data_type,
        )
        return model, LlamaDecoderLayer

    raise ValueError(f"Unsupported gated-attention model: {hf_model}")


def main(
    exp_name: str = "ToMoE-GatedAttn",
    dataset_list: list = ["mix"],
    dataset_ratio: list = [1],
    out_dir: str = None,
    hf_model: str = "meta-llama/Llama-2-7b-hf",
    dataset_path: str = "/orange/sgao1/sgao1/",
    learning_rate: float = None,
    total_n_step: int = 100000,
    start_iter: int = 0,
    batch_size: int = 1,
    use_fsdp: bool = True,
    use_ddp: bool = False,
    use_bf16: bool = False,
    use_fp32: bool = False,
    save_interval: int = 5000,
    num_workers: int = 2,
    rand_seed: int = None,
    kd_loss: bool = False,
    compile_flag: bool = True,
    hn_block_size: int = 2048,
    dataset_seed: int = 42,
    gate_lr: float = 1e-3,
    gate_rank: int = 128,
    gate_init_bias: float = 3.0,
    gate_reg_weight: float = 0.0,
    gate_reg_type: str = "l1",
    gate_target: float = 1.0,
    moe_num_experts: int = 8,
    moe_top_k: int = 1,
    moe_layers: str = "all",
    moe_expert_init: str = "balanced",
    moe_aux_loss_weight: float = 1.0,
    hn_path: str = None,
    freeze_base_model: bool = True,
    train_gate_only: bool = False,
):
    env = DistributedEnv()
    print(env)
    dist.init_process_group(
        "nccl",
        rank=env.global_rank,
        world_size=env.world_size,
        timeout=datetime.timedelta(seconds=3600 * 5),
    )
    if use_fp32:
        data_type = torch.float32
    elif use_bf16:
        data_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        data_type = torch.float16

    if out_dir is None:
        dir_name = exp_name + "_" + hf_model.replace("/", "_")
        out_dir = os.path.join("./", dir_name)
    if rand_seed is None:
        rand_seed = start_iter
    torch.manual_seed(rand_seed)
    if learning_rate is None:
        learning_rate = gate_lr
    if env.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    torch._inductor.config.realize_opcount_threshold = 100
    device_id = env.local_rank
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer = hf_tokenizer
    ignored_token = tokenizer.bos_token_id

    model, GatedAttnLlamaDecoderLayer = load_gated_attn_model(hf_model, data_type)
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    ignored_token = tokenizer.bos_token_id
    model.config.use_cache = False

    if hn_path is not None:
        model, converted_layers = construct_tomoe_from_hypernetwork(
            model,
            hn_path=hn_path,
            num_experts=moe_num_experts,
        )
    else:
        converted_layers = construct_tomoe_mlp_layers(
            model,
            num_experts=moe_num_experts,
            moe_layers=moe_layers,
            expert_init=moe_expert_init,
            top_k=moe_top_k,
        )

    attached_layers = attach_gated_attention_modules(
        model,
        gate_rank=gate_rank,
        gate_init_bias=gate_init_bias,
    )
    gate_param_count, moe_param_count = freeze_for_joint_training(
        model,
        freeze_base_model=freeze_base_model,
        train_gate_only=train_gate_only,
    )
    env.print_master(f"Converted MLP layers to ToMoE: {converted_layers}")
    env.print_master(f"ToMoE experts per converted MLP: {moe_num_experts}")
    env.print_master(f"Attached gated-attention modules: {attached_layers}")
    env.print_master(f"Trainable gated-attention parameters: {gate_param_count}")
    env.print_master(f"Trainable ToMoE router parameters: {moe_param_count}")
    env.print_master(model.config)

    tic = time.time()
    if "wiki" in dataset_list:
        result_dataset = load_hf_dataset_wiki("train", env.world_size * num_workers, dataset_seed)
    elif "alpaca" in dataset_list:
        result_dataset = load_hf_dataset_alpaca(env.world_size * num_workers, dataset_seed)
    elif "mix" in dataset_list:
        result_dataset = load_hf_dataset_mixed(env.world_size * num_workers, dataset_seed, root_path=dataset_path)
    else:
        raise ValueError(f"Unsupported dataset_list: {dataset_list}")

    dataloader = dataloader_creator(
        dataset=result_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        block_size=hn_block_size,
        num_workers=num_workers,
        cycling=True,
        rank=env.global_rank,
        world_size=env.world_size,
        ignored_token=ignored_token,
    )
    toc = time.time() - tic
    env.print(f"Initialilzing training dataset - done. Time elapse (s): {toc:.2f}")

    model.train()
    model.to(device_id)

    if use_bf16:
        model = model.to(data_type).to(device_id)
        if use_fsdp:
            auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={GatedAttnLlamaDecoderLayer})
            model = FSDP(model, auto_wrap_policy=auto_wrap_policy, use_orig_params=True)
    else:
        model = model.to(device_id)
        if use_fsdp:
            auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={GatedAttnLlamaDecoderLayer})
            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                use_orig_params=True,
                mixed_precision=MixedPrecision(param_dtype=data_type, reduce_dtype=data_type, buffer_dtype=data_type),
            )

    if compile_flag:
        model = torch.compile(model)
    if use_ddp:
        model = DDP(model)

    train_gated_attn(
        env,
        model,
        train_data=dataloader,
        ignored_token=ignored_token,
        start_iter=start_iter,
        max_iter=total_n_step,
        fsdp=use_fsdp,
        out_dir=out_dir,
        hn_block_size=hn_block_size,
        gate_lr=learning_rate,
        save_interval=save_interval,
        kd_loss=kd_loss,
        data_type=data_type,
        gate_reg_weight=gate_reg_weight,
        gate_reg_type=gate_reg_type,
        gate_target=gate_target,
        moe_aux_loss_weight=moe_aux_loss_weight,
        moe_top_k=moe_top_k,
    )


def train_gated_attn(
    env: DistributedEnv,
    model: torch.nn.Module,
    train_data: IterableDataset,
    start_iter=0,
    ignored_token=-1,
    log_interval=1,
    max_iter=None,
    fsdp=True,
    out_dir=None,
    hn_block_size=2048,
    gate_lr=1e-3,
    save_interval=5000,
    kd_loss=True,
    data_type: torch.dtype = torch.float16,
    gate_reg_weight: float = 0.0,
    gate_reg_type: str = "l1",
    gate_target: float = 1.0,
    moe_aux_loss_weight: float = 1.0,
    moe_top_k: int = 1,
) -> None:
    device_id = env.local_rank
    iter_num = start_iter
    if fsdp:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        scaler = ShardedGradScaler(enabled=(data_type != torch.float32))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(data_type != torch.float32))

    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad], "initial_lr": gate_lr}],
        lr=gate_lr,
        weight_decay=0.05,
        betas=(0.9, 0.999),
    )
    if kd_loss:
        kd_loss_fn = ForwardKLLoss(ignore_index=ignored_token)

    save_gated_attention_checkpoint(model, out_dir, "gated-attn-ckpt-initial.pt", env)
    torch.cuda.empty_cache()
    model.train()
    tic = time.time()

    for batch in train_data:
        if iter_num >= max_iter:
            break
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device_id)[:, :hn_block_size]
            targets = batch["labels"].to(device_id)[:, :hn_block_size]
            attention_mask = (input_ids != ignored_token).long().to(device_id)

        with autocast(device_type="cuda", dtype=data_type, enabled=(data_type != torch.float32)):
            if kd_loss:
                with torch.no_grad():
                    set_gated_attention_status(unwrap_model(model), False)
                    teacher_output = model(input_ids, attention_mask=attention_mask)
                    teacher_logits = teacher_output.logits if hasattr(teacher_output, "logits") else teacher_output
                    set_gated_attention_status(unwrap_model(model), True)

            model_output = model(
                input_ids,
                attention_mask=attention_mask,
                labels=targets,
                output_router_logits=(moe_aux_loss_weight > 0.0),
            )
            logits = model_output.logits if hasattr(model_output, "logits") else model_output

            with autocast(device_type="cuda", enabled=False):
                if kd_loss:
                    main_loss = 2 * kd_loss_fn(
                        logits.view(-1, logits.size(-1)),
                        teacher_logits.view(-1, teacher_logits.size(-1)),
                        targets.view(-1),
                    )
                elif hasattr(model_output, "loss") and model_output.loss is not None:
                    main_loss = model_output.loss
                else:
                    main_loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=ignored_token,
                    )

                if moe_aux_loss_weight > 0.0 and getattr(model_output, "router_logits", None):
                    from models.modeling_llama_tomoe_gated_attn import combined_moe_balance_loss

                    moe_aux_loss = combined_moe_balance_loss(model_output.router_logits, top_k=moe_top_k)
                else:
                    moe_aux_loss = torch.zeros((), device=main_loss.device, dtype=main_loss.dtype)

                if gate_reg_weight > 0.0 and gate_reg_type != "none":
                    gate_reg = gated_attention_reg_loss(
                        unwrap_model(model),
                        reg_type=gate_reg_type,
                        target=gate_target,
                    )
                else:
                    gate_reg = torch.zeros((), device=main_loss.device, dtype=main_loss.dtype)

                loss = main_loss + moe_aux_loss_weight * moe_aux_loss + gate_reg_weight * gate_reg

        if torch.isnan(loss):
            env.print_master("!!! nan loss detected !!!")
            loss.fill_(0)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        toc = time.time() - tic
        tic = time.time()
        if iter_num % log_interval == 0:
            env.print_master(
                f"iter {iter_num}/{max_iter}: loss {main_loss.item():.4f}, "
                f"moe_aux {moe_aux_loss.item():.4f}, gate_reg {gate_reg.item():.4f}, "
                f"total {loss.item():.4f}, time: {toc*1000:.2f}ms"
            )

        iter_num += 1
        if iter_num % save_interval == 0:
            save_gated_attention_checkpoint(model, out_dir, f"gated-attn-ckpt-iter-{iter_num:06d}.pt", env)
            torch.cuda.empty_cache()

    save_gated_attention_checkpoint(model, out_dir, "gated-attn-ckpt-final.pt", env)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    from jsonargparse import CLI

    CLI(main)
