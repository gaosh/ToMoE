import datetime
import os
import time
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import IterableDataset
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from data import dataloader_creator, load_hf_dataset_alpaca, load_hf_dataset_mixed, load_hf_dataset_wiki
from utils import DistributedEnv, log_softmax_fp32, softmax_fp32, unwrap_model

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, StateDictType
from torch.distributed.fsdp.wrap import always_wrap_policy, transformer_auto_wrap_policy


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


def iter_gated_attention_modules(model):
    for module in model.modules():
        if type(module).__name__ == "virtual_gate_module" and module.gate_module is not None:
            yield module.gate_module


def gated_attention_reg_loss(model, reg_type="l1", target=1.0):
    gates = []
    for gate_module in iter_gated_attention_modules(model):
        gate = getattr(gate_module, "last_gate", None)
        if gate is not None:
            gates.append(gate.float())

    if not gates:
        first_param = next(model.parameters())
        return torch.zeros((), device=first_param.device, dtype=first_param.dtype)

    gate_values = torch.cat([gate.reshape(-1) for gate in gates])
    target_tensor = torch.as_tensor(target, device=gate_values.device, dtype=gate_values.dtype)
    if reg_type == "l1":
        return gate_values.mean()
    if reg_type == "target_mean":
        return (gate_values.mean() - target_tensor).pow(2)
    if reg_type == "binary":
        return (gate_values * (1.0 - gate_values)).mean()
    if reg_type == "none":
        return torch.zeros((), device=gate_values.device, dtype=gate_values.dtype)
    raise ValueError(f"Unknown gate_reg_type: {reg_type}")


def gated_attention_parameters(model):
    params = []
    for gate_module in iter_gated_attention_modules(model):
        params.extend(list(gate_module.parameters()))
    return params


def gated_attention_state_dict(model):
    state_dict = {}
    for name, module in unwrap_model(model).named_modules():
        if type(module).__name__ == "virtual_gate_module" and module.gate_module is not None:
            for key, value in module.gate_module.state_dict().items():
                state_dict[f"{name}.gate_module.{key}"] = value.detach().cpu()
    return state_dict


def save_hn_and_gates(hn, model, out_dir, filename, env):
    if env.global_rank != 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    if hasattr(hn, "module"):
        hn_state_dict = hn.module.state_dict()
    elif hasattr(hn, "_orig_mod"):
        save_policy = FullStateDictConfig(
            offload_to_cpu=(env.world_size > 1),
            rank0_only=True,
        )
        with FSDP.state_dict_type(hn, StateDictType.FULL_STATE_DICT, save_policy):
            hn_state_dict = hn._orig_mod.state_dict()
    else:
        hn_state_dict = hn.state_dict()

    ckpt = {
        "hn": hn_state_dict,
        "gated_attention": gated_attention_state_dict(model),
    }
    torch.save(ckpt, os.path.join(out_dir, filename))


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
    dynamic_alpha: float = 1.0,
    load_balance_alpha: float = 1.0,
    dynamic_experts: int = 8,
    kd_loss: bool = False,
    compile_flag: bool = True,
    p: float = 0.48,
    lam: float = 16.0,
    hn_block_size: int = 2048,
    hn_lr: float = 1e-3,
    dataset_seed: int = 42,
    gate_rank: int = 128,
    gate_init_bias: float = 3.0,
    gate_reg_weight: float = 0.0,
    gate_reg_type: str = "l1",
    gate_target: float = 1.0,
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
        dir_name = exp_name + "_" + hf_model
        out_dir = os.path.join("./", dir_name)
    if rand_seed is None:
        rand_seed = start_iter
    if learning_rate is None:
        llama_learning_rate_per_sample = 0.0003 / (4 * 1024 * 1024)
        learning_rate = min(llama_learning_rate_per_sample * batch_size * 4096 * env.world_size, 0.0003)
    if env.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    torch._inductor.config.realize_opcount_threshold = 100
    device_id = env.local_rank
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    ignored_token = tokenizer.bos_token_id

    from tomoe.hypernetwork import experts_module_list, hn_module_list, hypernetwork
    from tomoe.pruning_helper import collect_info_reg_llama, help_functions_hn

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
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        ignored_token = tokenizer.bos_token_id
        PruneLlamaDecoderLayer = LlamaDecoderLayer
    else:
        raise ValueError(f"Unsupported hf_model for gated-attention ToMoE training: {hf_model}")

    model.config.use_cache = False
    attach_gated_attention_modules(model, gate_rank=gate_rank, gate_init_bias=gate_init_bias)
    env.print_master(model.config)
    env.print_master(model)

    tic = time.time()
    if "wiki" in dataset_list:
        result_dataset = load_hf_dataset_wiki("train", env.world_size * num_workers, dataset_seed)
    elif "alpaca" in dataset_list:
        result_dataset = load_hf_dataset_alpaca(env.world_size * num_workers, dataset_seed)
    elif "mix" in dataset_list:
        result_dataset = load_hf_dataset_mixed(env.world_size * num_workers, dataset_seed, root_path=dataset_path)
    else:
        raise ValueError(f"Unsupported dataset_list: {dataset_list}")

    dataloader_hn = dataloader_creator(
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

    param_reg = collect_info_reg_llama(model, p=p, lam=lam)
    rnn = hypernetwork(t_structures=param_reg.structures, experts=dynamic_experts)
    experts_list = experts_module_list(
        structures=param_reg.structures,
        model_dim=param_reg.model_dim,
        experts=dynamic_experts,
        alpha=dynamic_alpha,
        head_dim=param_reg.head_dim,
        num_kv_heads=param_reg.num_kv_heads,
    )
    hn_helper = help_functions_hn(
        param_reg.structures,
        load_balance_alpha=load_balance_alpha,
        num_experts=dynamic_experts,
    )
    env.print_master(param_reg.structures)

    rnn.to(device_id)
    experts_list.to(device_id)
    hn = hn_module_list(rnn, experts_list)
    hn.to(device_id)

    if env.world_size > 1:
        hn = DDP(hn, find_unused_parameters=False)

    model.eval()
    model.to(device_id)

    if use_bf16:
        model = model.to(data_type).to(device_id)
        if use_fsdp:
            auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={PruneLlamaDecoderLayer})
            model = FSDP(model, auto_wrap_policy=auto_wrap_policy, use_orig_params=True)
            if env.world_size > 1:
                hn_helper.set_expert_modules(unwrap_model(model), experts_list.module_list)
                experts_list.module_list.float()
    else:
        model = model.to(device_id)
        if use_fsdp:
            auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={PruneLlamaDecoderLayer})
            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                use_orig_params=True,
                mixed_precision=MixedPrecision(param_dtype=data_type, reduce_dtype=data_type, buffer_dtype=data_type),
            )
            if env.world_size > 1:
                hn_helper.set_expert_modules(unwrap_model(model), experts_list.module_list)
                experts_list.module_list.float()

    if env.world_size == 1:
        hn_helper.set_expert_modules(unwrap_model(model), experts_list.module_list)
        experts_list.module_list.float()

    if compile_flag:
        model = torch.compile(model)
    if use_ddp:
        model = DDP(model)

    tic = time.time()
    train_hn_with_gated_attention(
        env,
        model,
        hn=hn,
        train_hn_data=dataloader_hn,
        hn_helper=hn_helper,
        param_reg=param_reg,
        ignored_token=ignored_token,
        max_iter=total_n_step,
        out_dir=out_dir,
        p=p,
        hn_block_size=hn_block_size,
        hn_lr=hn_lr,
        fsdp=use_fsdp,
        save_interval=save_interval,
        data_type=data_type,
        kd_loss=kd_loss,
        gate_reg_weight=gate_reg_weight,
        gate_reg_type=gate_reg_type,
        gate_target=gate_target,
    )
    toc = time.time() - tic
    env.print_master(f"Total training time: {toc:.2f}")


def train_hn_with_gated_attention(
    env: DistributedEnv,
    model: torch.nn.Module,
    hn: torch.nn.Module,
    train_hn_data: IterableDataset,
    hn_helper,
    param_reg,
    start_iter=0,
    ignored_token=-1,
    log_interval=1,
    max_iter=None,
    fsdp=True,
    out_dir=None,
    p=None,
    hn_block_size=2048,
    hn_lr=1e-3,
    save_interval=5000,
    kd_loss=True,
    data_type: torch.dtype = torch.float16,
    gate_reg_weight: float = 0.0,
    gate_reg_type: str = "l1",
    gate_target: float = 1.0,
) -> None:
    device_id = env.local_rank
    iter_num = start_iter
    if fsdp:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        scaler = ShardedGradScaler(enabled=(data_type != torch.float32))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(data_type != torch.float32))

    gate_params = gated_attention_parameters(unwrap_model(model))
    optimizer = torch.optim.AdamW(
        [
            {"params": hn.parameters(), "initial_lr": hn_lr},
            {"params": gate_params, "initial_lr": hn_lr},
        ],
        lr=hn_lr,
        weight_decay=0.05,
        betas=(0.9, 0.999),
    )

    with torch.no_grad():
        pesudo_x = torch.randn(1).to(device_id)
        _ = hn(pesudo_x)

    env.print_master(f"Saving checkpoint to {out_dir}")
    save_hn_and_gates(hn, model, out_dir, f"hn-gated-attn-ckpt-{p:.2f}.pt", env)

    if kd_loss:
        kd_loss_fn = ForwardKLLoss(ignore_index=ignored_token)

    torch.cuda.empty_cache()

    for params in model.parameters():
        params.requires_grad = False
    for params in gate_params:
        params.requires_grad = True
    for params in hn.parameters():
        params.requires_grad = True

    hn.train()
    hn = hn.float()
    model.eval()
    env.print_master(hn)

    tic = time.time()
    for batch in train_hn_data:
        if iter_num >= max_iter:
            break
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device_id)[:, :hn_block_size]
            targets = batch["labels"].to(device_id)[:, :hn_block_size]
            attention_mask = (input_ids != ignored_token).long().to(device_id)

        with autocast(device_type="cuda", dtype=data_type, enabled=(data_type != torch.float32)):
            if kd_loss:
                with torch.no_grad():
                    hn_helper.set_gate_status(unwrap_model(model), False)
                    teacher_output = model(input_ids, attention_mask=attention_mask)
                    teacher_logits = teacher_output.logits if hasattr(teacher_output, "logits") else teacher_output
                    hn_helper.set_gate_status(unwrap_model(model), True)

            pesudo_x = torch.randn(1, device=device_id, dtype=torch.float32)
            with autocast(device_type="cuda", enabled=False):
                vectors, pair_loss, hard_out = hn(pesudo_x)

            hn_helper.set_gate_vectors(unwrap_model(model), vectors)
            model_output = model(input_ids, attention_mask=attention_mask)

            hard_out = hn_helper.get_attn_hard_out(unwrap_model(model), hard_out)
            load_balance_loss = hn_helper.load_balance_loss(unwrap_model(model))

            logits = model_output.logits if hasattr(model_output, "logits") else model_output
            with autocast(device_type="cuda", enabled=False):
                if kd_loss:
                    base_loss = 2 * kd_loss_fn(
                        logits.view(-1, logits.size(-1)),
                        teacher_logits.view(-1, teacher_logits.size(-1)),
                        targets.view(-1),
                    )
                else:
                    base_loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=ignored_token,
                    )

                reg_loss = param_reg(hard_out)
                if gate_reg_weight > 0.0 and gate_reg_type != "none":
                    gate_reg_loss = gated_attention_reg_loss(
                        unwrap_model(model),
                        reg_type=gate_reg_type,
                        target=gate_target,
                    )
                else:
                    gate_reg_loss = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)

                loss = base_loss + reg_loss + pair_loss + load_balance_loss + gate_reg_weight * gate_reg_loss

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
                f"iter {iter_num}/{max_iter}: loss {base_loss.item():.4f}, "
                f"reg_loss {reg_loss.item():.4f}, pair_loss {pair_loss.item():.4f}, "
                f"balance_loss {load_balance_loss.item():.4f}, gate_reg_loss {gate_reg_loss.item():.4f}, "
                f"time: {toc*1000:.2f}msS"
            )

        iter_num += 1
        if iter_num % save_interval == 0:
            save_hn_and_gates(hn, model, out_dir, f"hn-gated-attn-ckpt-iter-{iter_num:06d}-{p:.2f}.pt", env)
            torch.cuda.empty_cache()

    save_hn_and_gates(hn, model, out_dir, f"hn-gated-attn-ckpt-final-{p:.2f}.pt", env)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    from jsonargparse import CLI

    CLI(main)
