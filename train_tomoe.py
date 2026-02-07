import os
import time
import datetime
from functools import partial

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import autocast
from torch.cuda.amp import GradScaler 
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from datasets import IterableDataset

from utils import DistributedEnv, softmax_fp32, log_softmax_fp32, unwrap_model
from data import dataloader_creator, load_hf_dataset_wiki, load_hf_dataset_alpaca, load_hf_dataset_mixed

import bitsandbytes as bnb
# from hypernetwork import hypernetwork

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    always_wrap_policy,
    enable_wrap,
    wrap,
)

def kl_div_loss_with_ignore_index(predictions, targets, labels, ignore_index=-100):
    """
    Compute KL divergence loss with an option to ignore specific indices.
    
    Parameters:
    - predictions: Tensor of model outputs (logits) with shape (batch_size, num_classes).
    - targets: Tensor of target distributions (probabilities) with shape (batch_size, num_classes).
    - ignore_index: Index to ignore in the loss calculation, default is -100.
    
    Returns:
    - loss: KL divergence loss with ignored indices.
    """

    mask = (labels != ignore_index).to(predictions.get_device())
    mask_flat = mask.view(-1)

    valid_log_probs = predictions[mask_flat]
    valid_target_probs = targets[mask_flat]

    loss = F.kl_div(
        log_softmax_fp32(valid_log_probs, dim=-1,),
        softmax_fp32(valid_target_probs, dim=-1,).detach(),
        reduction="batchmean",)
    return loss

class ForwardKLLoss(torch.nn.Module):
  def __init__(self, ignore_index: int = -100):
    super().__init__()
    self.ignore_index = ignore_index

  def forward(self, student_logits, teacher_logits, labels) -> torch.Tensor:
    # Implementation from https://github.com/jongwooko/distillm
    # Computes the softmax of the teacher logits
    teacher_prob = softmax_fp32(teacher_logits, dim=-1, dtype=torch.float32).detach()
    # Computes the student log softmax probabilities
    student_logprob = log_softmax_fp32(student_logits, dim=-1, dtype=torch.float32)
    # Computes the forward KL divergence
    prod_probs = teacher_prob * student_logprob
    # Compute the sum
    x = torch.sum(prod_probs, dim=-1).view(-1)
    # We don't want to include the ignore labels in the average
    mask = (labels != self.ignore_index).int()
    # Loss is averaged over non-ignored targets
    return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

def round_to_block_size(current_rank, block_size=32):

    round_rank = max(block_size, (current_rank // block_size) * block_size)

    return round_rank

def main(
    exp_name: str = 'ToMoE',
    dataset_list: list = ['mix'],
    dataset_ratio: list = [1], 
    out_dir: str = None,
    hf_model: str = 'meta-llama/Llama-2-7b-hf',
    dataset_path: str = '/orange/sgao1/sgao1/',
    learning_rate: float = None,
    total_n_step: int = 100000,
    start_iter: int = 0, 
    batch_size: int = 1,
    use_fsdp: bool = True,
    use_ddp:bool = False,
    use_bf16: bool = False,
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
    hn_block_size = 2048,
    hn_lr: float = 1e-3,
    dataset_seed: int = 42,
):

    # Distributed environment setup
    env = DistributedEnv()
    print(env)
    # if env.world_size == 1:
    #     use_fsdp = False
    #     print('[WARNING] FSDP is disabled since there is only 1 GPU')
    dist.init_process_group("nccl", rank=env.global_rank, world_size=env.world_size, timeout=datetime.timedelta(seconds=3600*5))
    data_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # parameter processing
    if out_dir is None:
        dir_name = exp_name + '_' + hf_model
        out_dir = os.path.join('./', dir_name)
    if rand_seed is None:
        rand_seed = start_iter
    if learning_rate is None:
        llama_learning_rate_per_sample = 0.0003 / (4*1024*1024)
        learning_rate = min(llama_learning_rate_per_sample * batch_size * 4096 * env.world_size, 0.0003)
    if env.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    torch._inductor.config.realize_opcount_threshold = 100
    # GPU preparation
    device_id = env.local_rank
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Load model and optionally compress the model


    # prepare tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer = hf_tokenizer
    ignored_token = tokenizer.bos_token_id #EasyLM ignore bos in

    from tomoe.pruning_helper import help_functions_hn, collect_info_reg_phi, collect_info_reg_llama
    from tomoe.hypernetwork import hypernetwork, experts_module_list, single_experts_module, hn_module_list

    # wait for Qwen Implementation 

    # if hf_model == "Qwen/Qwen2.5-7B" or hf_model == "Qwen/Qwen2.5-14B":
    #     from models.modeling_qwen2_dpmoe import Qwen2ForCausalLM, Qwen2DecoderLayer

    #     model = Qwen2ForCausalLM.from_pretrained(hf_model, attn_implementation = "flash_attention_2",torch_dtype=torch.bfloat16, device_map=device_id)

    #     tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    #     ignored_token_value = '<|endoftext|>'
    #     ignored_token = tokenizer(ignored_token_value)['input_ids'][0]
    #     print(ignored_token)
    #     PruneLlamaDecoderLayer = Qwen2DecoderLayer

    if hf_model == "meta-llama/Llama-2-7b-hf" or hf_model == "meta-llama/Llama-2-13b-hf" or hf_model == 'meta-llama/Meta-Llama-3-8B':

        from models.modeling_llama_dpmoe import LlamaForCausalLM, LlamaDecoderLayer
        model = LlamaForCausalLM.from_pretrained(hf_model, attn_implementation = "flash_attention_2",torch_dtype=torch.bfloat16,)
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        ignored_token = tokenizer.bos_token_id
        PruneLlamaDecoderLayer = LlamaDecoderLayer


    model.config.use_cache = False
    config = model.config
    print(model)

    env.print_master(config)
    env.print_master(model)
    
    tic = time.time()

    if 'wiki' in dataset_list:
        result_dataset = load_hf_dataset_wiki('train', env.world_size*num_workers, dataset_seed)
    elif 'alpaca' in dataset_list:
        result_dataset = load_hf_dataset_alpaca(env.world_size*num_workers, dataset_seed)
    elif 'mix' in dataset_list:
        result_dataset = load_hf_dataset_mixed(env.world_size*num_workers, dataset_seed, root_path=dataset_path)

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
    
    param_reg = collect_info_reg_llama(model, p = p,lam = lam)
        

    rnn = hypernetwork(t_structures = param_reg.structures, experts=dynamic_experts)
    experts_list = experts_module_list(structures= param_reg.structures, model_dim = param_reg.model_dim, experts=dynamic_experts, alpha=dynamic_alpha, head_dim=param_reg.head_dim, num_kv_heads=param_reg.num_kv_heads)


    hn_helper = help_functions_hn(param_reg.structures, load_balance_alpha=load_balance_alpha,num_experts=dynamic_experts)

    print(param_reg.structures)

    rnn.to(device_id)
    experts_list.to(device_id)
    hn = hn_module_list(rnn,experts_list)

    
    hn.to(device_id)
    my_auto_wrap_policy = always_wrap_policy

    if env.world_size>1:
        hn = DDP(hn, find_unused_parameters=False)

    model.eval()
    model.to(device_id)
    
    if use_bf16:
        model = model.to(data_type).to(device_id)
        if use_fsdp:
           
            my_auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={PruneLlamaDecoderLayer})
            
            model = FSDP(
                model, 
                auto_wrap_policy=my_auto_wrap_policy,
                use_orig_params=True
                )

            if env.world_size>1:
                hn_helper.set_expert_modules(unwrap_model(model), experts_list.module_list)
    else:
        model = model.to(device_id) #.to(torch.bfloat16)
        if use_fsdp:
            
            my_auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={PruneLlamaDecoderLayer})

            model = FSDP(
                model, 
                auto_wrap_policy=my_auto_wrap_policy,
                use_orig_params=True, 
                #ignored_states=ignored_params,
                mixed_precision=MixedPrecision(param_dtype=data_type, reduce_dtype=data_type, buffer_dtype=data_type), #cast_forward_inputs=True
                )

            if env.world_size>1:
                hn_helper.set_expert_modules(unwrap_model(model), experts_list.module_list)

    if env.world_size ==1:
        hn_helper.set_expert_modules(unwrap_model(model), experts_list.module_list)


            
    if compile_flag:
        model = torch.compile(model)
    if use_ddp:
        model = DDP(model)


    tic = time.time()
    train_hn(
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

        kd_loss=kd_loss,
        )
    toc = time.time() - tic
    env.print_master(f"Total training time: {toc:.2f}")

def train_hn(
    env: DistributedEnv,
    model: torch.nn.Module,
    hn: torch.nn.Module or torch.nn.ModuleList,
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
    kd_loss = True,


) -> None:
    data_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device_id = env.local_rank
    iter_num = start_iter
    if fsdp:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        scaler = ShardedGradScaler()
    else:
        scaler = torch.cuda.amp.GradScaler()

    optimizer = torch.optim.AdamW([{'params':hn.parameters(), 'initial_lr':hn_lr}], lr=hn_lr, weight_decay=0.05,betas=(0.9, 0.999))

    tic = time.time()
    
    with torch.no_grad():
        pesudo_x = torch.randn(1).to(device_id)
        _ = hn(pesudo_x)

    env.print_master(f"Saving checkpoint to {out_dir}")
    hn_path = os.path.join(out_dir, f"hn-ckpt-{p:.2f}.pt")

    # unwrap common wrappers
    if hasattr(hn, "module"):           # DDP
        state_dict_hn = hn.module.state_dict()

    elif hasattr(hn, "_orig_mod"):       # FSDP
        save_policy = FullStateDictConfig(
            offload_to_cpu=(env.world_size > 1),
            rank0_only=True,
        )
        with FSDP.state_dict_type(hn, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict_hn = hn._orig_mod.state_dict()

    else:                               # single GPU / plain module
        state_dict_hn = hn.state_dict()
    if kd_loss:
        kd_loss_fn = ForwardKLLoss(ignore_index=ignored_token)

    torch.save(state_dict_hn, hn_path)
        

    torch.cuda.empty_cache()

    for params in model.parameters():
        params.requires_grad = False
    for params in hn.parameters():
        params.requires_grad = True
    hn.train()

    env.print_master(hn)

    for batch in train_hn_data:


        if iter_num>=max_iter:
            break
        with torch.no_grad():
            input_ids, targets = batch['input_ids'].to(device_id), batch['labels'].to(device_id)
            input_ids = input_ids[:,:hn_block_size]
            targets = targets[:,:hn_block_size]
            attention_mask = (input_ids != ignored_token).long().to(device_id)



        with autocast(device_type='cuda', dtype=data_type):

            if kd_loss:
                with torch.no_grad():
                    hn_helper.set_gate_status(unwrap_model(model), False)
                    teacher_output = model(input_ids, attention_mask=attention_mask)
                    if hasattr(teacher_output, 'logits'):
                        teacher_logits = teacher_output.logits
                    else:
                        teacher_logits = teacher_output

                    hn_helper.set_gate_status(unwrap_model(model), True)    

            pesudo_x = torch.randn(1).to(device_id)

            vectors, pair_loss, hard_out  = hn(pesudo_x)

            hn_helper.set_gate_vectors(unwrap_model(model),vectors)

            model_output = model(input_ids, attention_mask=attention_mask)



            hard_out = hn_helper.get_attn_hard_out(unwrap_model(model), hard_out)
            # pair_attn_loss = hn_helper.pair_attn_loss(unwrap_model(model))
            # pair_loss += pair_attn_loss
            load_balance_loss = hn_helper.load_balance_loss(unwrap_model(model))

            if hasattr(model_output, 'logits'):
                logits = model_output.logits
                #loss = model_output.loss
            else:
                logits = model_output

            if kd_loss:
                #loss = 16 * kl_div_loss_with_ignore_index(logits.view(-1, logits.size(-1)), teacher_logits.view(-1, teacher_logits.size(-1)), targets.view(-1), ignore_index=ignored_token)
                loss = 2 * kd_loss_fn(logits.view(-1, logits.size(-1)), teacher_logits.view(-1, teacher_logits.size(-1)), targets.view(-1))
            else:
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignored_token)

            reg_loss = param_reg(hard_out)

            loss = loss + reg_loss + pair_loss + load_balance_loss

        
        if torch.isnan(loss):
            # The data may be noisy. Ignore it when loss is nan.
            env.print_master(f"!!! nan loss detected !!!")
            loss.fill_(0)

        # if bf_16:
        toc = time.time() - tic

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()


        toc = time.time() - tic
        tic = time.time()
        if iter_num % log_interval == 0:
    
            env.print_master(f"iter {iter_num}/{max_iter}: loss {(loss-reg_loss-pair_loss-load_balance_loss).item():.4f}, reg_loss {reg_loss.item():.4f}, pair_loss {pair_loss.item():.4f}, balance_loss {load_balance_loss.item():.4f}, time: {toc*1000:.2f}msS")

        iter_num += 1
        if iter_num % save_interval == 0:
            if env.global_rank == 0:
                hn_path = os.path.join(out_dir, f"hn-ckpt-iter-{iter_num:06d}-{p:.2f}.pt")
                env.print_master(f"Saving checkpoint to {out_dir}")

                if hasattr(hn, "module"):  # DDP
                    state_dict_hn = hn.module.state_dict()

                elif hasattr(hn, "_orig_mod"):  # FSDP
                    save_policy = FullStateDictConfig(
                        offload_to_cpu=(env.world_size > 1),
                        rank0_only=True,
                    )
                    with FSDP.state_dict_type(hn, StateDictType.FULL_STATE_DICT, save_policy):
                        state_dict_hn = hn._orig_mod.state_dict()

                else:  # plain / single GPU
                    state_dict_hn = hn.state_dict()

                torch.save(state_dict_hn, hn_path)

            torch.cuda.empty_cache()


    if env.global_rank == 0:
        hn_path = os.path.join(out_dir, f"hn-ckpt-final-{p:.2f}.pt")
        env.print_master(f"Saving checkpoint to {out_dir}")

        if hasattr(hn, "module"):  # DDP
            state_dict_hn = hn.module.state_dict()

        elif hasattr(hn, "_orig_mod"):  # FSDP
            save_policy = FullStateDictConfig(
                offload_to_cpu=(env.world_size > 1),
                rank0_only=True,
            )
            with FSDP.state_dict_type(hn, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict_hn = hn._orig_mod.state_dict()

        else:  # plain / single GPU
            state_dict_hn = hn.state_dict()

        torch.save(state_dict_hn, hn_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    from jsonargparse import CLI
    CLI(main)