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

from utils import DistributedEnv, softmax_fp32, log_softmax_fp32
from data import dataloader_creator, load_hf_dataset_wiki, load_hf_dataset_alpaca


from flashlm.compression.factorize.param_util import unwrap_model
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
    teacher_prob = F.softmax(teacher_logits, dim=-1, dtype=torch.float32).detach()
    # Computes the student log softmax probabilities
    student_logprob = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
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
    exp_name: str = 'FlashLM',
    dataset_list: list = ['refinedweb'],
    dataset_ratio: list = [1], 
    out_dir: str = None,
    hf_model: str = '/group-volume/models/AIC/aic-v06/hf',
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

    adam_8bit:bool = False,
    # hn_groupsize: int = 1,
    dynamic_alpha: float = 1.0,
    load_balance_alpha: float = 1.0,
    dynamic_beta: float = 1.0,
    dynamic_experts: int = 8,
    
    kd_loss: bool = False,
    foward_kl_loss: bool = False,

    compile_flag: bool = True,

    p: float = 0.48,
    lam: float = 16.0,
    
    hn_block_size = 2048,
   
    hn_lr: float = 1e-3,
    min_hn_lr: float = 1e-3,

    T: float = 0.4,
    dataset_seed: int = 42,
):
    """ Llama model pretraining recipe
    Args:
        exp_name: it will be part of the output folder name if out_dir=None
        dataset_list: ex: --dataset_list ['pile', 'korean', 'slimpajama', 'refinedweb']
        dataset_ratio: ex: --dataset_ratio [0.2, 0.3, 0.2, 0.3]
        out_dir: output folder
        hf_model: ex: huggyllama/llama-7b, /group-volume/models/llm/llama/hugginface/Llama-2-7b-hf
        learning_rate: if none, the recipe will compute it automatically based on the batch size, block size, and number of GPU
        total_n_step: total number of training steps
        start_iter: This is approximated resume (no overhead). Ex: start_iter=2000, the code will start from 2000 iter with a different batch sampling seed.
        resume_iter: This is the exact resume (large overhead). Ex: resume_iter=1000, the code will generate the first 1000 batches without training. The training starts at iter=1001.
        use_bf16: bf16 does not need mixed-precision, thus it is friendly for gradient accumulation
        rand_seed: the seed will be used to shuffle the dataloader
        non_hf_tokenizer_path: The HF tokenizer is very slow with Pile dataset due to an unknown issue (but the mixed dataset is ok). Use an external tokenizer when training with Pile only. ex: /data-sets/sdp-text/llm-models/llama/7B/tokenizer.model
        cd /group-volume/users/s.gao1/code/FlashLM
    """

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
        user_name = 's.gao1'
        dateTimeObj = datetime.datetime.now()
        out_dir = os.path.join('/group-volume/models/temp', user_name, exp_name)
    if rand_seed is None:
        rand_seed = start_iter
    if learning_rate is None:
        llama_learning_rate_per_sample = 0.0003 / (4*1024*1024)
        learning_rate = min(llama_learning_rate_per_sample * batch_size * 4096 * env.world_size, 0.0003)
    if env.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

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
        result_dataset = load_hf_dataset_mixed(env.world_size*num_workers, dataset_seed)

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

    hn_helper.set_qk_hyperparameters(model, qk_sample_rate = dynamic_qk_sample_rate, grad_w=dynamic_grad_w, pv_detach_flag=dynamic_pv_detach_flag, block_dropout=block_dropout_rate)

    rnn.to(device_id)
    experts_list.to(device_id)
    hn = hn_module_list(rnn,experts_list)

    
    hn.to(device_id)
    my_auto_wrap_policy = always_wrap_policy
    if attn_experts:
        hn.set_hn_inputs_grad()
        if env.world_size>1:
            hn = FSDP(
                hn, 
                auto_wrap_policy = my_auto_wrap_policy,
                use_orig_params=True, 
                mixed_precision=MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
                )
        # else:
        #     hn = DDP(hn, find_unused_parameters=False)
    else:
        if env.world_size>1:
            #hn = DDP(hn)
            hn = DDP(hn, find_unused_parameters=False)
    model.eval()
    # model.train()
    model.to(device_id)

    if do_train_hn_pruning:
        from flashlm.compression.prune.pruning_helper import collect_info_reg, help_functions_hn, collect_info_reg_phi, collect_info_reg_llama, collect_info_reg_opt, collect_info_reg_phi_constrained
        from flashlm.compression.prune.hypernetwork import hypernetwork, simplifed_gate, mlp_net
        from flashlm.models import PruneLlamaForCausalLM, PruneLlamaDecoderLayer
        
        if hf_model == "microsoft/phi-1_5" or hf_model == "microsoft/phi-2":
            from flashlm.models.modeling_phi_prune import PhiForCausalLM, PhiDecoderLayer
            model = PhiForCausalLM.from_pretrained(hf_model)
            tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
            ignored_token = tokenizer.bos_token_id
            PruneLlamaDecoderLayer = PhiDecoderLayer
        elif hf_model == "facebook/opt-125m" or hf_model == "facebook/opt-1.3b" or hf_model == "facebook/opt-2.7b" or hf_model == "facebook/opt-6.7b":
            from flashlm.models.modeling_opt_prune import OPTForCausalLM, OPTDecoderLayer
            model = OPTForCausalLM.from_pretrained(hf_model)
            tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
            ignored_token = tokenizer.bos_token_id
            PruneLlamaDecoderLayer = OPTDecoderLayer
        elif hf_model == "dfurman/LLaMA-13B" or hf_model == "meta-llama/Llama-2-7b-hf" or hf_model == "meta-llama/Llama-2-13b-hf" or hf_model == '/group-volume/users/chiheng/TransformerCompression/experiments/results/model/llama2-7b' or hf_model == '/group-volume/users/chiheng/TransformerCompression/experiments/results/model/llama2-13b-raw' or hf_model == 'lmsys/vicuna-7b-v1.3':
            # from flashlm.models.modeling_flashllama_prune import LlamaForCausalLM, LlamaDecoderLayer

            from flashlm.models.modeling_flashllama_prune import FlashLlamaForCausalLM, FlashLlamaDecoderLayer

            model = FlashLlamaForCausalLM.from_pretrained(hf_model)
            tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
            ignored_token = tokenizer.bos_token_id
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token 
                ignored_token = tokenizer.eos_token_id
            PruneLlamaDecoderLayer = FlashLlamaDecoderLayer
        if hf_model == "Qwen/Qwen2.5-7B" or hf_model == "Qwen/Qwen2.5-14B":
            from flashlm.models.modeling_qwen2_prune import Qwen2ForCausalLM, Qwen2DecoderLayer
            model = Qwen2ForCausalLM.from_pretrained(hf_model, attn_implementation = "flash_attention_2",torch_dtype=torch.bfloat16,)
            tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True,)
            ignored_token_value = '<|endoftext|>'
            ignored_token = tokenizer(ignored_token_value)['input_ids'][0]
#            ignored_token = tokenizer.bos_token_id
            print(ignored_token)
            PruneLlamaDecoderLayer = Qwen2DecoderLayer

        elif hf_model == "/group-volume/models/llm/llama/hugginface/llama-7b":
            from flashlm.models.modeling_flashllama_prune import FlashLlamaForCausalLM, FlashLlamaDecoderLayer

            model = FlashLlamaForCausalLM.from_pretrained(hf_model)
            tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
            ignored_token = tokenizer.bos_token_id
            PruneLlamaDecoderLayer = FlashLlamaDecoderLayer
        else:
            model = PruneLlamaForCausalLM.from_pretrained(hf_model)
        # .from_pretrained(hf_model)
        model.config.use_cache = False
        config = model.config
        #print(model)

        env.print_master(config)
        env.print_master(model)
        
        tic = time.time()
        # env.print(f"Initialilzing training dataset:{dataset_list} with ratios {dataset_ratio}")
        # if hf_model == '':

        if 'minipile' in dataset_list:
            result_dataset = load_hf_dataset_minipile('train', env.world_size*num_workers)
        elif 'wiki' in dataset_list:
            result_dataset = load_hf_dataset_wiki('train', env.world_size*num_workers, dataset_seed)
        elif 'orca' in dataset_list:
            result_dataset = load_hf_dataset_orca_dpo(env.world_size*num_workers, dataset_seed)
        elif 'alpaca' in dataset_list:
            result_dataset = load_hf_dataset_alpaca(env.world_size*num_workers, dataset_seed)
        elif 'wizardlM' in dataset_list:
            result_dataset = load_hf_dataset_wizardlMv2(env.world_size*num_workers, dataset_seed)
        elif 'mix' in dataset_list:
            result_dataset = load_hf_dataset_mixed(env.world_size*num_workers, dataset_seed)
        elif 'new_mix' in dataset_list:
            result_dataset = load_hf_dataset_new_mixed(env.world_size*num_workers, dataset_seed)
        else:
            result_dataset = load_hf_dataset_pile_dedup('validation', env.world_size*num_workers)
        val_dataloader_hn = dataloader_creator(
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
        if hf_model == "microsoft/phi-1_5" or hf_model == "microsoft/phi-2":
            if constrained == 'same':
                param_reg = collect_info_reg_phi_constrained(model, p = p,lam = lam)
            else:
                param_reg = collect_info_reg_phi(model, p = p,lam = lam)
        elif hf_model == "facebook/opt-125m" or hf_model == "facebook/opt-1.3b" or hf_model == "facebook/opt-2.7b" or hf_model == "facebook/opt-6.7b":
            param_reg = collect_info_reg_opt(model, p = p,lam = lam)
        elif hf_model == "dfurman/LLaMA-13B" or hf_model == "Qwen/Qwen2.5-7B" or hf_model == "Qwen/Qwen2.5-14B" or hf_model == "/group-volume/models/llm/llama/hugginface/llama-7b" or hf_model == '/group-volume/users/chiheng/TransformerCompression/experiments/results/model/llama2-7b' or hf_model == 'lmsys/vicuna-7b-v1.3' or hf_model == '/group-volume/users/chiheng/TransformerCompression/experiments/results/model/llama2-13b-raw':
            param_reg = collect_info_reg_llama(model, p = p,lam = lam)
        else:
            param_reg = collect_info_reg(model, p = p,lam = lam)
        # print(param_reg.structures)
        # print(p)
        if simple_gate:
            hn = simplifed_gate(t_structures = param_reg.structures)
        elif mlp_gate:
            hn = mlp_net(t_structures = param_reg.structures)
        else:
            hn = hypernetwork(t_structures = param_reg.structures, group_size=1)
        print(hn)
        hn_helper = help_functions_hn(param_reg.structures, constrained=constrained)

        hn.to(device_id)
        hn = DDP(hn)
        model.to(device_id)

    if do_train_hn_semi_pruning:
        from flashlm.compression.semi_structure.semi_pruning_helper import collect_info_reg, help_functions_hn
        from flashlm.compression.semi_structure.hypernetwork import hypernetwork, simplifed_gate
        from flashlm.compression.semi_structure.semi_pruning_helper import model_replace
        if hf_model == "microsoft/phi-1_5" or hf_model == "microsoft/phi-2":
            model = AutoModelForCausalLM.from_pretrained(hf_model, torch_dtype="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        elif slice_gpt:
            model = AutoModelForCausalLM.from_pretrained(hf_model, torch_dtype="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        else:
            model = FlashLlamaForCausalLM.from_pretrained(hf_model)
        # .from_pretrained(hf_model)
        model.config.use_cache = False
        config = model.config
        print(model)

        env.print_master(config)
        env.print_master(model)
        
        tic = time.time()
        if use_minipile:
            result_dataset = load_hf_dataset_minipile('train', env.world_size*num_workers)
        else:
            result_dataset = load_hf_dataset_pile_dedup('validation', env.world_size*num_workers)
        val_dataloader_hn = dataloader_creator(
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
        group_info = {}
        group_info['groups_in_dim'] = groups_in_dim
        group_info['groups_out_dim'] = groups_out_dim
        if hf_model == "microsoft/phi-1_5" or hf_model == "microsoft/phi-2" or slice_gpt:
            if slice_gpt:
                model_dim = True
            else:
                model_dim = False
            model_replace(model, group_info=group_info, hf_model='phi-1_5',model_dim=model_dim)
        else:
            model_replace(model, group_info=group_info)
        param_reg = collect_info_reg(model, p = p,lam = lam)
        if simple_gate:
            hn = simplifed_gate(t_structures = param_reg.structures, num_groups=hn_groups, reinmax = use_reinmax)
        else:
            hn = hypernetwork(t_structures = param_reg.structures, num_groups=hn_groups, reinmax = use_reinmax, hard_flag=hard_flag, param_flag=semi_params)
            hn.T = T
        hn_helper = help_functions_hn(param_reg.structures,gamma=gamma)
        #hn_helper.set_mask_status(model, use_mask=False)
        hn_helper.set_gate_status(model, use_gate=True)
        hn_helper.set_scale_weight(model, scale_weight=scale_weight)
        if soft_rank:
            hn_helper.init_rank_reg(model)

        hn.to(device_id)
        print(hn)
        if env.world_size >1:
            hn = DDP(hn)
        print(model)
        # my_auto_wrap_policy = always_wrap_policy
        # hn = FSDP(
        #         hn, 
        #         auto_wrap_policy = my_auto_wrap_policy,
        #         use_orig_params=True, 
        #         mixed_precision=MixedPrecision(param_dtype=data_type, reduce_dtype=data_type, buffer_dtype=data_type)
        #         )
        model.to(device_id)

    if do_train_hn:
        from flashlm.models import ShareLlamaForCausalLM, ShareLlamaDecoderLayer
        from flashlm.compression.weightsharing.weightsharing_helper import collect_info_reg, help_functions_hn

        model = ShareLlamaForCausalLM.from_pretrained(hf_model)
        # .from_pretrained(hf_model)
        model.config.use_cache = False
        config = model.config
        print(model)

        env.print_master(config)
        env.print_master(model)
        
        tic = time.time()
        # env.print(f"Initialilzing training dataset:{dataset_list} with ratios {dataset_ratio}")
        if use_minipile:
            result_dataset = load_hf_dataset_minipile('train', env.world_size*num_workers)
        else:
            result_dataset = load_hf_dataset_pile_dedup('validation', env.world_size*num_workers)
        val_dataloader_hn = dataloader_creator(
            dataset=result_dataset, 
            tokenizer=tokenizer,
            batch_size=batch_size, 
            block_size=hn_block_size,
            num_workers=num_workers,
            cycling=False,
            rank=env.global_rank,
            world_size=env.world_size,
            ignored_token=ignored_token,
            )
        toc = time.time() - tic
        env.print(f"Initialilzing training dataset - done. Time elapse (s): {toc:.2f}")

        param_reg = collect_info_reg(model, p = p,lam = lam)

        hn = hypernetwork(t_structures = param_reg.structures)

        hn_helper = help_functions_hn(param_reg.structures)

        hn_helper.set_weight_sharing_flag(model, share_flag=True)
        hn_helper.set_weight_sharing_flag_att(model, share_flag=True)

        hn.to(device_id)
        if env.world_size>1:
            hn = DDP(hn)
        #model.to(device_id)
    
    if use_bf16:
        model = model.to(data_type).to(device_id)
        if use_fsdp:
            if do_train_hn_pruning or do_train_hn_dynamic_pruning:
                my_auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={PruneLlamaDecoderLayer})
            elif do_train_hn_semi_pruning:
                my_auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={FlashLlamaDecoderLayer})
            else:
                my_auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={ShareLlamaDecoderLayer})
            model = FSDP(
                model, 
                auto_wrap_policy=my_auto_wrap_policy,
                use_orig_params=True
                )
            if do_train_hn_dynamic_pruning:
                if env.world_size>1:
                    hn_helper.set_expert_modules(unwrap_model(model), experts_list.module_list)
    else:
        model = model.to(device_id) #.to(torch.bfloat16)
        if use_fsdp:
            if do_train_hn_pruning or do_train_hn_dynamic_pruning:
                my_auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={PruneLlamaDecoderLayer})
            elif do_train_hn_semi_pruning:
                my_auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={FlashLlamaDecoderLayer})
            else:
                my_auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={ShareLlamaDecoderLayer})
            # if do_train_hn_dynamic_pruning:
            #     modules = list(model.modules())
            #     ignored_params = []
            #     for layer_id in range(len(modules)):
            #         m = modules[layer_id]
            #         if type(m).__name__ == 'virtual_dynamic_operation':
            #             ignored_params.append(m.router.linear_router.weight)
            #             ignored_params.append(m.router.linear_decoder.weight)
            #             ignored_params.append(m.router.ln.weight)
            #             ignored_params.append(m.router.ln.bias)
            #     print(ignored_params[0].size())
            # else:
            ignored_params=None
            

            model = FSDP(
                model, 
                auto_wrap_policy=my_auto_wrap_policy,
                use_orig_params=True, 
                #ignored_states=ignored_params,
                mixed_precision=MixedPrecision(param_dtype=data_type, reduce_dtype=data_type, buffer_dtype=data_type), #cast_forward_inputs=True
                )
            if do_train_hn_dynamic_pruning:
                if env.world_size>1:
                    hn_helper.set_expert_modules(unwrap_model(model), experts_list.module_list)
    
    if env.world_size ==1 and do_train_hn_dynamic_pruning:
        hn_helper.set_expert_modules(unwrap_model(model), experts_list.module_list)


            #print(ignored_params[0].size())
    #print(model)
    if compile_flag:
        model = torch.compile(model)
    if use_ddp:
        model = DDP(model)
    
    # kd_loss = True
    # mix_loss = False
    # # if hf_model !=  "meta-llama/Llama-2-13b-hf":
    # #     adam_8bit = False
    # if hf_model == "microsoft/Phi-3-mini-4k-instruct":
    #     kd_loss = False
    #     mix_loss = True
    # if do_train_hn_pruning:
    #     kd_loss = False
    #     mix_loss = False

    if do_train_hn or do_train_hn_pruning or do_train_hn_semi_pruning or do_train_hn_dynamic_pruning:
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
            bf_16=use_bf16,
            out_dir=out_dir,
            p=p,
            model_size=model_size,
            hn_block_size=hn_block_size,
            hn_lr=hn_lr,
            use_sch=use_sch,
            min_hn_lr=min_hn_lr,
            semi_params=semi_params,
            soft_rank=soft_rank,
            use_fsdp=use_fsdp,
            load_balance=load_balance,
            dynamic_transit=dynamic_transit,
            save_interval=save_interval,
            adam_8bit=adam_8bit,
            kd_loss=kd_loss,
            mix_loss = mix_loss,
            top_k_mlp = top_k_mlp,
            foward_kl_loss=foward_kl_loss,
            gated_attn_flag=dynamic_gated_attn_flag,
            )
        toc = time.time() - tic
        env.print_master(f"Total training time: {toc:.2f}")

def pair_loss_weight(step, warmup_start=0, warmup_iters=500):
    """
    Returns the weight for the pair_loss at the given step.

    warmup_start: step where warmup begins (0 → 100)
    warmup_iters: number of steps to go from 0 → 1 (100 → 600)
    """
    if step < warmup_start:
        return 0.0
    
    progress = (step - warmup_start) / warmup_iters
    return min(max(progress, 0.0), 1.0)

def train_hn(
    env: DistributedEnv,
    model: torch.nn.Module,
    hn: torch.nn.Module or torch.nn.ModuleList,
    train_hn_data: IterableDataset,
    hn_helper,
    param_reg,
    experts_list = None,
    start_iter=0,
    ignored_token=-1,
    log_interval=1,
    max_iter=None,
    bf_16=True,
    fsdp=True,
    out_dir=None,
    p=None,
    model_size:str = '7B',
    hn_block_size=2048,
    hn_lr=1e-3,
    min_hn_lr=1e-3,
    use_sch=False,
    semi_params=False,
    soft_rank=False,
    use_fsdp=False,
    load_balance = False,
    dynamic_transit = 1.0,
    save_interval=5000,
    scheduler_start_iter=9000,
    kd_loss = True,
    foward_kl_loss=False,
    mix_loss = False,
    adam_8bit=True,
    top_k_mlp = False,
    gated_attn_flag=False,
) -> None:
    data_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device_id = env.local_rank
    iter_num = start_iter
    if fsdp:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        scaler = ShardedGradScaler()
    else:
        scaler = torch.cuda.amp.GradScaler()
    #if foward_kl_loss:
    kd_loss_fn = ForwardKLLoss(ignore_index=ignored_token)
    # if hasattr(hn, 'module'):
    #     hn_group = hn.module.groups
    # else:
    #     hn_group = hn.groups
    #optimizer = torch.optim.AdamW([{'params':hn.parameters(), 'initial_lr':hn_lr}], lr=hn_lr, weight_decay=0.05)
    # if hasattr(hn, "module"):
    #     scale_weight_flag = hn.module.scale_list
    # else:
    #     scale_weight_flag = hn.scale_list
    if use_sch:
        if hn_lr == min_hn_lr:
            min_hn_lr = 0.1*hn_lr
    # if scale_weight_flag is not None:
    #     if hasattr(hn, 'module'):
    #         optimizer = torch.optim.AdamW([
    #             {'params':hn.module.bi_GRU.parameters(), 'initial_lr':hn_lr, 'eta_min': min_hn_lr},
    #             {'params':hn.module.linear_list_tp.parameters(), 'initial_lr':hn_lr, 'eta_min': min_hn_lr},
    #             {'params':hn.module.ln_tp.parameters(), 'initial_lr':hn_lr, 'eta_min': min_hn_lr},
    #             {'params':hn.module.scale_list.parameters(), 'initial_lr':0.1*hn_lr, 'lr': 0.1*hn_lr, 'eta_min': 0.1*min_hn_lr},
    #         ], lr=hn_lr, weight_decay=0.05)
    #     else:
    #         optimizer = torch.optim.AdamW([
    #             {'params':hn.bi_GRU.parameters(), 'initial_lr':hn_lr, 'eta_min': min_hn_lr},
    #             {'params':hn.linear_list_tp.parameters(), 'initial_lr':hn_lr, 'eta_min': min_hn_lr},
    #             {'params':hn.ln_tp.parameters(), 'initial_lr':hn_lr, 'eta_min': min_hn_lr},
    #             {'params':hn.scale_list.parameters(), 'initial_lr':0.1*hn_lr, 'lr': 0.1*hn_lr, 'eta_min': 0.1*min_hn_lr},
    #         ], lr=hn_lr, weight_decay=0.05)
    # else:
    if adam_8bit:
        #optimizer = bnb.optim.AdamW8bit([{'params':hn.parameters(), 'initial_lr':hn_lr}], lr=hn_lr, weight_decay=0.05,betas=(0.9, 0.999))
        from soap import SOAP
        optimizer = SOAP(hn.parameters(), lr = hn_lr, betas=(.95, .95), weight_decay=0.05, precondition_frequency=10)
    else:
        optimizer = torch.optim.AdamW([{'params':hn.parameters(), 'initial_lr':hn_lr}], lr=hn_lr, weight_decay=0.05,betas=(0.9, 0.999))
    # betas=(0.9, 0.95)
   
    # if scale_weight_flag is not None:
    #     scheduler = CosineAnnealingLR_with_Group(optimizer, T_max=max_iter, eta_min=min_hn_lr, last_epoch=iter_num-1)
    # else:
    #scheduler_start_iter
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter-scheduler_start_iter, eta_min=min_hn_lr, last_epoch=iter_num-1)

    tic = time.time()
    with torch.no_grad():
        pesudo_x = torch.randn(1).to(device_id)
        _ = hn(pesudo_x)
    if env.world_size == 1:
        if env.global_rank == 0:
            state_dict_hn = hn.state_dict()
            env.print_master(f"Saving checkpoint to {out_dir}")
            hn_path = os.path.join(out_dir, f"hn-ckpt-{p:.2f}.pt")
            torch.save(state_dict_hn, hn_path)

    else:
        if hasattr(hn, "module"):
            state_dict_hn = hn.module.state_dict()
            env.print_master(f"Saving checkpoint to {out_dir}")
            hn_path = os.path.join(out_dir, f"hn-ckpt-{p:.2f}.pt")
            torch.save(state_dict_hn, hn_path)

        else:
            if env.world_size == 1:
                save_policy = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
            else:
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            with FSDP.state_dict_type(hn, StateDictType.FULL_STATE_DICT, save_policy):

                state_dict_hn = hn._orig_mod.state_dict()
                if env.global_rank == 0:
                    env.print_master(f"Saving hn checkpoint to {out_dir}")
                    hn_path = os.path.join(out_dir, f"hn-ckpt-{p:.2f}.pt")
                    torch.save(state_dict_hn, hn_path)
        

    torch.cuda.empty_cache()

    for params in model.parameters():
        params.requires_grad = False
    for params in hn.parameters():
        params.requires_grad = True
    hn.train()
    hn_moe_ddp_flag = False
    #if hasattr(hn,'module'):
    if isinstance(hn, torch.nn.parallel.DistributedDataParallel):

        if hasattr(hn.module,'model_list') and isinstance(hn.module.model_list, torch.nn.ModuleList):
            hn_moe_ddp_flag=True
    elif isinstance(hn, FSDP):
        if hasattr(hn._fsdp_wrapped_module,'model_list'):
            hn_moe_ddp_flag=True
    else:
        if isinstance(hn.model_list, torch.nn.ModuleList):
            hn_moe_ddp_flag=True
    env.print_master(hn_moe_ddp_flag)
    # if hasattr(model, "_orig_mod") and not use_fsdp:
    #     model = model._orig_mod

    env.print_master(hn)
    #print(model)
#    print('hn_moe_ddp_flag:' + str(hn_moe_ddp_flag))
    for batch in train_hn_data:

        reg_c_loss =  torch.scalar_tensor(0).to(device_id).float()
        width_loss =  torch.scalar_tensor(0).to(device_id).float()
        # The criteria to stop training 
        # torch.cuda.empty_cache()
        if iter_num>=max_iter:
            break
        with torch.no_grad():
            input_ids, targets = batch['input_ids'].to(device_id), batch['labels'].to(device_id)
            input_ids = input_ids[:,:hn_block_size]
            targets = targets[:,:hn_block_size]
            attention_mask = (input_ids != ignored_token).long().to(device_id)

            # print(input_ids.size())
            # print(targets.size())
        #print(input_ids.size())
        # (To be cleaned)
        # if bf_16:
        #     vectors = hn()
        #     hn_helper.set_gate_vectors(unwrap_model(model),vectors)
        #     with autocast(device_type='cuda',dtype=torch.bfloat16):
        #         logits = model(input_ids)
        #         loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignored_token)    
        # else:


        with autocast(device_type='cuda', dtype=data_type):
            # if semi_params:
            #     if hasattr(hn, 'module'):
            #         scales,biases = hn.module.param_forward()
            #     else:
            #         scales,biases = hn.param_forward()
            #     hn_helper.set_params_vectors(unwrap_model(model),scales,biases)
            if kd_loss or mix_loss:
                with torch.no_grad():
                    hn_helper.set_gate_status(unwrap_model(model), False)
                    teacher_output = model(input_ids, attention_mask=attention_mask)
                    if hasattr(teacher_output, 'logits'):
                        teacher_logits = teacher_output.logits
                    else:
                        teacher_logits = teacher_output

                    hn_helper.set_gate_status(unwrap_model(model), True)    
                #torch.cuda.empty_cache()
            if hn_moe_ddp_flag: 
                pesudo_x = torch.randn(1).to(device_id)
                #        _ = hn(pesudo_x)
                # vectors, pair_loss, hard_c_out, hard_out = hn(pesudo_x)
                vectors, width_loss, pair_loss, hard_c_out, hard_out  = hn(pesudo_x)
                #pair_loss = pair_loss_weight(step=iter_num)*pair_loss
                hn_helper.set_gate_vectors(unwrap_model(model),vectors)
            else:
                pesudo_x = torch.randn(1).to(device_id)
                vectors = hn(pesudo_x)
                hn_helper.set_gate_vectors(unwrap_model(model),vectors)
            #
            
            model_output = model(input_ids, attention_mask=attention_mask)

            # if hn_moe_ddp_flag or isinstance(hn.model_list, torch.nn.ModuleList):
            if hn_moe_ddp_flag:
                # hard_out = hn_helper.get_hard_out(unwrap_model(model))
                if hasattr(hn, "module"):
                    attn_exp_flag = hn.module.model_list[1].attn_experts
                else:
                    attn_exp_flag = hn.model_list[1].attn_experts
                if not attn_exp_flag and not gated_attn_flag:
                    # if top_k_mlp:
                    #     hard_out = hn_helper.get_mlp_topk_hard_out(unwrap_model(model))
                    # else:
                    hard_out = hn_helper.get_attn_hard_out(unwrap_model(model), hard_out)
                    pair_attn_loss = hn_helper.pair_attn_loss(unwrap_model(model))
                    pair_loss += pair_attn_loss
                    #pair_loss = pair_loss_weight(step=iter_num)*pair_loss

                if load_balance:
                    #print(unwrap_model(model))
                    load_balance_loss = hn_helper.load_balance_loss(unwrap_model(model))
                else:
                    load_balance_loss = torch.Tensor([0]).squeeze().to(device_id)

            if hasattr(model_output, 'logits'):
                logits = model_output.logits
                #loss = model_output.loss
            else:
                logits = model_output
            #logits = model(input_ids)
            if kd_loss:
                #loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignored_token)
                if foward_kl_loss:
                    loss = 2 * kd_loss_fn(logits.reshape(-1, logits.size(-1)), teacher_logits.reshape(-1, teacher_logits.size(-1)), targets.reshape(-1))
                else:
                    loss = 8 * kl_div_loss_with_ignore_index(logits.view(-1, logits.size(-1)), teacher_logits.view(-1, teacher_logits.size(-1)), targets.view(-1), ignore_index=ignored_token) +  kd_loss_fn(logits.reshape(-1, logits.size(-1)), teacher_logits.reshape(-1, teacher_logits.size(-1)), targets.reshape(-1))

                #loss = 2 * kd_loss_fn(logits.reshape(-1, logits.size(-1)), teacher_logits.reshape(-1, teacher_logits.size(-1)), targets.reshape(-1))
                # loss = 16 * torch.nn.KLDivLoss(reduction='batchmean')(torch.nn.functional.log_softmax(logits.view(-1, logits.size(-1)), dim=-1), torch.nn.functional.softmax(teacher_logits.view(-1, teacher_logits.size(-1)), dim=-1))
                # loss = 8 * torch.nn.KLDivLoss(reduction='batchmean',log_target=True)(torch.nn.functional.log_softmax(teacher_logits.view(-1, teacher_logits.size(-1)), dim=-1), torch.nn.functional.log_softmax(logits.view(-1, logits.size(-1)), dim=-1))
                #loss = 0.5*loss + 0.5*kd_loss
            elif mix_loss:
                #kd_loss_value = 16 * torch.nn.KLDivLoss(reduction='batchmean')(torch.nn.functional.log_softmax(logits.view(-1, logits.size(-1)), dim=-1), torch.nn.functional.softmax(teacher_logits.view(-1, teacher_logits.size(-1)), dim=-1))
                kd_loss_value = loss = 16 * kl_div_loss_with_ignore_index(logits.view(-1, logits.size(-1)), teacher_logits.view(-1, teacher_logits.size(-1)), targets.view(-1), ignore_index=ignored_token)
                lm_loss =  torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignored_token)
                loss = 0.7*kd_loss_value + 0.3*lm_loss
            else:
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignored_token)


            if not hn_moe_ddp_flag:
                    if hasattr(hn,'module'):
                        hard_out = hn.module.hard_output()
                    else:
                        hard_out = hn.hard_out()
        
            # reg_loss = torch.Tensor([0])
            #print(loss)
            reg_loss = param_reg(hard_out)

            pair_scale = float((iter_num % 3) == 0)
            reg_scale  = float((iter_num % 3) != 0)
            #float((iter_num % 2) == 1)
            loss = loss + reg_scale*reg_loss
            if soft_rank:
                soft_rank_loss = hn_helper.rank_reg(unwrap_model(model))
                loss = loss + soft_rank_loss
                #soft_rank_loss
            if hn_moe_ddp_flag:
                loss = loss + width_loss + pair_scale*pair_loss + load_balance_loss + reg_c_loss

        
        if torch.isnan(loss):
            # The data may be noisy. Ignore it when loss is nan.
            env.print_master(f"!!! nan loss detected !!!")
            loss.fill_(0)

        # if bf_16:
        toc = time.time() - tic
        # print(str(toc*1000) +' ms')
        # print(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(hn.parameters(), 1.0)
        # for name, param in hn.named_parameters():
        #     if param.grad is None:
        #         print(name)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        # else:
        #     loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()
        if use_sch and iter_num > scheduler_start_iter:
            scheduler.step()
        # scheduler.step()

        toc = time.time() - tic
        tic = time.time()
        if iter_num % log_interval == 0:
            if use_sch:
                if soft_rank:
                    env.print_master(f"iter {iter_num}/{max_iter}: loss {loss.item():.4f}, reg_loss {reg_loss.item():.4f}, soft_rank {soft_rank_loss.item():.4f}, lr: {scheduler.get_last_lr()}, time: {toc*1000:.2f}msS")
                elif hn_moe_ddp_flag:
                    env.print_master(f"iter {iter_num}/{max_iter}: loss {(loss-reg_scale*reg_loss-pair_scale*pair_loss-width_loss-load_balance_loss-reg_c_loss).item():.4f}, reg_loss {reg_loss.item():.4f}, pair_loss {pair_loss.item():.4f}, width_loss {width_loss.item():.4f}, reg_c_loss {reg_c_loss.item():.4f}, balance_loss {load_balance_loss.item():.4f}, lr: {scheduler.get_last_lr()},  time: {toc*1000:.2f}msS")
                else:
                    env.print_master(f"iter {iter_num}/{max_iter}: loss {loss.item():.4f}, reg_loss {reg_loss.item():.4f}, lr: {scheduler.get_last_lr()}, time: {toc*1000:.2f}msS")
            else:
                if hn_moe_ddp_flag:
                    env.print_master(f"iter {iter_num}/{max_iter}: loss {(loss-reg_scale*reg_loss-pair_scale*pair_loss-width_loss-load_balance_loss-reg_c_loss).item():.4f}, reg_loss {reg_loss.item():.4f}, pair_loss {pair_loss.item():.4f}, width_loss {width_loss.item():.4f}, reg_c_loss {reg_c_loss.item():.4f}, balance_loss {load_balance_loss.item():.4f}, time: {toc*1000:.2f}msS")
                else:
                    env.print_master(f"iter {iter_num}/{max_iter}: loss {loss.item():.4f}, reg_loss {reg_loss.item():.4f}, time: {toc*1000:.2f}msS")
        iter_num += 1
        if iter_num % save_interval == 0:
            if env.world_size == 1:
                if env.global_rank == 0:
                    state_dict_hn = hn.state_dict()
                    env.print_master(f"Saving checkpoint to {out_dir}")
                    hn_path = os.path.join(out_dir, f"hn-ckpt-iter-{iter_num:06d}-{p:.2f}.pt")
                    torch.save(state_dict_hn, hn_path)

            else:
                if hasattr(hn, "module"):
                    state_dict_hn = hn.module.state_dict()
                    env.print_master(f"Saving checkpoint to {out_dir}")
                    hn_path = os.path.join(out_dir, f"hn-ckpt-iter-{iter_num:06d}-{p:.2f}.pt")
                    torch.save(state_dict_hn, hn_path)

                else:
                    if env.world_size == 1:
                        save_policy = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
                    else:
                        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

                    with FSDP.state_dict_type(hn, StateDictType.FULL_STATE_DICT, save_policy):
                        state_dict_hn = hn._orig_mod.state_dict()
                        if env.global_rank == 0:
                            env.print_master(f"Saving hn checkpoint to {out_dir}")
                            hn_path = os.path.join(out_dir, f"hn-ckpt-iter-{iter_num:06d}-{p:.2f}.pt")
                            torch.save(state_dict_hn, hn_path)
            torch.cuda.empty_cache()

    if env.world_size == 1:
        if env.global_rank == 0:
            state_dict_hn = hn.state_dict()
            env.print_master(f"Saving checkpoint to {out_dir}")
            hn_path = os.path.join(out_dir, f"hn-ckpt-final-{p:.2f}.pt")
            torch.save(state_dict_hn, hn_path)

    else:
        if hasattr(hn, "module"):
            state_dict_hn = hn.module.state_dict()
            env.print_master(f"Saving checkpoint to {out_dir}")
            hn_path = os.path.join(out_dir, f"hn-ckpt-final-{p:.2f}.pt")
            torch.save(state_dict_hn, hn_path)

        else:
            if env.world_size == 1:
                save_policy = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
            else:
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            with FSDP.state_dict_type(hn, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict_hn = hn._orig_mod.state_dict()
                if env.global_rank == 0:
                    env.print_master(f"Saving hn checkpoint to {out_dir}")
                    hn_path = os.path.join(out_dir, f"hn-ckpt-final-{p:.2f}.pt")
                    torch.save(state_dict_hn, hn_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    from jsonargparse import CLI
    CLI(main)