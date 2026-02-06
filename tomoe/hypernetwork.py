import torch
import torch.nn as nn
import torch.nn.functional as F
# from misc_functions import custom_grad_weight
import numpy as np
import math
from typing import Optional

class custom_grad_weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grad_w=1):
        ctx.grad_w = grad_w
        input_clone = input.clone()
        return input_clone.float()
    @staticmethod
    def backward(ctx, grad_out):
        grad_input = ctx.grad_w * grad_out
        return grad_input, None

def sample_gumbel(shape,eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U+eps)+eps)

def gumbel_sigmoid_function(logits: torch.Tensor, tau: float = 1, hard: bool = False,  sample: bool = True, offset=0) -> torch.Tensor:
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """
    if sample:
        device = logits.get_device()
        gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format, device=device).exponential_().log()
        )  # ~Gumbel(0, 1)
        gumbels = (logits + gumbels + offset) / tau  # ~Gumbel(logits, tau)
    else:
        gumbels = (logits + offset) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        y_hard = torch.round(y_soft)
        ret = (y_hard - y_soft).detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def gumbel_softmax_sample(logits,  T, sample=True):
    gumbel_sample = sample_gumbel(logits.size())
    if logits.get_device() == -1:
        logits = logits.cpu()
        gumbel_sample = gumbel_sample.cpu()
    else:
        gumbel_sample = gumbel_sample.to(logits.get_device())

    if sample:
        y = logits + gumbel_sample
    else:
        y = logits
    return F.softmax(y / T, dim=-1)


def gumbel_softmax(logits, T, hard_sample=False, return_soft=False, sample=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    shape = logits.size()
    logits = logits.view(-1, shape[-1])
    # y = gumbel_softmax_sample(logits, T)
    y = gumbel_softmax_sample(logits, T, sample)
    if not hard_sample:
        return y.view(shape)
    else:
        #shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        if return_soft:
            return y_hard.view(shape), y.view(shape)
        else:
            return y_hard.view(shape)

def generate_mask_function(th, soft_range, pos):
    outputs = (1/soft_range)*(soft_range + th - pos)
    return torch.clamp(outputs, min=0, max=1)

def hard_sample(out):
    binary_out = torch.round(out)
    binary_out = (binary_out - out).detach() + out
    return binary_out

def hard_topk(out, k):
    # Compute the binary mask of top-k elements along the last dimension
    topk = torch.topk(out, k, dim=-1)
    indices = topk.indices
    mask = torch.zeros_like(out)
    mask.scatter_(-1, indices, 1.0)
    # Save nothing for backward since gradient flows through as identity
    mask = (mask - out).detach() + out
    return mask

def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)

def minmax_reg_loss(x,y,c=0):
    x=x+c
    y=y+c
    max_tensor = torch.maximum(x,y)
    min_tensor = torch.minimum(x,y)

    loss = torch.log(max_tensor/min_tensor)
    return loss.mean()

def experts_union(experts_feature):
    return 1 - torch.prod(1 - experts_feature, dim=0)

class experts_module_list(nn.Module):
    def __init__(self, structures, model_dim, experts=8, alpha=1, head_dim=128, qk_static_flag=True, num_kv_heads=1):
        super(experts_module_list, self).__init__()
        self.structures = structures
        self.attn_flag = [False]*len(self.structures)
        self.module_list = nn.ModuleList()
        self.attn_experts = False
        self.use_gated_attn_flag = [False]*len(self.structures)
        self.use_top_k_mlp = [False]*len(self.structures)
        for i in range(len(self.structures)):
            if self.structures[i] <= model_dim:
                #if not attn_experts:
                self.attn_flag[i] = True

            module = single_experts_module(structures[i], model_dim, head_dim, experts, self.attn_flag[i], qk_static_flag=qk_static_flag, num_kv_heads=num_kv_heads)

            self.module_list.append(module)
                    
        
        self.alpha = alpha
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_modules = len(self.module_list)

    def resource_forward(self, rnn_states):
        total_pair_loss = 0
        width_list = []

        for i in range(self.num_modules):
            if self.attn_flag[i]:
                pair_loss, width_mean = self.module_list[i].emb_constrain(rnn_states[i])
            else:
                pair_loss, width_mean= self.module_list[i].emb_constrain(rnn_states[i])

            total_pair_loss += pair_loss

            width_list.append(width_mean)

        total_pair_loss = total_pair_loss/self.num_modules

        return self.alpha*total_pair_loss, width_list
    
class hn_module_list(nn.Module):
    def __init__(self, rnn, experts_list):
        super(hn_module_list, self).__init__()
        self.model_list = torch.nn.ModuleList([rnn, experts_list])
    def forward(self, x=None):
        vectors = self.model_list[0]()
        pair_loss, hard_out = self.model_list[1].resource_forward(vectors)

        return vectors, pair_loss, hard_out
    
    def set_hn_inputs_grad(self, ):
        self.model_list[0].inputs.requires_grads = True

class single_experts_module(nn.Module):
    def __init__(self, mlp_dim, model_dim, head_dim=None, experts=8, attn_flag=False, qk_static_flag=False, num_kv_heads=1):
        super(single_experts_module, self).__init__()
        self.experts = experts
        self.T = 0.4
        self.base = 3.0
        self.emb_dim = 128
        self.mlp_dim = mlp_dim
        self.model_dim = model_dim
        self.num_kv_heads = num_kv_heads
        #self.attn_experts = attn_experts
        assert type(self.experts) == int
        assert self.experts >= 1

        if attn_flag:
            self.linear_router = nn.Linear(self.model_dim, self.emb_dim, bias=False)

            self.linear_decoder = nn.Linear(self.emb_dim, head_dim + int(head_dim/2), bias=False)
            #self.top_k_flag = top_k_flag

        else:
            self.linear_router = nn.Linear(self.model_dim, self.experts, bias=False) 
            self.linear_decoder = nn.Linear(self.emb_dim, mlp_dim, bias=False)
            #self.top_k_flag = False

        self.ln = nn.LayerNorm([self.emb_dim])
        self.experts_for_eval = None
        self.attn_flag = attn_flag
        self.head_dim = head_dim

        self.qk_static_flag = qk_static_flag
        self.width = None
    def forward(self, x=None, rnn_state=None):
        #self.rnn_state = rnn_state.mean(dim=0)
        router_logits = None
        if self.attn_flag:
            out = self.linear_router(x)
           
            #num_tokens = x.size(1)
            batch_size, num_tokens, _ = x.shape  # x: (B, T, emb_dim)
            routed_emb = out + rnn_state.mean(dim=0)
            
            output_dynamic = self.linear_decoder(F.gelu(self.ln(routed_emb)))[... , :self.head_dim]  # Shape: [batch_size, head_dim + int(mlp_dim/2)]
            output_constant = self.linear_decoder(F.gelu(self.ln(rnn_state.mean(dim=0).unsqueeze(0))))[..., self.head_dim:]
            output_constant = output_constant.expand(batch_size, num_tokens, -1)

            out_before_binary = torch.cat([output_dynamic, output_constant], dim=-1)
            if output_dynamic.ndim==2:
                output_dynamic = output_dynamic.unsqueeze(0)

            binary = gumbel_sigmoid_function(logits=out_before_binary, tau=self.T, offset=self.base, hard=True, sample=True)
            self.binary = binary
        
        else:
            out = self.linear_router(x)
            batch_size, num_tokens, _ = x.shape  # x: (B, T, emb_dim)
            router_logits = gumbel_softmax(out, T=self.T, hard_sample=True)

            routed_emb = torch.matmul(router_logits, rnn_state)
            out_before_binary = self.linear_decoder(F.gelu(self.ln(routed_emb)))
            binary = gumbel_sigmoid_function(logits=out_before_binary, tau=self.T, offset=self.base, hard=True, sample=True)
        
        return binary, router_logits


    def prepare_experts(self, rnn_state, non_uniform = False):
        #8x128
        # full_embeding = self.experts_embeding + rnn_state[None,:]
        if self.attn_flag:
            # width_mean = witdh_cover = 0
            self.rnn_state = rnn_state
            device = rnn_state.get_device()
            if self.qk_static_flag:
                output_constant = self.linear_decoder(F.gelu(self.ln(rnn_state.mean(dim=0).unsqueeze(0))))[:, self.head_dim:]
                binary_approx_part2 = gumbel_sigmoid_function(output_constant, offset=self.base, tau=self.T, sample=True).squeeze()
                binary_part2 = hard_sample(binary_approx_part2)
                if binary_part2.sum() == 0:
                    idx = torch.argmax(binary_approx_part2)
                    binary_part2[idx] = 1.0

                self.experts_for_eval = binary_part2.repeat(2)
                
                width_mean = [torch.scalar_tensor(0).to(device).float(), 2*binary_part2.sum(dim=-1).item()]
                width_cover = torch.scalar_tensor(0).to(device).float()
                return width_mean, width_cover
            width_cover = width_mean = torch.scalar_tensor(0).to(device).float()
            return width_mean, width_cover
        else:
            full_embeding = rnn_state
            
            #8xmiddle
            out_before_binary = self.linear_decoder(F.gelu(self.ln(full_embeding)))
            #8xmiddle
            binary_approx = gumbel_sigmoid_function(out_before_binary, offset=self.base, tau=self.T, sample=True).squeeze()

            binary = hard_sample(binary_approx)
            # binary = binary_approx

            #width_mean = binary.sum(dim=-1).mean()
            if not non_uniform:
                width_mean = binary.sum(dim=-1).max()
                #width_cover = torch.count_nonzero(binary.mean(0))
                values, indices = torch.topk(binary_approx, k=int(width_mean.item()), dim=-1, largest=True, sorted=True)
                # if self.attn_flag:
                
                self.experts_for_eval = torch.zeros_like(binary).to(torch.uint8)
                for i in range(indices.size(0)):
                    self.experts_for_eval[i,indices[i]] = 1
                width_cover = experts_union(self.experts_for_eval)
            else:
                self.experts_for_eval = binary
                #.to(torch.uint8)
                width_mean = binary.sum(dim=-1).max()
                width_cover = experts_union(binary)
            print(width_mean)
            return width_mean, width_cover
    
    def emb_constrain(self, rnn_state):
        #8x128
        # full_embeding = self.experts_embeding + rnn_state[None,:]
        
        if self.attn_flag:
            #width_final = []
            #full_embeding = rnn_state
            device = rnn_state.get_device()
            pair_loss =  torch.scalar_tensor(0).to(device).float()
            width_final = torch.scalar_tensor(0).to(device).float()
            # width_final = torch.scalar_tensor(0).to(device).float()
            # output_constant = self.linear_decoder(F.gelu(self.ln(full_embeding.mean(dim=0).unsqueeze(0))))[:, self.head_dim:]
            # binary = gumbel_sigmoid_function(logits=output_constant, tau=self.T, offset=self.base, sample=True, hard=True).squeeze()
            # width_final.append(binary[...,:self.head_dim].sum(-1).squeeze())

            return pair_loss.to(device), width_final
        else:
            full_embeding = rnn_state
            
            out_before_binary = self.linear_decoder(F.gelu(self.ln(full_embeding)))
            binary = gumbel_sigmoid_function(logits=out_before_binary, tau=self.T, offset=self.base, sample=True, hard=True).squeeze()

            device = binary.get_device()

            union_of_experts = experts_union(binary)
            
            width_final = binary.sum(dim=-1).max().squeeze()

            pair_loss = minmax_reg_loss(union_of_experts.mean(), torch.scalar_tensor(1).to(device).float(), c=0.001)


            return pair_loss.to(device), width_final


class hypernetwork(nn.Module):
    def __init__(self, t_structures, emb_dim=128, experts=8, constant_expert=False):
        super(hypernetwork, self).__init__()

        self.t_sp = t_structures
        self.constant_expert = constant_expert
        if self.constant_expert:
            experts = experts+1
        self.h0 = torch.zeros(2,experts,emb_dim//2)

        self.bi_GRU = nn.GRU(emb_dim//4, emb_dim//2, bidirectional=True)
        self.inputs = nn.Parameter(torch.Tensor(len(t_structures),experts,emb_dim//4))
        nn.init.normal_(self.inputs)
        self.inputs.requires_grad=False
        

    def forward(self):
        if self.inputs.get_device() == -1:
            self.h0 = self.h0.cpu()
        else:
            self.h0 = self.h0.to(self.inputs.get_device())

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            outputs, hn = self.bi_GRU(self.inputs, self.h0)

        return outputs.to(self.inputs.dtype)

class virtual_dynamic_operation(nn.Module):
    def __init__(self, middle_dim, emb_dim=128):
        super().__init__()
        self.router = None
        self.emb_dim = emb_dim
        self.middle_dim = middle_dim
        self.router_logits = None

        self.rnn_state = torch.zeros(self.emb_dim)
    def forward(self, input):
        if self.router is not None:
            outputs, router_logits = self.router(input, self.rnn_state)
            self.router_logits = router_logits

            return outputs
        else:
            return torch.ones(self.middle_dim).to(input.get_device())
    
    def set_rnn_state(self, rnn_state):
        self.rnn_state = torch.zeros(rnn_state.size(0), self.emb_dim)

        assert rnn_state.squeeze().size() == self.rnn_state.squeeze().size()
        if rnn_state is not None:
            self.rnn_state = rnn_state.squeeze()
        else:
            self.rnn_state = rnn_state

    def set_router_module(self, expert_module):
        self.router = expert_module
    def reset_router_module(self,):
        self.router = None


    def router_logits_balance_loss(self,):
        if isinstance(self.router, single_experts_module):
            if self.router.attn_flag:
                return 0
            else:
                num_experts = self.router.experts
                _, selected_experts = torch.topk(self.router_logits, k=1, dim=-1) # [batch_size X sequence_length, top_k]
                expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts) # [batch_size X sequence_length, top_k, num_experts]
                tokens_per_expert = torch.mean(expert_mask.float(), dim=0) # [top_k, num_experts]
                # Compute the average probability of routing to these experts
                router_prob_per_expert = torch.mean(self.router_logits, dim=0) # [num_experts]
                overall_loss = torch.mean(tokens_per_expert * router_prob_per_expert.unsqueeze(0)) # / top_k
                return overall_loss * num_experts


def generate_random_mask_like(input_tensor, mask_prob=0.96):
    """
    Generate a random binary mask with the same shape, device, and dtype as the input tensor.

    Args:
        input_tensor (torch.Tensor): Reference tensor to infer shape, device, and dtype.
        mask_prob (float): Probability of setting a position to 1 (masked).
    
    Returns:
        torch.Tensor: A binary mask of the same shape, device, and dtype as input_tensor.
    """
    bs, seq_len, _, head_dim = input_tensor.shape
    mask = (torch.rand(bs, seq_len, 1, head_dim, dtype=torch.float, device="cpu") < mask_prob).to(input_tensor.dtype)
    return mask.to(input_tensor.device)  # Move to the correct device

class virtual_basic_operation(nn.Module):
    def __init__(self, dim,ex_dict={}):
        super().__init__()
        self.dim = dim
        self.pruning_vector = torch.ones(dim)
        self.ex_dict = ex_dict
        # if ex_dict_vo['head_dim']
        # print(self.pruning_vector.size())

    def forward(self, input, pv_detach=False, grad_w=1.0, mode = None):
        #print(self.pruning_vector.size())
        dtype = input.dtype
        if self.pruning_vector.size() == input.size():
            #print(self.pruning_vector.sum(-1).max())
            return input*self.pruning_vector

        if len(input.size())==4:
            if self.dim == input.size(1) or self.dim == input.size(-1):
                seq_len = input.size(-2)
                if len(self.pruning_vector.squeeze().size())==1:
                    seq_len = 1
                # print(input.size())
                # print(self.pruning_vector.size())
                if mode == 'head':
                    if len(self.pruning_vector.squeeze().size())==1:
                        p_v =  self.pruning_vector.squeeze()[...,:self.dim].view(1,1,-1, 1).transpose(1,2)
                    else:
                        p_v =  self.pruning_vector.squeeze()[...,:self.dim].view(1,seq_len,-1, 1).transpose(1,2)
                elif mode == 'inner_head':
                    if (self.pruning_vector.size(-1)-self.dim) < input.size(-1):
                        p_v =  self.pruning_vector.squeeze()[...,self.dim:].view(1,seq_len, -1, self.pruning_vector.size(-1)-self.dim).repeat(1,1,1,2).transpose(1,2)
                        # print(self.pruning_vector.size())
                        # print(p_v.size())
                    else:
                        p_v =  self.pruning_vector.squeeze()[...,self.dim:].view(1,seq_len, -1, input.size(-1)).transpose(1,2)
                elif mode == 'inner_head_v':
                    if len(self.pruning_vector.squeeze().size())==1:
                        p_v =  self.pruning_vector.squeeze()[...,:self.dim].view(1,1,1, -1).transpose(1,2)
                        # print(self.pruning_vector.size())
                        # print(p_v.size())
                    else:
                        p_v =  self.pruning_vector.squeeze()[...,:self.dim].view(1,seq_len, -1, input.size(-1)).transpose(1,2)
                elif mode == 'inner_head_o':
                    seq_len = input.size(1)
                    if len(self.pruning_vector.squeeze().size())==1:
                        p_v =  self.pruning_vector.squeeze()[...,:self.dim].view(1,1,1, -1).transpose(1,2)
                    else:
                        p_v =  self.pruning_vector.squeeze()[...,:self.dim].view(1,seq_len, -1, input.size(-1))
                p_v = custom_grad_weight.apply(p_v, grad_w)
                if pv_detach:
                    p_v = p_v.detach()
                input = p_v.expand_as(input)*input
                if mode == 'inner_head_v' and len(self.pruning_vector.squeeze().size())==1:
                    input = generate_random_mask_like(input).expand_as(input)*input

            else:

                p_v = self.pruning_vector[None,None,None,:]
                p_v = custom_grad_weight.apply(p_v, grad_w)
                if pv_detach:
                    p_v = p_v.detach()
                if input.get_device() == -1:
                    p_v = p_v.cpu()
                else:
                    p_v = p_v.to(input.get_device())

                input = p_v.expand_as(input)*input
            # return input
        elif len(input.size())==3:
            p_v = self.pruning_vector[None,None,:]
            p_v = custom_grad_weight.apply(p_v, grad_w)
            if pv_detach:
                    p_v = p_v.detach()
            if input.get_device() == -1:
                p_v = p_v.cpu()
            else:
                p_v = p_v.to(input.get_device())
            input = p_v.expand_as(input) * input
        elif len(input.size())==2:
            p_v = self.pruning_vector[None,:]
            p_v = custom_grad_weight.apply(p_v, grad_w)
            if pv_detach:
                    p_v = p_v.detach()
            if input.get_device() == -1:
                p_v = p_v.cpu()
            else:
                p_v = p_v.to(input.get_device())
            input = p_v.expand_as(input) * input
        # print(p_v.sum(-1).max())
        return input.to(dtype)
    
    
    def set_vector_value(self, value):
        # print(self.pruning_vector.size())
        self.pruning_vector = torch.ones(self.dim)
        # assert value.squeeze().size(-1) == self.pruning_vector.squeeze().size(-1)
        #if value is not None:
        # print(value.size())
        self.pruning_vector = value.squeeze()
        if len(self.pruning_vector.size())==2:
            self.pruning_vector = self.pruning_vector.unsqueeze(0)
        # else:
        #     self.pruning_vector = value
    def get_parameters(self):
        return 0

class virtual_block_basic_operation(virtual_basic_operation):
    def __init__(self, dim,ex_dict={}):
        super().__init__(dim=dim, ex_dict=ex_dict)

class virtual_vo_operation(virtual_basic_operation):
    def __init__(self, dim,ex_dict={}):
        super().__init__(dim=dim, ex_dict=ex_dict)

class virtual_att_operation(virtual_basic_operation):
    def __init__(self, dim,ex_dict={}):
        super().__init__(dim=dim, ex_dict=ex_dict)
        self.head_dim = ex_dict['head_dim']
    def get_parameters(self):
        return self.ex_dict['dim_1'] * self.ex_dict['dim_2'] * self.ex_dict['num_weight']
    def forward(self, input):
        # self.pruning_vector =  self.pruning_vector.repeat_interleave(self.head_dim)
        if len(input.size())==4:
            p_v = self.pruning_vector[None,None,:,None]
            if input.get_device() == -1:
                p_v = p_v.cpu()
            else:
                p_v = p_v.to(input.get_device())

            input = p_v.expand_as(input)*input
        return input

class virtual_block_attn_operation(virtual_basic_operation):
    def __init__(self, dim,ex_dict={}):
        super().__init__(dim=dim, ex_dict=ex_dict)
        self.head_dim = ex_dict['head_dim']
    def get_parameters(self):
        if 'num_kv_heads' in self.ex_dict:
            return self.ex_dict['dim_1']*self.ex_dict['num_kv_heads']*self.ex_dict['head_dim']*2 + self.ex_dict['dim_1']*self.ex_dict['num_heads']*self.ex_dict['head_dim']*2
        else:
            return self.ex_dict['dim_1'] * self.ex_dict['dim_2'] * self.ex_dict['num_weight']

class virtual_mlp_operation(virtual_basic_operation):
    def __init__(self, dim,ex_dict={}):
        super().__init__(dim=dim, ex_dict=ex_dict)
    def get_parameters(self):
       
        return self.ex_dict['dim_1'] * self.ex_dict['dim_2'] * self.ex_dict['num_weight']