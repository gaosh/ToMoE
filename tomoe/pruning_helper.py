import torch
from tqdm.auto import tqdm
import time
from torch.cuda.amp import GradScaler 
import torch.nn as nn


from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
# from hn_modules.param_util import unwrap_model
# from hn_modules.param_util import unwrap_model
import numpy as np
from torch.cuda.amp import autocast
import torch.nn as nn
import os
import math
from .hypernetwork import experts_union, minmax_reg_loss, custom_grad_weight, hard_sample
import torch.nn.functional as F

def entropy_loss(probs):
    """
    Computes the entropy loss of a batch of probability distributions.

    Args:
        probs (torch.Tensor): Tensor of shape (batch_size, num_classes) representing
                              the probabilities output by a model (after softmax).

    Returns:
        torch.Tensor: A scalar tensor representing the average entropy over the batch.
    """
    # Add a small value to probs to prevent log(0)
    probs = probs + 1e-8
    log_probs = torch.log(probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # Entropy for each sample
    return entropy.mean()  # Average entropy over the batch

def log_inv_function(sum_params, sum_ori_params, p):

    param_ratio = (sum_params) / (sum_ori_params)

    if param_ratio>p:

        clampled_p_ratio = torch.clamp(param_ratio, min=p)

        loss = torch.log(clampled_p_ratio/p)
    else:
        clampled_p_ratio = torch.clamp(param_ratio, max=p)

        loss = torch.log(p/clampled_p_ratio)
            
    return loss

class collect_info_reg_llama(nn.Module):
    def __init__(self, model, p=None, lam=4.0, factor=0.7):
        super(collect_info_reg_llama, self).__init__()
        self.sum_ori_params = 0
        self.p = p
        self.constant_p = factor*self.p
        self.in_dim_list = []
        self.out_dim_list = []
        self.num_w_list = []
        self.structures = []
        self.gate_type = []
        self.lam = lam
        self.model_dim = None
        self.width_piror = None
        print(self.gate_type)

        modules = list(model.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if type(m).__name__ == 'virtual_block_basic_operation':
                if self.model_dim == None:
                    self.model_dim = m.dim
                #self.structures.append(m.dim)
                self.in_dim_list.append(None)
                self.out_dim_list.append(None)
                self.num_w_list.append(None)
                self.gate_type.append('mlp_block')
            if type(m).__name__ == 'virtual_mlp_operation':
                ori_param = m.get_parameters()
                self.sum_ori_params += ori_param
                self.in_dim_list.append(m.ex_dict['dim_1'])
                self.out_dim_list.append(m.ex_dict['dim_2'])
                self.num_w_list.append(m.ex_dict['num_weight'])
                self.structures.append(m.dim)
                self.gate_type.append('mlp')
            if type(m).__name__ == 'virtual_vo_operation':
                # ori_param = m.get_parameters()
                self.in_dim_list.append(m.ex_dict['dim_1'])
                self.out_dim_list.append(m.ex_dict['dim_2'])
                #self.structures.append(m.dim)
                #num_groups
                self.num_groups = m.ex_dict['num_groups']
                self.head_dim = m.ex_dict['head_dim']
                #self.num_heads = m.ex_dict['dim_1']//m.ex_dict['head_dim']
                self.num_heads = m.ex_dict['num_heads']
                self.num_kv_heads = m.ex_dict['num_kv_heads']
                #self.structures.append(m.dim)
                self.structures.append(m.dim)
                self.gate_type.append('attn')
            if type(m).__name__ == 'virtual_block_attn_operation':
                if self.model_dim == None:
                    self.model_dim = m.dim
                ori_param = m.get_parameters()
                self.sum_ori_params += ori_param
                self.in_dim_list.append(m.ex_dict['dim_1'])
                self.out_dim_list.append(m.ex_dict['dim_2'])
                self.num_w_list.append(m.ex_dict['num_weight'])
                #self.structures.append(m.dim)
                #num_groups
                self.num_groups = m.ex_dict['num_groups']
                self.head_dim = m.ex_dict['head_dim']
                #self.num_heads = m.ex_dict['dim_1']//m.ex_dict['head_dim']
                self.gate_type.append('attn_block')
            if type(m).__name__ == 'virtual_basic_operation':
                if self.model_dim == None:
                    self.model_dim = m.dim
                #self.structures.append(m.dim)
                self.in_dim_list.append(None)
                self.out_dim_list.append(None)
                self.num_w_list.append(None)
                self.gate_type.append('basic_gate')

        print("number of oringal parameters: %.3f" % (self.sum_ori_params / 10 ** 6))
        print(self.gate_type)
    
    def count_current_params(self, vectors):
        print("number of heads: %.1f" %(self.num_heads))
        print("number of kv heads: %.1f" %(self.num_heads))
        print("model dim: %.1f"%(self.model_dim))
        with torch.no_grad():
            att_flag = False
            sum_params = 0
            i = 0
            ind = 0
            model_dim = self.model_dim
            while i < len(self.out_dim_list):
                #### for attn block ####
                if self.gate_type[i] == 'attn':
                    #current_params =  model_dim * 2 * vectors[ind]*self.head_dim + model_dim * 2 *  vectors[ind]*self.head_dim*self.num_groups
                    if isinstance(vectors[ind], list): 
                        #current_params = model_dim * 2 * self.num_heads*vectors[ind][1] + model_dim * 2* vectors[ind][0]*self.num_heads*self.num_groups
                        reg_heads = model_dim * self.num_heads*vectors[ind][1] +  model_dim * self.num_heads*vectors[ind][0] 
                        kv_heads = model_dim * self.num_kv_heads*vectors[ind][1] +  model_dim * self.num_kv_heads*vectors[ind][0] 
                        current_params = reg_heads + kv_heads
                    else:
                        current_params =  model_dim * 2 * vectors[ind]*self.head_dim + model_dim * 2 *  vectors[ind]*self.head_dim*self.num_groups
                    i=i+1
                    ind +=1
                    sum_params += current_params
                    att_flag = True

                elif self.gate_type[i] == 'mlp':
                    block_mlp_in_dim = model_dim
                    block_mlp_middle_dim = vectors[ind]
                    block_mlp_out_dim = model_dim
                    current_params = block_mlp_in_dim*block_mlp_middle_dim +  block_mlp_in_dim*block_mlp_middle_dim + block_mlp_middle_dim*block_mlp_out_dim
                    i=i+1
                    ind +=1
                    sum_params += current_params
                else:
                    i=i+1
            print("current parameters: %.3f" % (sum_params / 10 ** 6))
            return sum_params
        
    def forward(self, vectors):
        if self.width_piror != None:
            index = 0
            sum_loss = 0
            model_dim = self.model_dim
            for index in range(len(vectors)):
                if self.structures[index] == self.num_heads:
                    current_params =  model_dim * 2 * self.num_heads*vectors[index][1] + model_dim * 2 *  vectors[index][0]*self.head_dim*self.num_groups
                    target_params = model_dim * 2 * self.width_piror[index]*self.head_dim + model_dim * 2 *  self.width_piror[index]*self.head_dim*self.num_groups
                    local_params =  model_dim * 2 * self.structures[index] *self.head_dim + model_dim * 2 *   self.structures[index] *self.head_dim*self.num_groups
                else:
                    block_mlp_in_dim = model_dim
                    block_mlp_middle_dim = vectors[index]
                    block_mlp_out_dim = model_dim
                    current_params = block_mlp_in_dim*block_mlp_middle_dim +  block_mlp_in_dim*block_mlp_middle_dim + block_mlp_middle_dim*block_mlp_out_dim
                    target_params = block_mlp_in_dim*self.width_piror[index] +  block_mlp_in_dim*self.width_piror[index] + self.width_piror[index]*block_mlp_out_dim
                    local_params =  block_mlp_in_dim*self.structures[index] +  block_mlp_in_dim*self.structures[index] + self.structures[index]*block_mlp_out_dim
                sum_loss = sum_loss + minmax_reg_loss(current_params/local_params, target_params/local_params, c=0.01)
                #index+=1
            loss = sum_loss/len(vectors)
        else:

            att_flag = False
            sum_params = 0
            np_sum_params = 0

            i = 0
            ind = 0
            model_dim = self.model_dim
            while i < len(self.out_dim_list):
                #### for attn block ####
                if self.gate_type[i] == 'attn':
                    
                    # current_params =  model_dim * 2 * self.num_heads*vectors[ind][1] + model_dim * 2 * vectors[ind][0]*self.num_heads*self.num_groups
                    reg_heads = model_dim * self.num_heads*vectors[ind][1] +  model_dim * self.num_heads*vectors[ind][0] 
                    kv_heads = model_dim * self.num_kv_heads*vectors[ind][1] +  model_dim * self.num_kv_heads*vectors[ind][0] 
                    current_params = reg_heads + kv_heads
            
                    ind +=1
                    i=i+1
                    sum_params += current_params
                    att_flag = True

                elif self.gate_type[i] == 'mlp':
                    block_mlp_in_dim = model_dim
                    block_mlp_middle_dim = vectors[ind]
                    block_mlp_out_dim = model_dim
                    current_params = block_mlp_in_dim*block_mlp_middle_dim +  block_mlp_in_dim*block_mlp_middle_dim + block_mlp_middle_dim*block_mlp_out_dim
                    i=i+1
                    ind +=1
                    if not torch.is_tensor(sum_params):
                        sum_params = torch.as_tensor(sum_params, device=current_params.device, dtype=current_params.dtype)

                    # promote scalar -> tensor with matching shape if needed
                    if sum_params.ndim == 0:
                        sum_params = sum_params.expand_as(current_params).clone()
                    sum_params += current_params

                else:
                    i=i+1
            param_ratio = sum_params / (self.sum_ori_params - np_sum_params)
            corrected_p = (self.sum_ori_params*self.p - np_sum_params)/(self.sum_ori_params - np_sum_params)
            # print(np_sum_params)
            # print(param_ratio)
            if not torch.is_tensor(corrected_p):
                corrected_p = torch.as_tensor(corrected_p, device=current_params.device, dtype=current_params.dtype)
            loss = minmax_reg_loss(param_ratio, corrected_p, c=0.001)

        return self.lam * loss

class collect_info_reg_phi(nn.Module):
    def __init__(self, model, p=None, lam=4.0, factor=0.7):
        super(collect_info_reg_phi, self).__init__()
        self.sum_ori_params = 0
        self.p = p
        self.constant_p = factor*self.p
        self.in_dim_list = []
        self.out_dim_list = []
        self.num_w_list = []
        self.structures = []
        self.gate_type = []
        self.lam = lam
        self.model_dim = None
        print(self.gate_type)

        modules = list(model.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            # print(type(m))
            # if isinstance(m, virtual_share_operation):
            if type(m).__name__ == 'virtual_block_basic_operation':
                if self.model_dim == None:
                    self.model_dim = m.dim
                #self.structures.append(m.dim)
                self.in_dim_list.append(None)
                self.out_dim_list.append(None)
                self.num_w_list.append(None)
                self.gate_type.append('mlp_block')
            if type(m).__name__ == 'virtual_mlp_operation':
                ori_param = m.get_parameters()
                self.sum_ori_params += ori_param
                self.in_dim_list.append(m.ex_dict['dim_1'])
                self.out_dim_list.append(m.ex_dict['dim_2'])
                self.num_w_list.append(m.ex_dict['num_weight'])
                self.structures.append(m.dim)
                self.gate_type.append('mlp')
            if type(m).__name__ == 'virtual_vo_operation':
                # ori_param = m.get_parameters()
                self.in_dim_list.append(m.ex_dict['dim_1'])
                self.out_dim_list.append(m.ex_dict['dim_2'])
                #self.structures.append(m.dim)
                #num_groups
                self.num_groups = m.ex_dict['num_groups']
                self.head_dim = m.ex_dict['head_dim']
                self.num_heads = m.ex_dict['dim_1']//m.ex_dict['head_dim']
                #self.structures.append(m.dim)
                self.structures.append(m.dim)
                self.gate_type.append('attn')
            if type(m).__name__ == 'virtual_block_attn_operation':
                if self.model_dim == None:
                    self.model_dim = m.dim
                ori_param = m.get_parameters()
                self.sum_ori_params += ori_param
                self.in_dim_list.append(m.ex_dict['dim_1'])
                self.out_dim_list.append(m.ex_dict['dim_2'])
                self.num_w_list.append(m.ex_dict['num_weight'])
                #self.structures.append(m.dim)
                #num_groups
                self.num_groups = m.ex_dict['num_groups']
                self.head_dim = m.ex_dict['head_dim']
                self.num_heads = m.ex_dict['dim_1']//m.ex_dict['head_dim']
                self.gate_type.append('attn_block')
            if type(m).__name__ == 'virtual_basic_operation':
                if self.model_dim == None:
                    self.model_dim = m.dim
                #self.structures.append(m.dim)
                self.in_dim_list.append(None)
                self.out_dim_list.append(None)
                self.num_w_list.append(None)
                self.gate_type.append('basic_gate')

        print("number of oringal parameters: %.3f" % (self.sum_ori_params / 10 ** 6))
        print(self.gate_type)
    
    def count_current_params(self, vectors):
        with torch.no_grad():
            att_flag = False
            sum_params = 0

            i = 0
            ind = 0
            model_dim = self.model_dim
            while i < len(self.out_dim_list):
                #### for attn block ####
                if self.gate_type[i] == 'attn':
                    
                    if isinstance(vectors[ind], list): 
                        current_params = model_dim * 2 * self.num_heads*vectors[ind][1] + model_dim * 2* vectors[ind][0]*self.num_heads*self.num_groups
                    else:

                        current_params =  model_dim * 2 * vectors[ind]*self.head_dim + model_dim * 2 *  vectors[ind]*self.head_dim*self.num_groups
                    i=i+1
                    ind +=1
                    sum_params += current_params
                    att_flag = True
                elif self.gate_type[i] == 'attn_block' and att_flag == False:

                    current_params = model_dim * 2 * self.out_dim_list[i] + model_dim * 2 * self.in_dim_list[i]
                    i=i+1
                    sum_params += current_params

                elif self.gate_type[i] == 'mlp':
                    block_mlp_in_dim = model_dim
                    block_mlp_middle_dim = vectors[ind]
                    block_mlp_out_dim = model_dim
                    current_params = block_mlp_in_dim*block_mlp_middle_dim + block_mlp_middle_dim*block_mlp_out_dim
                    i=i+1
                    ind +=1
                    sum_params += current_params
                else:
                    i=i+1
            print("current parameters: %.3f" % (sum_params / 10 ** 6))
            return sum_params
    
    def forward(self, vectors):

        att_flag = False
        sum_params = 0
        np_sum_params = 0

        i = 0
        ind = 0
        model_dim = self.model_dim
        while i < len(self.out_dim_list):
            #### for attn block ####
            if self.gate_type[i] == 'attn':

                if isinstance(vectors[ind], list): 
                    current_params = model_dim * 2 * self.num_heads*vectors[ind][1] + model_dim * 2* vectors[ind][0]*self.num_heads*self.num_groups
                else:
                    current_params =  model_dim * 2 * vectors[ind]*self.head_dim + model_dim * 2 *  vectors[ind]*self.head_dim*self.num_groups
        
                ind +=1
                i=i+1
                sum_params += current_params
                att_flag = True
            elif self.gate_type[i] == 'attn_block' and att_flag == False:
                current_params = model_dim * 2 * self.out_dim_list[i] + model_dim * 2 * self.in_dim_list[i]
                i=i+1
                np_sum_params += current_params
            elif self.gate_type[i] == 'mlp':
                block_mlp_in_dim = model_dim
                block_mlp_middle_dim = vectors[ind]
                block_mlp_out_dim = model_dim
                current_params = block_mlp_in_dim*block_mlp_middle_dim + block_mlp_middle_dim*block_mlp_out_dim
                i=i+1
                ind +=1
                sum_params += current_params
                #print(i)
                #print(vectors[ind])
            else:
                i=i+1
        param_ratio = sum_params / (self.sum_ori_params - np_sum_params)
        corrected_p = (self.sum_ori_params*self.p - np_sum_params)/(self.sum_ori_params - np_sum_params)

        if param_ratio>corrected_p:

            clampled_p_ratio = torch.clamp(param_ratio, min=corrected_p)

            loss = torch.log(clampled_p_ratio/corrected_p)
        else:
            clampled_p_ratio = torch.clamp(param_ratio, max=corrected_p)

            loss = torch.log(corrected_p/clampled_p_ratio)

        return self.lam * loss

def is_numeric(var):
    return isinstance(var, (int, float, complex))

class help_functions_hn(nn.Module):
    def __init__(self, structures, load_balance_alpha=1, num_experts=8, beta=2.0, attn_alpha=2.0, attn_max_reg_loss=False,):
        self.structures = structures
        
        self.num_moe_layers = len(structures)
        self.router_logits_stat = torch.zeros(self.num_moe_layers, 3)
        self.num_experts = num_experts
        self.load_balance_alpha = load_balance_alpha
        self.num_evaluate_batch = 0
        self.beta = beta

        self.dynamic_head_list = [[0,0] for i in range(int(self.num_moe_layers/2))]
        self.width_list = []
        for i in range(1, self.num_moe_layers + 1):
            if i % 2 == 1:      # odd: 1,3,5,...
                self.width_list.append([0, 0])
            else:               # even: 2,4,6,...
                self.width_list.append(0)
        self.attn_max_reg_loss = attn_max_reg_loss
        self.attn_alpha = attn_alpha

        self.att_topk_list = [0]*int(self.num_moe_layers/2)
    def print_info(self,vectors):
        print(self.structures)
        config = []
        for i in range(len(vectors)):
            config.append(vectors[i].sum().item())

        print(config)
    
    def prepare_for_eval_topk(self,module_list, vectors):
        for i in range(len(module_list)):
            module_list[i].prepare_experts(vectors[i])

    def prepare_for_eval(self, module_list, vectors, non_uniform=False, return_vector_union=False):
        #ind = 0
        width_list = []
        witdh_cover_list = []
        width_union_list = []
        for i in range(len(module_list)):
            #print(vectors[i].size())
            width, witdh_cover = module_list[i].prepare_experts(vectors[i], non_uniform=non_uniform)
            if hasattr(width, 'item'):
                width_list.append(int(width.item()))
            else:
                width_list.append(width)
            #if isinstance(witdh_cover, list):
            if is_numeric(witdh_cover):
                witdh_cover_list.append(witdh_cover)
            else:
               witdh_cover_list.append(int(witdh_cover.sum().item()))
            if return_vector_union:
                width_union_list.append(witdh_cover)
        print(witdh_cover_list)
        #prepare_experts
        print(width_list)
        if return_vector_union:
            return width_list, width_union_list
        else:
            return width_list
    def set_expert_modules(self, model, module_list):
        modules = list(model.modules())
        ind = 0
        #virtual_dynamic_operation
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if type(m).__name__ == 'virtual_dynamic_operation':
                m.set_router_module(module_list[ind])
                ind+=1
                #print(m.router)

    def set_gate_vectors(self, model, vectors):
        modules = list(model.modules())
        ind = 0
        #(vectors.size())
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            #virtual_block_basic_operation
            if type(m).__name__ == 'virtual_dynamic_operation':
                m.set_rnn_state(vectors[ind])
                ind+=1

    def set_gate_status(self, model, use_gate=False):
        modules = list(model.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]

            if hasattr(m, 'use_gate'):
                m.use_gate = use_gate
                #print(m.use_gate)
            if hasattr(m, 'use_att_gate'):
                m.use_att_gate = use_gate
                #print(m.use_att_gate)
            

    def get_hard_out(self, model):
        width_list = []
        modules = list(model.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if type(m).__name__ == 'single_experts_module':
                width_list.append(m.binary_approx_x.sum(-1).mean())
        return width_list

    def get_attn_hard_out(self,model, width_list, ref_width_list=None, qk_static=True, eval_flag=False):
        # width_list = []
        index = 0
        if ref_width_list is not None:
            attn_width_list = ref_width_list
        else:
            attn_width_list = []
            modules = list(model.modules())
            for layer_id in range(len(modules)):
                m = modules[layer_id]
                if type(m).__name__ == 'single_experts_module':
                    current_list = []
#                    if m.attn_flag and not m.static_flag:
                    if m.attn_flag:
                        #attn_width_list.append(m.binary_approx_x.sum(-1).max())
                        current_list.append(m.binary_approx_x[...,:m.mlp_dim].sum(-1).max())
                        if m.binary_approx_x.size(-1) - m.mlp_dim < m.head_dim:
                            binary_head_dim = m.binary_approx_x[...,m.mlp_dim:].repeat(1,1,2).view(-1, m.num_kv_heads,m.head_dim)
                        else:
                            binary_head_dim = m.binary_approx_x[...,m.mlp_dim:].view(-1, m.num_kv_heads,m.head_dim)
                        #current_list.append(m.binary_approx_x[:,m.mlp_dim:].sum(-1).max())
                        current_list.append(binary_head_dim.sum(-1).max())
                        attn_width_list.append(current_list)
        if len(attn_width_list) == 0:
            return width_list
        for i in range(len(width_list)):
            if eval_flag:
#            if len(width_list[i])>1:
                if qk_static:
        #            print(width_list[i])
                    if isinstance(width_list[i],List):
                        width_list[i][0] = attn_width_list[index][0]
                        index+=1
            else:
                if width_list[i] == None or width_list[i] == 0:
                    width_list[i] = attn_width_list[index]
                    index+=1
        return width_list

    def get_self_entropy_loss(self, model):
        sum_loss = 0
        num_attn_layers = 0
        modules = list(model.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if type(m).__name__ == 'single_experts_module':
                if hasattr(m, 'soft_approx_x'):
                    sum_loss += entropy_loss(m.soft_approx_x)
                    num_attn_layers+=1
        return sum_loss/num_attn_layers
    
    def get_attn_reg_loss(self, model, iter_num=0, targets=None, num_heads=32):
        decay_factor = 0.999
        window_iters = 1000
        sum_loss = 0
        num_attn_layers = 0
        modules = list(model.modules())

        if targets!=None:
            truncated_targets = [item for item in targets if item <= num_heads]
        for layer_id in range(len(modules)):
            m = modules[layer_id]

            if type(m).__name__ == 'single_experts_module':
                if m.attn_flag:
                    if targets!=None:
                        current_target = truncated_targets[num_attn_layers]
                        binary_app = m.binary_approx_x[:,:m.mlp_dim]
                        ratio_tensor = torch.Tensor([int(current_target.item())/binary_app.size(-1)]).to(binary_app.get_device())
                        sum_loss+=minmax_reg_loss(binary_app.sum(-1)/binary_app.size(-1), ratio_tensor,c=0.01)
                    num_attn_layers+=1
        sum_loss=sum_loss/num_attn_layers
        return self.attn_alpha * sum_loss

    
    def pair_attn_loss(self, model):
        sum_loss = 0
        num_moe_layers = 0
        modules = list(model.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if type(m).__name__ == 'single_experts_module':
                #if m.attn_flag and not m.static_flag:
                if m.attn_flag:
                    if m.qk_static_flag:
                        union_of_experts = experts_union(m.binary_approx_x[:,:m.head_dim])
                    else:
                        union_of_experts = experts_union(m.binary_approx_x)
                    target = torch.ones(union_of_experts.shape, dtype=union_of_experts.dtype,device=union_of_experts.device)
                    #pair_loss = minmax_reg_loss(union_of_experts.mean(), torch.scalar_tensor(1).to(union_of_experts.get_device()).float(), c=0.001)
                    pair_loss = minmax_reg_loss(union_of_experts, target.float(), c=0.001)
                    pair_loss = custom_grad_weight.apply(pair_loss,0.2)
                    sum_loss+=pair_loss
                num_moe_layers+=1
            if type(m).__name__ == 'single_topk_experts_module':
                union_of_experts = experts_union(m.binary_approx_x)
                pair_loss = minmax_reg_loss(union_of_experts.mean(), torch.scalar_tensor(1).to(union_of_experts.get_device()).float(), c=0.001)
                sum_loss+=pair_loss
                num_moe_layers+=1
        return self.beta * sum_loss/num_moe_layers


    def load_balance_loss(self, model):
        num_moe_layers = 0
        sum_loss = 0
        modules = list(model.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if type(m).__name__ == 'virtual_dynamic_operation':

                if hasattr(m,'router_logits') and (m.router_logits != None):
                    #print(m.router_logits)
                    if m.router_logits.size(-1) == m.router.experts:
                        sum_loss = sum_loss + m.router_logits_balance_loss()
                        num_moe_layers+=1
        return self.load_balance_alpha * sum_loss/num_moe_layers
    
    def set_qk_gate(self, model, flag=True):
        modules = list(model.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if type(m).__name__ == 'PhiSdpaAttention' or type(m).__name__ == 'PhiFlashAttention2' or type(m).__name__ == 'PhiAttention':
                m.apply_qk_gate = flag
            if type(m).__name__ == 'LlamaSdpaAttention' or type(m).__name__ == 'LlamaFlashAttention2' or type(m).__name__ == 'LlamaAttention':
                m.apply_qk_gate = flag
    
    def set_qk_hyperparameters(self, model, qk_sample_rate = 0.6, grad_w=2.0, pv_detach_flag=False, block_dropout=0.1):
        modules = list(model.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if type(m).__name__ == 'PhiSdpaAttention' or type(m).__name__ == 'PhiFlashAttention2' or type(m).__name__ == 'PhiAttention':
                m.qk_sample_rate = qk_sample_rate
                m.grad_w = grad_w
                m.pv_detach_flag = pv_detach_flag
            if type(m).__name__ == 'LlamaSdpaAttention' or type(m).__name__ == 'LlamaFlashAttention2' or type(m).__name__ == 'LlamaAttention':
                m.qk_sample_rate = qk_sample_rate
                m.grad_w = grad_w
                m.pv_detach_flag = pv_detach_flag
            if type(m).__name__ == 'LlamaDecoderLayer':
                m.resid_dropout.p = block_dropout

    def assign_width(self, model, width_list, num_heads=32):
        modules = list(model.modules())
        att_head_idx = 0
        att_list = [item for item in width_list if item <= num_heads]
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if type(m).__name__ == 'single_experts_module':
                if m.attn_flag:
                    m.width = att_list[att_head_idx].item()
                    att_head_idx+=1

    def accumlate_router_logits(self, model, attn_accumlate=True):
        modules = list(model.modules())
        num_moe_layers_idx = 0
        att_head_idx = 0
        # self.dynamic_width
        # dynamic_head_list = []
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if type(m).__name__ == 'single_experts_module':
                if m.attn_flag:
                    self.dynamic_head_list[att_head_idx][0] += m.dynamic_width[0]
                    self.dynamic_head_list[att_head_idx][1] += m.dynamic_width[1]
                    if att_head_idx == 0 and attn_accumlate:
                        self.num_evaluate_batch+=1
                    att_head_idx+=1
            if type(m).__name__ == 'virtual_dynamic_operation':

                seq_logits = m.router_logits.max(-1)[0].squeeze().to(self.router_logits_stat.dtype).cpu()

                
                self.router_logits_stat[num_moe_layers_idx,0] += seq_logits.max()
                self.router_logits_stat[num_moe_layers_idx,1] += seq_logits.min()
                self.router_logits_stat[num_moe_layers_idx,2] += seq_logits.mean()
                if num_moe_layers_idx == 0:
                    #num_tokens = m.router_logits.size(1)
                    
                    # self.num_evaluate_tokens += num_tokens
                    self.num_evaluate_batch+=1
                num_moe_layers_idx += 1 
    
    def get_router_logits(self,):
        return self.router_logits_stat/self.num_evaluate_batch
    def get_dynamic_head_list(self, ):
        return [[self.dynamic_head_list[ind][0]/self.num_evaluate_batch, self.dynamic_head_list[ind][1]/self.num_evaluate_batch] for ind in range(len(self.dynamic_head_list))]
    def get_dynamic_width_list(self, ):
        final_list = self.width_list
        for i in range(len(final_list)):
            if isinstance(final_list[i], list):
                # element is a list like [a, b]
                final_list[i] = [x / self.num_evaluate_batch for x in final_list[i]]
            else:
                # element is a scalar
                final_list[i] = final_list[i] / self.num_evaluate_batch
        return final_list