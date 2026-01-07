import torch
import torch.nn.functional as F

def softmax_fp32(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Promote to float32 for the computation
    x_float = x.float()
    # Standard stable softmax: subtract max along dim
    max_x, _ = x_float.max(dim=dim, keepdim=True)
    x_stable = x_float - max_x
    exp_x = torch.exp(x_stable)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    probs = exp_x / sum_exp
    # Cast back to original dtype (e.g., fp16/bf16)
    return probs.to(x.dtype)

def log_softmax_fp32(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Again, compute in float32
    x_float = x.float()
    # Use logsumexp for numerical stability
    logsumexp = torch.logsumexp(x_float, dim=dim, keepdim=True)
    log_probs = x_float - logsumexp
    return log_probs.to(x.dtype)