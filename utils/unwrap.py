import torch.nn as nn

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (:obj:`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    if hasattr(model, "_orig_mod"):
        return unwrap_model(model._orig_mod)
    elif hasattr(model, "_fsdp_wrapped_module"):
        return unwrap_model(model._fsdp_wrapped_module)
    else:
        return model
