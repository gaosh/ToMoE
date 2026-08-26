"""
Example:

python eval_tomoe_gated_actual_moe_ppl.py \
  --model_name_or_path /orange/sgao1/sgao1/saved_models/tomoe_gated_actual_moe_llama3_8b \
  --datasets wikitext \
  --block_size 2048 \
  --max_tokens 524288 \
  --device cuda:0 \
  --torch_dtype bfloat16
"""

import gc
import math
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_eval_data(dataset_name: str) -> str:
    if dataset_name == "wikitext":
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(testdata["text"])
    if dataset_name == "ptb":
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test", trust_remote_code=True)
        return "\n\n".join(testdata["sentence"])
    if dataset_name == "c4":
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        return " ".join(testdata[:1100]["text"])
    raise ValueError("invalid dataset name (wikitext, ptb, c4 are allowed)")


def parameter_memory_bytes(model):
    return sum(param.numel() * param.element_size() for param in model.parameters())


def print_model_parameter_report(model, prefix="[model-parameter-count]"):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    dtype_bytes = {}
    for param in model.parameters():
        dtype_bytes[param.dtype] = dtype_bytes.get(param.dtype, 0) + param.numel() * param.element_size()
    print(prefix)
    print(f"total_params: {total_params / 1_000_000:.3f}M")
    print(f"trainable_params: {trainable_params / 1_000_000:.3f}M")
    for dtype, nbytes in sorted(dtype_bytes.items(), key=lambda item: str(item[0])):
        print(f"parameter_memory_{dtype}: {nbytes / (1024 ** 3):.3f}GiB")


def print_cuda_memory(prefix, device="cuda"):
    if not torch.cuda.is_available():
        return
    with torch.cuda.device(torch.device(device)):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"[cuda-memory] {prefix}")
        print(f"free: {free_bytes / (1024 ** 3):.3f}GiB")
        print(f"total: {total_bytes / (1024 ** 3):.3f}GiB")
        print(f"allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.3f}GiB")
        print(f"reserved: {torch.cuda.memory_reserved() / (1024 ** 3):.3f}GiB")
        print(f"max_allocated: {torch.cuda.max_memory_allocated() / (1024 ** 3):.3f}GiB")


def maybe_move_model_to_device(model, device, allow_cpu_fallback=True, cuda_margin_gib=2.0):
    if device.startswith("cuda"):
        with torch.cuda.device(torch.device(device)):
            required_bytes = parameter_memory_bytes(model) + int(cuda_margin_gib * 1024**3)
            free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < required_bytes:
            message = (
                f"[ppl] not enough free CUDA memory for model parameters. "
                f"required_at_least={(required_bytes / (1024 ** 3)):.3f}GiB, "
                f"free={(free_bytes / (1024 ** 3)):.3f}GiB, device={device}."
            )
            if allow_cpu_fallback:
                print(message)
                print("[ppl] falling back to CPU evaluation. This is slower but avoids CUDA OOM.")
                device = "cpu"
            else:
                raise RuntimeError(message)
    model.to(device)
    return device


@torch.inference_mode()
def evaluate_ppl(model, tokenizer, datasets="wikitext", block_size=2048, max_tokens=524288, device="cuda"):
    model.eval()
    for dsname in datasets.split(","):
        text = load_eval_data(dsname.strip())
        encoded_text = tokenizer.encode(text, return_tensors="pt")
        if max_tokens is not None and max_tokens > 0:
            encoded_text = encoded_text[:, :max_tokens]

        nlls = 0.0
        toks = 0
        last_logits_shape = None
        for start in range(0, encoded_text.shape[1] - 1, block_size):
            inp = encoded_text[:, start : start + block_size].to(device=device, dtype=torch.long)
            if inp.shape[1] < 2:
                continue
            output = model(inp)
            logits = output.logits if hasattr(output, "logits") else output
            nll = F.cross_entropy(logits[0, :-1], inp[0, 1:], reduction="sum")
            nlls += float(nll.item())
            toks += inp.shape[1] - 1
            last_logits_shape = tuple(logits.shape)
            del output, logits, inp

        if toks == 0:
            raise RuntimeError(f"No evaluation tokens for dataset={dsname}")
        ppl = math.exp(nlls / toks)
        print(f"[ppl] dataset={dsname} tokens={toks} logits_shape={last_logits_shape} ppl={ppl:.4f}")


def main(
    model_name_or_path: Optional[str] = None,
    tokenizer_name_or_path: str = None,
    datasets: str = "wikitext",
    block_size: int = 2048,
    max_tokens: int = 524288,
    device: str = "cuda:0",
    torch_dtype: str = "bfloat16",
    low_cpu_mem_usage: bool = True,
    load_on_cpu: bool = True,
    allow_cpu_fallback: bool = True,
    cuda_margin_gib: float = 2.0,
):
    if model_name_or_path is None:
        raise ValueError("Please pass --model_name_or_path /path/to/exported_model")

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[torch_dtype]
    tokenizer_path = tokenizer_name_or_path or model_name_or_path

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device={device}, but CUDA is not available.")

    print_cuda_memory("before loading model", device=device if device.startswith("cuda") else "cuda")
    device_map = None
    if device.startswith("cuda") and not load_on_cpu:
        device_map = {"": device}

    print(f"[ppl] loading model from {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        device_map=device_map,
    )
    print_model_parameter_report(model)
    print_cuda_memory("after loading model", device=device if device.startswith("cuda") else "cuda")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if load_on_cpu:
        device = maybe_move_model_to_device(
            model,
            device=device,
            allow_cpu_fallback=allow_cpu_fallback,
            cuda_margin_gib=cuda_margin_gib,
        )
    print_cuda_memory("before ppl evaluation", device=device if device.startswith("cuda") else "cuda")
    evaluate_ppl(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        block_size=block_size,
        max_tokens=max_tokens,
        device=device,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
