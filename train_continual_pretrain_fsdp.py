"""
Continual pretraining with FSDP FULL_SHARD on pre-tokenized packed shards.

Example:
torchrun --nproc_per_node=8 train_continual_pretrain_fsdp.py \
  --model_name_or_path /path/to/tomoe_or_llama_model \
  --data_dirs /path/to/pretokenized_subset1 /path/to/pretokenized_subset2 \
  --output_dir /path/to/continual_pretrain_output \
  --seq_len 8192 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --weight_decay 0.1 \
  --warmup_steps 1000 \
  --max_steps 20000 \
  --max_train_tokens 25B \
  --bf16 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
  --logging_steps 10 \
  --save_steps 1000
"""

import argparse
import datetime
import glob
import math
import os
import shutil
import time
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
from torch import autocast
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import DistributedEnv, unwrap_model


def parse_args():
    parser = argparse.ArgumentParser(description="FSDP continual pretraining on packed token shards.")

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)

    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--file_pattern", type=str, default="*.npy")
    parser.add_argument("--data_dtype", type=str, default="uint32", choices=["uint16", "uint32", "int32", "int64"])
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--max_train_tokens",
        type=str,
        default=None,
        help="Stop after this many trained tokens, e.g. 25B, 500M, or 1000000000.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true")

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", type=str, default="LlamaDecoderLayer")

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=20000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def parse_token_count(value):
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", "")
    multipliers = {
        "k": 1_000,
        "m": 1_000_000,
        "b": 1_000_000_000,
        "t": 1_000_000_000_000,
    }
    suffix = text[-1]
    if suffix in multipliers:
        return int(float(text[:-1]) * multipliers[suffix])
    return int(float(text))


def tokens_per_optimizer_step(args, env):
    return (
        args.gradient_accumulation_steps
        * env.world_size
        * args.per_device_train_batch_size
        * args.seq_len
    )


def infer_effective_max_steps(args, env, max_train_tokens):
    token_limited_steps = None
    if max_train_tokens is not None:
        token_limited_steps = math.ceil(max_train_tokens / tokens_per_optimizer_step(args, env))

    if args.max_steps is not None and token_limited_steps is not None:
        return min(args.max_steps, token_limited_steps)
    if args.max_steps is not None:
        return args.max_steps
    return token_limited_steps


def setup_distributed():
    env = DistributedEnv()
    dist.init_process_group(
        backend="nccl",
        rank=env.global_rank,
        world_size=env.world_size,
        timeout=datetime.timedelta(seconds=3600 * 5),
    )
    torch.cuda.set_device(env.local_rank)
    return env


def dtype_from_name(name):
    import numpy as np

    return {
        "uint16": np.uint16,
        "uint32": np.uint32,
        "int32": np.int32,
        "int64": np.int64,
    }[name]


def build_dataset(args):
    from data.dataloader_packed import PackedTokenDataset

    datasets = [
        PackedTokenDataset(
            data_dir=data_dir,
            seq_len=args.seq_len,
            file_pattern=args.file_pattern,
            dtype=dtype_from_name(args.data_dtype),
            shuffle_shards=True,
        )
        for data_dir in args.data_dirs
    ]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def build_dataloader(dataset, args, env):
    sampler = DistributedSampler(
        dataset,
        num_replicas=env.world_size,
        rank=env.global_rank,
        shuffle=True,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    return loader, sampler


def validate_model_name_or_path(path):
    if path is None or str(path).strip() == "" or str(path).startswith("/path/to/"):
        raise ValueError(
            "model_name_or_path is still a placeholder. "
            "Pass a real local model directory or HuggingFace repo id."
        )
    if os.path.isabs(path) and not os.path.isdir(path):
        raise FileNotFoundError(
            f"model_name_or_path is an absolute path but does not exist: {path}"
        )


def build_model(args, env):
    load_path = args.resume_from_checkpoint or args.model_name_or_path
    validate_model_name_or_path(load_path)
    dtype = torch.bfloat16 if args.bf16 else None
    env.print_master(f"Loading model from: {load_path}")
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def resolve_transformer_layer_cls(model, class_name):
    matches = {type(module) for module in model.modules() if type(module).__name__ == class_name}
    if not matches:
        available = sorted({type(module).__name__ for module in model.modules() if "DecoderLayer" in type(module).__name__})
        raise ValueError(
            f"Could not find transformer layer class '{class_name}' in model. "
            f"Decoder-like classes found: {available}"
        )
    if len(matches) > 1:
        names = sorted(cls.__module__ + "." + cls.__name__ for cls in matches)
        raise ValueError(f"Found multiple classes named '{class_name}': {names}")
    return next(iter(matches))


def wrap_fsdp(model, args, env):
    model.to(env.local_rank)
    layer_cls = resolve_transformer_layer_cls(model, args.fsdp_transformer_layer_cls_to_wrap)
    auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})
    mixed_precision = None
    if args.bf16:
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )


def build_optimizer(model, args, env):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if param.ndim < 2 or lname.endswith("bias") or "norm" in lname or "ln" in lname:
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer_cls = torch.optim.AdamW
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb

            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            env.print_master("[warning] bitsandbytes is not installed; falling back to torch.optim.AdamW.")

    return optimizer_cls(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )


def build_scheduler(optimizer, args):
    if args.effective_max_steps is None:
        return None

    def lr_lambda(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return max(1e-8, float(step + 1) / float(args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, args.effective_max_steps - args.warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def maybe_load_tokenizer(args, env):
    tokenizer_path = args.tokenizer_name_or_path or args.model_name_or_path
    try:
        validate_model_name_or_path(tokenizer_path)
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as exc:
        env.print_master(f"[warning] Could not load tokenizer from {tokenizer_path}: {exc}")
        return None


def copy_custom_code_files(args, save_dir):
    source_dirs = [args.model_name_or_path]
    if args.resume_from_checkpoint:
        source_dirs.append(args.resume_from_checkpoint)
    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            continue
        for path in glob.glob(os.path.join(source_dir, "*.py")):
            shutil.copy(path, os.path.join(save_dir, os.path.basename(path)))


def save_checkpoint(model, tokenizer, optimizer, scheduler, args, env, tag, global_step, epoch, consumed_tokens):
    save_dir = os.path.join(args.output_dir, tag)
    if env.global_rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        model_state = model.state_dict()

    try:
        optim_state = FSDP.optim_state_dict(model, optimizer)
    except Exception:
        optim_state = optimizer.state_dict()

    if env.global_rank == 0:
        base = unwrap_model(model)
        base.save_pretrained(save_dir, state_dict=model_state, safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)
        copy_custom_code_files(args, save_dir)

        training_state = {
            "optimizer": optim_state,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "global_step": global_step,
            "epoch": epoch,
            "consumed_tokens": consumed_tokens,
        }
        torch.save(training_state, os.path.join(save_dir, "training_state.pt"))

    dist.barrier()


def load_training_state(model, optimizer, scheduler, checkpoint_dir, env):
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if not os.path.exists(state_path):
        env.print_master(f"[warning] No training_state.pt found in {checkpoint_dir}; model weights were loaded only.")
        return 0, 0, 0

    state = torch.load(state_path, map_location="cpu")
    if "optimizer" in state and state["optimizer"] is not None:
        try:
            optim_state = FSDP.optim_state_dict_to_load(model, optimizer, state["optimizer"])
        except Exception:
            optim_state = state["optimizer"]
        optimizer.load_state_dict(optim_state)

    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])

    global_step = int(state.get("global_step", 0))
    epoch = int(state.get("epoch", 0))
    consumed_tokens = int(state.get("consumed_tokens", 0))
    env.print_master(f"Resumed training state from {checkpoint_dir} at global_step={global_step}.")
    return global_step, epoch, consumed_tokens


def consumed_tokens_for_step(global_step, args, env):
    return (
        global_step
        * args.gradient_accumulation_steps
        * env.world_size
        * args.per_device_train_batch_size
        * args.seq_len
    )


def train(args):
    env = setup_distributed()
    torch.manual_seed(args.seed + env.global_rank)
    max_train_tokens = parse_token_count(args.max_train_tokens)
    args.effective_max_steps = infer_effective_max_steps(args, env, max_train_tokens)
    if env.global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.effective_max_steps is not None:
        env.print_master(f"Effective max optimizer steps for scheduler: {args.effective_max_steps}")

    tokenizer = maybe_load_tokenizer(args, env)
    dataset = build_dataset(args)
    dataloader, sampler = build_dataloader(dataset, args, env)

    model = build_model(args, env)
    model = wrap_fsdp(model, args, env)
    optimizer = build_optimizer(model, args, env)
    scheduler = build_scheduler(optimizer, args)

    global_step = 0
    start_epoch = 0
    consumed_tokens = 0
    if args.resume_from_checkpoint:
        global_step, start_epoch, consumed_tokens = load_training_state(
            model,
            optimizer,
            scheduler,
            args.resume_from_checkpoint,
            env,
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    log_tic = time.time()
    last_log_tokens = consumed_tokens
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    has_step_limit = args.effective_max_steps is not None
    total_epochs = args.num_train_epochs if not has_step_limit else 10**9
    for epoch in range(start_epoch, total_epochs):
        sampler.set_epoch(epoch)
        for micro_step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(env.local_rank, non_blocking=True)[:, : args.seq_len]
            # HuggingFace CausalLM heads shift labels internally. PackedTokenDataset
            # also exposes shifted labels for manual-CE training, so clone inputs here
            # to keep the standard outputs.loss objective correct.
            labels = input_ids.clone()

            accumulation_index = micro_step % args.gradient_accumulation_steps
            should_sync = accumulation_index == args.gradient_accumulation_steps - 1
            sync_context = nullcontext() if should_sync else model.no_sync()

            with sync_context:
                with autocast(device_type="cuda", dtype=dtype, enabled=args.bf16):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()

            running_loss += float(loss.detach().item()) * args.gradient_accumulation_steps

            if not should_sync:
                continue

            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                model.clip_grad_norm_(args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            consumed_tokens = consumed_tokens_for_step(global_step, args, env)

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                elapsed = max(time.time() - log_tic, 1e-6)
                token_delta = consumed_tokens - last_log_tokens
                tokens_per_sec = token_delta / elapsed
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = running_loss / args.logging_steps
                env.print_master(
                    f"global_step={global_step} loss={avg_loss:.4f} lr={lr:.3e} "
                    f"tokens/sec={tokens_per_sec:.2f} consumed_tokens={consumed_tokens}"
                )
                running_loss = 0.0
                log_tic = time.time()
                last_log_tokens = consumed_tokens

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    args,
                    env,
                    tag=f"checkpoint-{global_step}",
                    global_step=global_step,
                    epoch=epoch,
                    consumed_tokens=consumed_tokens,
                )

            reached_max_steps = args.max_steps is not None and global_step >= args.max_steps
            reached_max_tokens = max_train_tokens is not None and consumed_tokens >= max_train_tokens
            if reached_max_steps or reached_max_tokens:
                save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    args,
                    env,
                    tag="final",
                    global_step=global_step,
                    epoch=epoch,
                    consumed_tokens=consumed_tokens,
                )
                dist.destroy_process_group()
                return

    save_checkpoint(
        model,
        tokenizer,
        optimizer,
        scheduler,
        args,
        env,
        tag="final",
        global_step=global_step,
        epoch=total_epochs - 1,
        consumed_tokens=consumed_tokens,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    train(parse_args())
