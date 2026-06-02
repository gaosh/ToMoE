"""
Stage-2 supervised fine-tuning with FSDP on Tulu-style chat data.

This script reuses the continual pretraining FSDP/model/checkpoint helpers, but
changes the data and loss semantics:
- CPT trains on every token in packed pretraining blocks.
- SFT trains only assistant response tokens; system/user/template tokens use
  label -100 and are ignored by the standard HF causal LM loss.
- Tulu-3 SFT data is loaded as a stream and tokenized lazily.
"""

import argparse
import inspect
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from datasets import load_dataset, load_dataset_builder
try:
    from datasets.distributed import split_dataset_by_node
except Exception:
    split_dataset_by_node = None
from torch import autocast
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from train_continual_pretrain_fsdp import (
    build_model,
    build_optimizer,
    load_balancing_loss,
    load_source_config,
    load_training_state,
    log_elapsed,
    maybe_compile_model,
    maybe_load_tokenizer,
    reduce_loss_stats,
    save_checkpoint,
    setup_distributed,
    validate_loaded_config,
    validate_output_dir,
    wrap_fsdp,
)
from utils import unwrap_model


TULU_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|system|>\\n' + message['content'] + '\\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user|>\\n' + message['content'] + '\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|assistant|>\\n' + message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|assistant|>\\n' }}"
    "{% endif %}"
)


def parse_args():
    parser = argparse.ArgumentParser(description="FSDP SFT on Tulu-3-style messages data.")

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--chat_template_name", type=str, default="auto", choices=["auto", "tulu", "none"])

    parser.add_argument("--dataset_name", type=str, default="allenai/tulu-3-sft-mixture")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--moe_aux_loss_weight", type=float, default=0.01)
    parser.add_argument("--tomoe_moe_impl", type=str, default="naive", choices=["naive", "grouped_gemm"])

    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--load_model_on_gpu", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="default")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager", "auto", "none"],
    )
    parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", type=str, default="LlamaDecoderLayer")

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_shard_size", type=str, default="5GB")
    parser.add_argument("--save_optimizer", action="store_true", default=False)
    parser.add_argument("--save_optimizer_latest_only", action="store_true", default=False)
    parser.add_argument("--save_at_iter0", action="store_true", default=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def ensure_tokenizer_ready(tokenizer, args):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.chat_template_name == "tulu":
        tokenizer.chat_template = TULU_CHAT_TEMPLATE
    elif args.chat_template_name == "auto":
        if tokenizer.chat_template is None:
            default_template = getattr(tokenizer, "default_chat_template", None)
            tokenizer.chat_template = default_template or TULU_CHAT_TEMPLATE
    elif tokenizer.chat_template is None:
        raise ValueError("Tokenizer does not define chat_template; SFT requires tokenizer.apply_chat_template.")


def normalize_messages(messages):
    if not isinstance(messages, list) or not messages:
        return None
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if role is None or content is None:
            return None
        normalized.append({"role": str(role), "content": str(content)})
    return normalized


def apply_chat_ids(tokenizer, messages, max_length=None, truncation=False, add_generation_prompt=False):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        truncation=truncation,
        max_length=max_length,
    )


def assistant_labels_with_mask(tokenizer, messages, max_length):
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=max_length,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = rendered["input_ids"]
    assistant_mask = rendered.get("assistant_masks") or rendered.get("assistant_tokens_mask")
    if assistant_mask is None:
        return None
    if sum(int(mask) for mask in assistant_mask) == 0 and any(message["role"] == "assistant" for message in messages):
        return None
    labels = [token_id if int(mask) == 1 else -100 for token_id, mask in zip(input_ids, assistant_mask)]
    return input_ids, labels


def assistant_labels_with_prefix_spans(tokenizer, messages, max_length):
    input_ids = apply_chat_ids(tokenizer, messages, max_length=max_length, truncation=True)
    labels = [-100] * len(input_ids)
    for idx, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        try:
            start_ids = apply_chat_ids(
                tokenizer,
                messages[:idx],
                max_length=None,
                truncation=False,
                add_generation_prompt=True,
            )
        except Exception:
            start_ids = apply_chat_ids(
                tokenizer,
                messages[:idx] + [{"role": "assistant", "content": ""}],
                max_length=None,
                truncation=False,
                add_generation_prompt=False,
            )
        end_ids = apply_chat_ids(tokenizer, messages[: idx + 1], max_length=None, truncation=False)
        start = min(len(start_ids), len(input_ids))
        end = min(len(end_ids), len(input_ids))
        for pos in range(start, end):
            labels[pos] = input_ids[pos]
    return input_ids, labels


def tokenize_sft_example(example, tokenizer, max_length):
    messages = normalize_messages(example.get("messages"))
    if messages is None:
        return {"input_ids": [], "attention_mask": [], "labels": [], "valid_label_tokens": 0}

    tokenized = None
    try:
        tokenized = assistant_labels_with_mask(tokenizer, messages, max_length)
    except Exception:
        tokenized = None
    if tokenized is None:
        input_ids, labels = assistant_labels_with_prefix_spans(tokenizer, messages, max_length)
    else:
        input_ids, labels = tokenized

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    valid_label_tokens = sum(1 for label in labels if label != -100)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "valid_label_tokens": valid_label_tokens,
    }


class StreamingSFTDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, args):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.args = args

    def __iter__(self):
        dataset = self.dataset
        worker_info = get_worker_info()
        if worker_info is not None:
            dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)
        yielded = 0
        for example in dataset:
            tokenized = tokenize_sft_example(example, self.tokenizer, self.args.max_seq_length)
            if tokenized["valid_label_tokens"] <= 0:
                continue
            yield tokenized
            yielded += 1
            if self.args.max_train_samples is not None and yielded >= self.args.max_train_samples:
                return


def build_sft_dataset(args, tokenizer, env):
    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        cache_dir=args.dataset_cache_dir,
        streaming=True,
    )
    if args.shuffle_buffer_size and args.shuffle_buffer_size > 0:
        dataset = dataset.shuffle(buffer_size=args.shuffle_buffer_size, seed=args.seed)
    if split_dataset_by_node is not None:
        dataset = split_dataset_by_node(dataset, rank=env.global_rank, world_size=env.world_size)
    else:
        dataset = dataset.shard(num_shards=env.world_size, index=env.global_rank)
    return StreamingSFTDataset(dataset, tokenizer, args)


class SFTDataCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def build_dataloader(dataset, tokenizer, args, env):
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=SFTDataCollator(tokenizer),
    )
    return loader


def get_streaming_split_num_examples(args, env):
    try:
        builder = load_dataset_builder(args.dataset_name, cache_dir=args.dataset_cache_dir)
        split_info = builder.info.splits.get(args.dataset_split)
        if split_info is None:
            env.print_master(f"[sft] split size unavailable for split={args.dataset_split}")
            return None
        num_examples = int(split_info.num_examples)
        if args.max_train_samples is not None:
            num_examples = min(num_examples, int(args.max_train_samples))
        return num_examples
    except Exception as exc:
        env.print_master(f"[sft] could not infer streaming split size: {exc}")
        return None


def infer_max_steps(args, dataloader, env):
    if args.max_train_steps is not None:
        return args.max_train_steps
    num_examples = get_streaming_split_num_examples(args, env)
    if num_examples is None:
        env.print_master("[sft] streaming split size unknown; using constant LR/no step limit.")
        return None
    examples_per_step = (
        env.world_size
        * args.per_device_train_batch_size
        * args.gradient_accumulation_steps
    )
    steps_per_epoch = max(1, math.ceil(num_examples / examples_per_step))
    return steps_per_epoch * args.num_train_epochs


def build_scheduler(optimizer, args):
    if args.effective_max_steps is None:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    total_steps = max(1, args.effective_max_steps)
    warmup_steps = int(math.ceil(total_steps * args.warmup_ratio))

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        if args.lr_scheduler_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def reduce_token_stats(valid_label_tokens, total_tokens, total_tokens_including_padding, device):
    stats = torch.tensor(
        [valid_label_tokens, total_tokens, total_tokens_including_padding],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return [float(item) for item in stats.tolist()]


def supports_output_router_logits(model):
    try:
        signature = inspect.signature(unwrap_model(model).forward)
    except Exception:
        return True
    return "output_router_logits" in signature.parameters


def train(args):
    env = setup_distributed()
    torch.manual_seed(args.seed + env.global_rank)
    validate_output_dir(args)
    source_config_path, source_config = load_source_config(args)
    if env.global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    tic = time.time()
    tokenizer = maybe_load_tokenizer(args, env)
    if tokenizer is None:
        raise RuntimeError("SFT requires a tokenizer.")
    ensure_tokenizer_ready(tokenizer, args)
    log_elapsed(env, "tokenizer load", tic)

    tic = time.time()
    dataset = build_sft_dataset(args, tokenizer, env)
    dataloader = build_dataloader(dataset, tokenizer, args, env)
    log_elapsed(env, "dataset/dataloader build", tic)

    tic = time.time()
    model = build_model(args, env)
    log_elapsed(env, "model from_pretrained", tic)
    validate_loaded_config(model, source_config, env)

    tic = time.time()
    model = wrap_fsdp(model, args, env)
    log_elapsed(env, "FSDP wrap", tic)
    model = maybe_compile_model(model, args, env)
    output_router_logits = supports_output_router_logits(model) and args.moe_aux_loss_weight > 0

    args.effective_max_steps = infer_max_steps(args, dataloader, env)
    env.print_master(f"Effective max SFT optimizer steps: {args.effective_max_steps}")

    tic = time.time()
    optimizer = build_optimizer(model, args, env)
    scheduler = build_scheduler(optimizer, args)
    log_elapsed(env, "optimizer/scheduler build", tic)

    global_step = 0
    start_epoch = 0
    consumed_tokens = 0
    if args.resume_from_checkpoint:
        global_step, start_epoch, consumed_tokens = load_training_state(
            model,
            optimizer,
            scheduler,
            args.resume_from_checkpoint,
            args,
            env,
        )

    if args.save_at_iter0 and global_step == 0:
        save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            env=env,
            tag="checkpoint-iter0",
            global_step=0,
            epoch=0,
            consumed_tokens=0,
            source_config_path=source_config_path,
            source_config=source_config,
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    running_loss_sums = {"total": 0.0, "lm": 0.0, "balance": 0.0}
    running_loss_count = 0
    running_step_time = 0.0
    running_valid_label_tokens = 0.0
    running_total_tokens = 0.0
    running_total_tokens_including_padding = 0.0
    step_total_tokens = 0.0
    log_tic = time.time()
    optimizer_step_tic = time.time()

    for epoch in range(start_epoch, args.num_train_epochs):
        for micro_step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(env.local_rank, non_blocking=True)
            attention_mask = batch["attention_mask"].to(env.local_rank, non_blocking=True)
            labels = batch["labels"].to(env.local_rank, non_blocking=True)

            valid_label_tokens = int((labels != -100).sum().item())
            total_tokens = int(attention_mask.sum().item())
            total_tokens_including_padding = int(input_ids.numel())

            accumulation_index = micro_step % args.gradient_accumulation_steps
            should_sync = accumulation_index == args.gradient_accumulation_steps - 1
            sync_context = nullcontext() if should_sync else model.no_sync()

            with sync_context:
                with autocast(device_type="cuda", dtype=dtype, enabled=args.bf16):
                    model_kwargs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }
                    if output_router_logits:
                        model_kwargs["output_router_logits"] = True
                    outputs = model(**model_kwargs)
                    lm_loss = outputs.loss
                    balance_loss = load_balancing_loss(getattr(outputs, "router_logits", None))
                    if balance_loss is None:
                        balance_loss = lm_loss.new_zeros(())
                    total_loss = lm_loss + args.moe_aux_loss_weight * balance_loss
                    backward_loss = total_loss / args.gradient_accumulation_steps
                    raw_lm_loss = lm_loss.detach()
                    raw_balance_loss = balance_loss.detach()
                    raw_total_loss = total_loss.detach()
                backward_loss.backward()

            running_loss_sums["total"] += float(raw_total_loss.item())
            running_loss_sums["lm"] += float(raw_lm_loss.item())
            running_loss_sums["balance"] += float(raw_balance_loss.item())
            running_loss_count += 1
            running_valid_label_tokens += valid_label_tokens
            running_total_tokens += total_tokens
            running_total_tokens_including_padding += total_tokens_including_padding
            step_total_tokens += total_tokens

            if not should_sync:
                continue

            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                model.clip_grad_norm_(args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            _, global_step_tokens, _ = reduce_token_stats(
                0,
                step_total_tokens,
                0,
                device=torch.device("cuda", env.local_rank),
            )
            consumed_tokens += int(global_step_tokens)
            step_total_tokens = 0.0
            running_step_time += time.time() - optimizer_step_tic
            optimizer_step_tic = time.time()

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                elapsed = max(time.time() - log_tic, 1e-6)
                avg_losses = reduce_loss_stats(
                    running_loss_sums,
                    running_loss_count,
                    device=torch.device("cuda", env.local_rank),
                )
                valid, total, total_with_pad = reduce_token_stats(
                    running_valid_label_tokens,
                    running_total_tokens,
                    running_total_tokens_including_padding,
                    device=torch.device("cuda", env.local_rank),
                )
                env.print_master(
                    f"global_step={global_step} loss={avg_losses['total']:.4f} "
                    f"lm_loss={avg_losses['lm']:.4f} load_balance_loss={avg_losses['balance']:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"step_time={running_step_time / args.logging_steps:.3f}s "
                    f"tokens/sec={total / elapsed:.2f} tokens/sec_including_padding={total_with_pad / elapsed:.2f} "
                    f"valid_label_tokens={int(valid)} total_tokens={int(total)} "
                    f"total_tokens_including_padding={int(total_with_pad)}"
                )
                running_loss_sums = {"total": 0.0, "lm": 0.0, "balance": 0.0}
                running_loss_count = 0
                running_step_time = 0.0
                running_valid_label_tokens = 0.0
                running_total_tokens = 0.0
                running_total_tokens_including_padding = 0.0
                log_tic = time.time()

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
                    source_config_path=source_config_path,
                    source_config=source_config,
                )

            if args.effective_max_steps is not None and global_step >= args.effective_max_steps:
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
                    source_config_path=source_config_path,
                    source_config=source_config,
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
        epoch=args.num_train_epochs - 1,
        consumed_tokens=consumed_tokens,
        source_config_path=source_config_path,
        source_config=source_config,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    train(parse_args())
