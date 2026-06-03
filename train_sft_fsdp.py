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
    load_source_config,
    load_training_state,
    maybe_compile_model,
    maybe_load_tokenizer,
    save_checkpoint,
    setup_distributed,
    validate_loaded_config,
    validate_output_dir,
    wrap_fsdp,
)


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


def dummy_sft_example(tokenizer):
    token_id = tokenizer.pad_token_id
    if token_id is None:
        token_id = tokenizer.eos_token_id
    if token_id is None:
        token_id = 0
    return {
        "input_ids": [int(token_id)],
        "attention_mask": [0],
        "labels": [-100],
        "valid_label_tokens": 0,
    }


def assistant_labels_with_prefix_spans(tokenizer, messages, max_length):
    input_ids = apply_chat_ids(tokenizer, messages, max_length=max_length, truncation=True)
    labels = [-100] * len(input_ids)
    for idx, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        # The fallback Tulu template does not contain {% generation %} blocks, so
        # assistant-only labels are built from explicit rendered prefix spans
        # rather than return_assistant_tokens_mask=True.
        start_ids = apply_chat_ids(
            tokenizer,
            messages[:idx],
            max_length=None,
            truncation=False,
            add_generation_prompt=True,
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
        return dummy_sft_example(tokenizer)

    input_ids, labels = assistant_labels_with_prefix_spans(tokenizer, messages, max_length)

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    valid_label_tokens = sum(1 for label in labels if label != -100)
    if valid_label_tokens <= 0:
        return dummy_sft_example(tokenizer)
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


def distributed_sum_scalar(value, device):
    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def distributed_reduce_loss_sums(loss_sums, loss_count, device):
    stats = torch.tensor(
        [
            loss_sums["total"],
            loss_sums["lm"],
            loss_sums["balance"],
            float(loss_count),
        ],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return {
        "sums": {
            "total": float(stats[0].item()),
            "lm": float(stats[1].item()),
            "balance": float(stats[2].item()),
        },
        "count": float(stats[3].item()),
    }


def distributed_dataloader_status(has_batch, failed, device):
    status = -1.0 if failed else (1.0 if has_batch else 0.0)
    t = torch.tensor(status, device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
    global_status = float(t.item())
    any_failed = global_status < 0.0
    all_have_batch = global_status > 0.5
    return all_have_batch, any_failed


def load_balancing_loss(router_logits, attention_mask):
    if router_logits is None:
        return None
    if torch.is_tensor(router_logits):
        router_logits = (router_logits,)

    token_mask = attention_mask.reshape(-1).bool()
    losses = []
    for layer_router in router_logits:
        if layer_router is None:
            continue
        router_probs = layer_router.float().reshape(-1, layer_router.shape[-1])
        if router_probs.numel() == 0:
            continue
        if token_mask.shape[0] == router_probs.shape[0]:
            router_probs = router_probs[token_mask]
        if router_probs.numel() == 0:
            continue
        if router_probs.min() < 0 or router_probs.max() > 1.0:
            router_probs = torch.softmax(router_probs, dim=-1)
        num_experts = router_probs.shape[-1]
        selected_experts = torch.argmax(router_probs, dim=-1)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).float()
        tokens_per_expert = expert_mask.mean(dim=0)
        router_prob_per_expert = router_probs.mean(dim=0)
        losses.append(num_experts * torch.sum(tokens_per_expert * router_prob_per_expert))
    if not losses:
        return None
    return torch.stack(losses).mean()


def train(args):
    env = setup_distributed()
    torch.manual_seed(args.seed + env.global_rank)
    validate_output_dir(args)
    source_config_path, source_config = load_source_config(args)
    if env.global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = maybe_load_tokenizer(args, env)
    if tokenizer is None:
        raise RuntimeError("SFT requires a tokenizer.")
    ensure_tokenizer_ready(tokenizer, args)

    dataset = build_sft_dataset(args, tokenizer, env)
    dataloader = build_dataloader(dataset, tokenizer, args, env)

    model = build_model(args, env)
    validate_loaded_config(model, source_config, env)

    model = wrap_fsdp(model, args, env)
    model = maybe_compile_model(model, args, env)

    args.effective_max_steps = infer_max_steps(args, dataloader, env)

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
    accum_loss_sums = {"total": 0.0, "lm": 0.0, "balance": 0.0}
    accum_loss_count = 0
    accum_real_tokens = 0.0
    optimizer_step_tic = time.time()

    for epoch in range(start_epoch, args.num_train_epochs):
        dataloader_iter = iter(dataloader)
        micro_step = 0
        while True:
            dataloader_error = None
            try:
                batch = next(dataloader_iter)
                has_batch = True
            except StopIteration:
                batch = None
                has_batch = False
            except Exception as exc:
                batch = None
                has_batch = False
                dataloader_error = exc

            reduction_device = torch.device("cuda", env.local_rank)
            all_have_batch, any_dataloader_failed = distributed_dataloader_status(
                has_batch,
                failed=dataloader_error is not None,
                device=reduction_device,
            )
            if any_dataloader_failed:
                if dataloader_error is not None:
                    raise dataloader_error
                raise RuntimeError(
                    f"A peer rank failed while reading the dataloader at epoch={epoch} "
                    f"micro_step={micro_step}."
                )
            if not all_have_batch:
                break

            input_ids = batch["input_ids"].to(env.local_rank, non_blocking=True)
            attention_mask = batch["attention_mask"].to(env.local_rank, non_blocking=True)
            labels = batch["labels"].to(env.local_rank, non_blocking=True)

            valid_label_tokens = int((labels != -100).sum().item())
            total_tokens = int(attention_mask.sum().item())

            accumulation_index = micro_step % args.gradient_accumulation_steps
            should_sync = accumulation_index == args.gradient_accumulation_steps - 1
            micro_step += 1
            sync_context = nullcontext() if should_sync else model.no_sync()
            has_loss_tokens = valid_label_tokens > 0
            model_attention_mask = attention_mask
            if total_tokens == 0:
                model_attention_mask = torch.ones_like(attention_mask)

            with sync_context:
                with autocast(device_type="cuda", dtype=dtype, enabled=args.bf16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=model_attention_mask,
                        labels=labels if has_loss_tokens else None,
                        output_router_logits=args.moe_aux_loss_weight > 0,
                    )
                    if has_loss_tokens:
                        lm_loss = outputs.loss
                        balance_loss = load_balancing_loss(
                            getattr(outputs, "router_logits", None),
                            attention_mask=attention_mask,
                        )
                        if balance_loss is None:
                            balance_loss = lm_loss.new_zeros(())
                        total_loss = lm_loss + args.moe_aux_loss_weight * balance_loss
                    else:
                        zero_loss = outputs.logits.float().sum() * 0.0
                        router_logits = getattr(outputs, "router_logits", None)
                        if router_logits is not None:
                            if torch.is_tensor(router_logits):
                                router_logits = (router_logits,)
                            for layer_router in router_logits:
                                if layer_router is not None:
                                    zero_loss = zero_loss + layer_router.float().sum() * 0.0
                        lm_loss = zero_loss
                        balance_loss = zero_loss
                        total_loss = zero_loss
                    backward_loss = total_loss / args.gradient_accumulation_steps
                    raw_lm_loss = lm_loss.detach()
                    raw_balance_loss = balance_loss.detach()
                    raw_total_loss = total_loss.detach()
                backward_loss.backward()

            accum_loss_sums["total"] += float(raw_total_loss.item())
            accum_loss_sums["lm"] += float(raw_lm_loss.item())
            accum_loss_sums["balance"] += float(raw_balance_loss.item())
            accum_loss_count += 1
            accum_real_tokens += total_tokens

            if not should_sync:
                continue

            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                model.clip_grad_norm_(args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            global_real_tokens = distributed_sum_scalar(
                accum_real_tokens,
                device=reduction_device,
            )
            global_loss_stats = distributed_reduce_loss_sums(
                accum_loss_sums,
                accum_loss_count,
                device=reduction_device,
            )
            consumed_tokens += int(global_real_tokens)
            running_loss_sums["total"] += global_loss_stats["sums"]["total"]
            running_loss_sums["lm"] += global_loss_stats["sums"]["lm"]
            running_loss_sums["balance"] += global_loss_stats["sums"]["balance"]
            running_loss_count += global_loss_stats["count"]
            accum_loss_sums = {"total": 0.0, "lm": 0.0, "balance": 0.0}
            accum_loss_count = 0
            accum_real_tokens = 0.0
            running_step_time += time.time() - optimizer_step_tic
            optimizer_step_tic = time.time()

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                avg_loss_count = max(float(running_loss_count), 1.0)
                avg_losses = {
                    "total": running_loss_sums["total"] / avg_loss_count,
                    "lm": running_loss_sums["lm"] / avg_loss_count,
                    "balance": running_loss_sums["balance"] / avg_loss_count,
                }
                env.print_master(
                    f"global_step={global_step} loss={avg_losses['total']:.4f} "
                    f"lm_loss={avg_losses['lm']:.4f} load_balance_loss={avg_losses['balance']:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"step_time={running_step_time / args.logging_steps:.3f}s"
                )
                running_loss_sums = {"total": 0.0, "lm": 0.0, "balance": 0.0}
                running_loss_count = 0
                running_step_time = 0.0

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
