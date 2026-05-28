#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/orange/sgao1/sgao1/continual_pretrain_outputs/tomoe_gated_llama3_8b/final}"
OUTPUT_DIR="${OUTPUT_DIR:-/orange/sgao1/sgao1/sft_outputs/tomoe_gated_llama3_8b_tulu3}"
DATASET_NAME="${DATASET_NAME:-allenai/tulu-3-sft-mixture}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-1000}"

NUM_WORKERS="${NUM_WORKERS:-4}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-8}"
FSDP_LAYER_CLS="${FSDP_LAYER_CLS:-LlamaDecoderLayer}"
MOE_AUX_LOSS_WEIGHT="${MOE_AUX_LOSS_WEIGHT:-0.01}"
TOMOE_MOE_IMPL="${TOMOE_MOE_IMPL:-grouped_gemm}"
TOMOE_MOE_IMPL="$(printf '%s' "${TOMOE_MOE_IMPL}" | xargs)"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29517}"
COMPILE_MODEL="${COMPILE_MODEL:-0}"
COMPILE_MODE="${COMPILE_MODE:-default}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
LOAD_MODEL_ON_GPU="${LOAD_MODEL_ON_GPU:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
SYNC_CUSTOM_CODE="${SYNC_CUSTOM_CODE:-1}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-1}"
PACKING="${PACKING:-0}"
SAVE_OPTIMIZER="${SAVE_OPTIMIZER:-0}"
SAVE_OPTIMIZER_LATEST_ONLY="${SAVE_OPTIMIZER_LATEST_ONLY:-0}"
SAVE_AT_ITER0="${SAVE_AT_ITER0:-0}"

if [ "${TOMOE_MOE_IMPL}" != "naive" ] && [ "${TOMOE_MOE_IMPL}" != "grouped_gemm" ]; then
    echo "TOMOE_MOE_IMPL must be 'naive' or 'grouped_gemm', got: ${TOMOE_MOE_IMPL}" >&2
    exit 1
fi

if [[ "${MODEL_NAME_OR_PATH}" = /* ]] && [ ! -d "${MODEL_NAME_OR_PATH}" ]; then
    echo "MODEL_NAME_OR_PATH is an absolute path but does not exist: ${MODEL_NAME_OR_PATH}" >&2
    exit 1
fi

if [ "${SYNC_CUSTOM_CODE}" = "1" ] && [ -d "${MODEL_NAME_OR_PATH}" ]; then
    MODELING_SRC="${SCRIPT_DIR}/models/modeling_llama_tomoe_gated_actual_moe.py"
    MODELING_DST="${MODEL_NAME_OR_PATH}/modeling_llama_tomoe_gated_actual_moe.py"
    if [ -f "${MODELING_DST}" ]; then
        echo "[sft] syncing custom modeling code: ${MODELING_DST}"
        cp "${MODELING_SRC}" "${MODELING_DST}"
    fi
fi

echo "[sft] model: ${MODEL_NAME_OR_PATH}"
echo "[sft] output: ${OUTPUT_DIR}"
echo "[sft] dataset: ${DATASET_NAME} split=${DATASET_SPLIT}"
echo "[sft] max_seq_length: ${MAX_SEQ_LENGTH}"
echo "[sft] packing: ${PACKING}"
echo "[sft] lr: ${LEARNING_RATE} scheduler=${LR_SCHEDULER_TYPE} warmup_ratio=${WARMUP_RATIO}"
echo "[sft] grad_accum: ${GRADIENT_ACCUMULATION_STEPS}"
echo "[sft] save_steps: ${SAVE_STEPS}"
echo "[sft] tomoe_moe_impl: ${TOMOE_MOE_IMPL}"
echo "[sft] master_port: ${MASTER_PORT}"

EXTRA_ARGS=()
if [ -n "${DATASET_CACHE_DIR}" ]; then
    EXTRA_ARGS+=(--dataset_cache_dir "${DATASET_CACHE_DIR}")
fi
if [ -n "${MAX_TRAIN_STEPS}" ]; then
    EXTRA_ARGS+=(--max_train_steps "${MAX_TRAIN_STEPS}")
fi
if [ "${COMPILE_MODEL}" = "1" ]; then
    EXTRA_ARGS+=(--compile_model --compile_mode "${COMPILE_MODE}")
fi
if [ "${LOAD_MODEL_ON_GPU}" = "1" ]; then
    EXTRA_ARGS+=(--load_model_on_gpu)
fi
if [ "${GRADIENT_CHECKPOINTING}" = "1" ]; then
    EXTRA_ARGS+=(--gradient_checkpointing)
fi
if [ "${USE_8BIT_ADAM}" = "1" ]; then
    EXTRA_ARGS+=(--use_8bit_adam)
fi
if [ "${PACKING}" = "1" ]; then
    EXTRA_ARGS+=(--packing)
fi
if [ "${SAVE_OPTIMIZER}" = "1" ]; then
    EXTRA_ARGS+=(--save_optimizer)
fi
if [ "${SAVE_OPTIMIZER_LATEST_ONLY}" = "1" ]; then
    EXTRA_ARGS+=(--save_optimizer_latest_only)
fi
if [ "${SAVE_AT_ITER0}" = "1" ]; then
    EXTRA_ARGS+=(--save_at_iter0)
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" train_sft_fsdp.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_split "${DATASET_SPLIT}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
    --num_workers "${NUM_WORKERS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --bf16 \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --moe_aux_loss_weight "${MOE_AUX_LOSS_WEIGHT}" \
    --tomoe_moe_impl "${TOMOE_MOE_IMPL}" \
    --fsdp_transformer_layer_cls_to_wrap "${FSDP_LAYER_CLS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
