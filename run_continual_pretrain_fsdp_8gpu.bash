#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PACKED_DATA_DIRS="${PACKED_DATA_DIRS:-}"

SEQ_LEN="${SEQ_LEN:-8192}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-20B}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_OPTIMIZER="${SAVE_OPTIMIZER:-0}"
SAVE_OPTIMIZER_LATEST_ONLY="${SAVE_OPTIMIZER_LATEST_ONLY:-0}"
SAVE_AT_ITER0="${SAVE_AT_ITER0:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FSDP_LAYER_CLS="${FSDP_LAYER_CLS:-LlamaDecoderLayer}"
MOE_AUX_LOSS_WEIGHT="${MOE_AUX_LOSS_WEIGHT:-0.02}"
TOMOE_MOE_IMPL="${TOMOE_MOE_IMPL:-naive}"
TOMOE_MOE_IMPL="$(printf '%s' "${TOMOE_MOE_IMPL}" | xargs)"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29513}"
COMPILE_MODEL="${COMPILE_MODEL:-1}"
COMPILE_MODE="${COMPILE_MODE:-default}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
SYNC_CUSTOM_CODE="${SYNC_CUSTOM_CODE:-1}"
LOAD_MODEL_ON_GPU="${LOAD_MODEL_ON_GPU:-1}"

if [[ -z "${MODEL_NAME_OR_PATH}" || -z "${OUTPUT_DIR}" || -z "${PACKED_DATA_DIRS}" ]]; then
    echo "Set MODEL_NAME_OR_PATH, OUTPUT_DIR, and PACKED_DATA_DIRS before launching." >&2
    echo "PACKED_DATA_DIRS is a colon-separated list of packed-data directories." >&2
    echo "Example: MODEL_NAME_OR_PATH=./model OUTPUT_DIR=./outputs/cpt PACKED_DATA_DIRS=./packed/fineweb:./packed/math bash $0" >&2
    exit 1
fi

if [[ "${MODEL_NAME_OR_PATH}" = /* ]] && [ ! -d "${MODEL_NAME_OR_PATH}" ]; then
    echo "MODEL_NAME_OR_PATH is an absolute path but does not exist: ${MODEL_NAME_OR_PATH}" >&2
    exit 1
fi

if [ "${TOMOE_MOE_IMPL}" != "naive" ] && [ "${TOMOE_MOE_IMPL}" != "grouped_gemm" ]; then
    echo "TOMOE_MOE_IMPL must be 'naive' or 'grouped_gemm', got: ${TOMOE_MOE_IMPL}" >&2
    exit 1
fi

if [ "${SYNC_CUSTOM_CODE}" = "1" ] && [ -d "${MODEL_NAME_OR_PATH}" ]; then
    MODELING_SRC="${SCRIPT_DIR}/models/modeling_llama_tomoe_gated_actual_moe.py"
    MODELING_DST="${MODEL_NAME_OR_PATH}/modeling_llama_tomoe_gated_actual_moe.py"
    if [ -f "${MODELING_DST}" ]; then
        echo "[continual-pretrain] syncing custom modeling code: ${MODELING_DST}"
        cp "${MODELING_SRC}" "${MODELING_DST}"
    fi
fi

if [[ -d "${MODEL_NAME_OR_PATH}" ]]; then
    MODEL_REALPATH="$(realpath "${MODEL_NAME_OR_PATH}")"
    OUTPUT_REALPATH="$(realpath -m "${OUTPUT_DIR}")"
    if [ "${MODEL_REALPATH}" = "${OUTPUT_REALPATH}" ]; then
        echo "OUTPUT_DIR must not be the same directory as MODEL_NAME_OR_PATH." >&2
        echo "MODEL_NAME_OR_PATH=${MODEL_REALPATH}" >&2
        echo "OUTPUT_DIR=${OUTPUT_REALPATH}" >&2
        exit 1
    fi
fi

IFS=':' read -r -a DATA_DIRS <<< "${PACKED_DATA_DIRS}"

if [ "${#DATA_DIRS[@]}" -eq 0 ]; then
    echo "No packed data directories found in PACKED_DATA_DIRS." >&2
    exit 1
fi

for data_dir in "${DATA_DIRS[@]}"; do
    if [[ ! -d "${data_dir}" ]]; then
        echo "Packed data directory does not exist: ${data_dir}" >&2
        exit 1
    fi
done

echo "[continual-pretrain] model: ${MODEL_NAME_OR_PATH}"
echo "[continual-pretrain] output: ${OUTPUT_DIR}"
echo "[continual-pretrain] save_steps: ${SAVE_STEPS}"
echo "[continual-pretrain] save_optimizer: ${SAVE_OPTIMIZER}"
echo "[continual-pretrain] save_optimizer_latest_only: ${SAVE_OPTIMIZER_LATEST_ONLY}"
echo "[continual-pretrain] save_at_iter0: ${SAVE_AT_ITER0}"
echo "[continual-pretrain] compile: ${COMPILE_MODEL} (${COMPILE_MODE})"
echo "[continual-pretrain] attention: ${ATTN_IMPLEMENTATION}"
echo "[continual-pretrain] moe_aux_loss_weight: ${MOE_AUX_LOSS_WEIGHT}"
echo "[continual-pretrain] tomoe_moe_impl: ${TOMOE_MOE_IMPL}"
echo "[continual-pretrain] load_model_on_gpu: ${LOAD_MODEL_ON_GPU}"
echo "[continual-pretrain] master_port: ${MASTER_PORT}"
echo "[continual-pretrain] data dirs:"
printf '  %s\n' "${DATA_DIRS[@]}"

EXTRA_ARGS=()
if [ "${COMPILE_MODEL}" = "1" ]; then
    EXTRA_ARGS+=(--compile_model --compile_mode "${COMPILE_MODE}")
fi
if [ "${LOAD_MODEL_ON_GPU}" = "1" ]; then
    EXTRA_ARGS+=(--load_model_on_gpu)
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

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" train_continual_pretrain_fsdp.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_dirs "${DATA_DIRS[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --seq_len "${SEQ_LEN}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --max_train_tokens "${MAX_TRAIN_TOKENS}" \
    --bf16 \
    --use_8bit_adam \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --moe_aux_loss_weight "${MOE_AUX_LOSS_WEIGHT}" \
    --tomoe_moe_impl "${TOMOE_MOE_IMPL}" \
    --fsdp_transformer_layer_cls_to_wrap "${FSDP_LAYER_CLS}" \
    --num_workers "${NUM_WORKERS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
#    --gradient_checkpointing \
