#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PRETOKENIZE_SCRIPT="${SCRIPT_DIR}/pretokenize_all.bash"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/orange/sgao1/sgao1/saved_models/tomoe_gated_actual_moe_llama3_8b}"
OUTPUT_DIR="${OUTPUT_DIR:-/orange/sgao1/sgao1/continual_pretrain_outputs/tomoe_gated_llama3_8b}"

SEQ_LEN="${SEQ_LEN:-8192}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-25B}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
NUM_WORKERS="${NUM_WORKERS:-1}"
FSDP_LAYER_CLS="${FSDP_LAYER_CLS:-LlamaDecoderLayer}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
COMPILE_MODEL="${COMPILE_MODEL:-0}"
COMPILE_MODE="${COMPILE_MODE:-default}"

if [ -z "${MODEL_NAME_OR_PATH}" ] || [ "${MODEL_NAME_OR_PATH}" = "/path/to/tomoe_or_llama_model" ]; then
    echo "Please set MODEL_NAME_OR_PATH to the real HF/local model directory before launching." >&2
    echo "Example:" >&2
    echo "  MODEL_NAME_OR_PATH=/orange/sgao1/sgao1/saved_models/tomoe_gated_actual_moe_llama3_8b bash $0" >&2
    exit 1
fi

if [[ "${MODEL_NAME_OR_PATH}" = /* ]] && [ ! -d "${MODEL_NAME_OR_PATH}" ]; then
    echo "MODEL_NAME_OR_PATH is an absolute path but does not exist: ${MODEL_NAME_OR_PATH}" >&2
    exit 1
fi

MODEL_REALPATH="$(realpath "${MODEL_NAME_OR_PATH}")"
OUTPUT_REALPATH="$(realpath -m "${OUTPUT_DIR}")"
if [ "${MODEL_REALPATH}" = "${OUTPUT_REALPATH}" ]; then
    echo "OUTPUT_DIR must not be the same directory as MODEL_NAME_OR_PATH." >&2
    echo "MODEL_NAME_OR_PATH=${MODEL_REALPATH}" >&2
    echo "OUTPUT_DIR=${OUTPUT_REALPATH}" >&2
    exit 1
fi

mapfile -t DATA_DIRS < <(
    awk '
        $1 == "--output_dir" {
            gsub(/\\$/, "", $2)
            print $2
        }
    ' "${PRETOKENIZE_SCRIPT}"
)

if [ "${#DATA_DIRS[@]}" -eq 0 ]; then
    echo "No packed data directories found from ${PRETOKENIZE_SCRIPT}" >&2
    exit 1
fi

echo "[continual-pretrain] model: ${MODEL_NAME_OR_PATH}"
echo "[continual-pretrain] output: ${OUTPUT_DIR}"
echo "[continual-pretrain] compile: ${COMPILE_MODEL} (${COMPILE_MODE})"
echo "[continual-pretrain] data dirs:"
printf '  %s\n' "${DATA_DIRS[@]}"

EXTRA_ARGS=()
if [ "${COMPILE_MODEL}" = "1" ]; then
    EXTRA_ARGS+=(--compile_model --compile_mode "${COMPILE_MODE}")
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" train_continual_pretrain_fsdp.py \
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
    --gradient_checkpointing \
    --use_8bit_adam \
    --fsdp_transformer_layer_cls_to_wrap "${FSDP_LAYER_CLS}" \
    --num_workers "${NUM_WORKERS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    "${EXTRA_ARGS[@]}"
