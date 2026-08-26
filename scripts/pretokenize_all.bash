#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B}"
FINEWEB_DATA_DIR="${FINEWEB_DATA_DIR:-}"
FINEWEB_PACKED_DIR="${FINEWEB_PACKED_DIR:-}"
OPENWEBMATH_DATA_DIR="${OPENWEBMATH_DATA_DIR:-}"
OPENWEBMATH_PACKED_DIR="${OPENWEBMATH_PACKED_DIR:-}"

for var_name in FINEWEB_DATA_DIR FINEWEB_PACKED_DIR OPENWEBMATH_DATA_DIR OPENWEBMATH_PACKED_DIR; do
    if [[ -z "${!var_name}" ]]; then
        echo "Set ${var_name} before running $0." >&2
        exit 1
    fi
done

python data/pretokenize_and_pack.py \
    --model_name "${MODEL_NAME}" \
    --data_dir "${FINEWEB_DATA_DIR}" \
    --output_dir "${FINEWEB_PACKED_DIR}" \
    --seq_len "${SEQ_LEN:-8192}" \
    --target_tokens "${FINEWEB_TARGET_TOKENS:-20000000000}" \
    --shard_sequences "${SHARD_SEQUENCES:-4096}" \
    --batch_size "${TOKENIZE_BATCH_SIZE:-2048}" \
    --shuffle_files \
    --seed "${FINEWEB_SEED:-42}" \
    --add_eos

python data/pretokenize_and_pack.py \
    --model_name "${MODEL_NAME}" \
    --data_dir "${OPENWEBMATH_DATA_DIR}" \
    --output_dir "${OPENWEBMATH_PACKED_DIR}" \
    --seq_len "${SEQ_LEN:-8192}" \
    --target_tokens "${OPENWEBMATH_TARGET_TOKENS:-5000000000}" \
    --shard_sequences "${SHARD_SEQUENCES:-4096}" \
    --batch_size "${TOKENIZE_BATCH_SIZE:-2048}" \
    --shuffle_files \
    --seed "${OPENWEBMATH_SEED:-43}" \
    --add_eos
