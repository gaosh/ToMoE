#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PACKED_DATA_DIRS="${PACKED_DATA_DIRS:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SEQ_LEN="${SEQ_LEN:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_BATCHES="${NUM_BATCHES:-2}"

if [[ -z "${PACKED_DATA_DIRS}" ]]; then
    echo "Set PACKED_DATA_DIRS to a colon-separated list of packed-data directories." >&2
    exit 1
fi

IFS=':' read -r -a DATA_DIRS <<< "${PACKED_DATA_DIRS}"

if [ "${#DATA_DIRS[@]}" -eq 0 ]; then
    echo "No packed data directories found in PACKED_DATA_DIRS." >&2
    exit 1
fi

echo "[dataset-test] data dirs:"
printf '  %s\n' "${DATA_DIRS[@]}"

echo "[dataset-test] launching torchrun with ${NPROC_PER_NODE} processes"

PYTHONUNBUFFERED=1 torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" test/test_distributed_packed_dataset.py \
    --data_dirs "${DATA_DIRS[@]}" \
    --seq_len "${SEQ_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_batches "${NUM_BATCHES}"
