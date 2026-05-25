#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PRETOKENIZE_SCRIPT="${SCRIPT_DIR}/pretokenize_all.bash"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SEQ_LEN="${SEQ_LEN:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_BATCHES="${NUM_BATCHES:-2}"

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

echo "[dataset-test] data dirs:"
printf '  %s\n' "${DATA_DIRS[@]}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" test_distributed_packed_dataset.py \
    --data_dirs "${DATA_DIRS[@]}" \
    --seq_len "${SEQ_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_batches "${NUM_BATCHES}"
