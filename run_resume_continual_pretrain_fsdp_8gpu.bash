#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-${1:-}}"
if [ -z "${RESUME_FROM_CHECKPOINT}" ]; then
    echo "Usage: RESUME_FROM_CHECKPOINT=/path/to/checkpoint bash $0" >&2
    echo "   or: bash $0 /path/to/checkpoint" >&2
    exit 1
fi

if [ ! -d "${RESUME_FROM_CHECKPOINT}" ]; then
    echo "RESUME_FROM_CHECKPOINT does not exist: ${RESUME_FROM_CHECKPOINT}" >&2
    exit 1
fi

# CPT resumes model weights from the checkpoint directory. If SAVE_OPTIMIZER=1
# and optimizer_state/ exists, each rank also restores its optimizer shard.
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${RESUME_FROM_CHECKPOINT}}"
export RESUME_FROM_CHECKPOINT
export SAVE_OPTIMIZER="${SAVE_OPTIMIZER:-1}"

echo "[resume-cpt] model_name_or_path=${MODEL_NAME_OR_PATH}"
echo "[resume-cpt] resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}"
echo "[resume-cpt] save_optimizer=${SAVE_OPTIMIZER}"

shift $(( $# > 0 ? 1 : 0 ))
bash run_continual_pretrain_fsdp_8gpu.bash \
    --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}" \
    "$@"
