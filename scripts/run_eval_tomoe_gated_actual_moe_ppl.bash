#!/usr/bin/env bash
set -euo pipefail

# Run perplexity evaluation for the exported ToMoE gated actual-MoE model.
#
# Defaults are aligned with run_export_tomoe_gated_actual_moe.bash.
# Override any setting from the command line, for example:
#   MODEL_NAME_OR_PATH=/path/to/model DATASETS=wikitext,ptb bash scripts/run_eval_tomoe_gated_actual_moe_ppl.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EVAL_SCRIPT="${EVAL_SCRIPT:-eval_tomoe_gated_actual_moe_ppl.py}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-}"

DATASETS="${DATASETS:-wikitext}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"
MAX_TOKENS="${MAX_TOKENS:-524288}"
DEVICE="${DEVICE:-cuda:0}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LOW_CPU_MEM_USAGE="${LOW_CPU_MEM_USAGE:-true}"
LOAD_ON_CPU="${LOAD_ON_CPU:-true}"
ALLOW_CPU_FALLBACK="${ALLOW_CPU_FALLBACK:-true}"
CUDA_MARGIN_GIB="${CUDA_MARGIN_GIB:-2.0}"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "[ppl] EVAL_SCRIPT does not exist: ${EVAL_SCRIPT}" >&2
  exit 1
fi

if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
  echo "Set MODEL_NAME_OR_PATH to the exported model or a continual-pretraining checkpoint." >&2
  exit 1
fi

if [[ "${MODEL_NAME_OR_PATH}" = /* ]] && [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "[ppl] MODEL_NAME_OR_PATH is an absolute path but does not exist: ${MODEL_NAME_OR_PATH}" >&2
  echo "[ppl] Build it first with scripts/run_export_tomoe_gated_actual_moe.bash or set MODEL_NAME_OR_PATH." >&2
  exit 1
fi

TOKENIZER_ARGS=()
if [[ -n "${TOKENIZER_NAME_OR_PATH}" ]]; then
  TOKENIZER_ARGS+=(--tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}")
fi

echo "[ppl] script=${EVAL_SCRIPT}"
echo "[ppl] model=${MODEL_NAME_OR_PATH}"
echo "[ppl] tokenizer=${TOKENIZER_NAME_OR_PATH:-${MODEL_NAME_OR_PATH}}"
echo "[ppl] datasets=${DATASETS}"
echo "[ppl] block_size=${BLOCK_SIZE}"
echo "[ppl] max_tokens=${MAX_TOKENS}"
echo "[ppl] device=${DEVICE}"
echo "[ppl] torch_dtype=${TORCH_DTYPE}"

python "${EVAL_SCRIPT}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  "${TOKENIZER_ARGS[@]}" \
  --datasets "${DATASETS}" \
  --block_size "${BLOCK_SIZE}" \
  --max_tokens "${MAX_TOKENS}" \
  --device "${DEVICE}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --low_cpu_mem_usage "${LOW_CPU_MEM_USAGE}" \
  --load_on_cpu "${LOAD_ON_CPU}" \
  --allow_cpu_fallback "${ALLOW_CPU_FALLBACK}" \
  --cuda_margin_gib "${CUDA_MARGIN_GIB}"
