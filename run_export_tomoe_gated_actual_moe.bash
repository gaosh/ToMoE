#!/usr/bin/env bash
set -euo pipefail

# Export Stage-1 HN + gated-attention checkpoint into a standalone
# ToMoE actual-MoE + gated-attention HuggingFace model.
#
# This script overwrites the existing output model directory in-place.
# It does not run PPL evaluation. Use eval_tomoe_gated_actual_moe_ppl.py separately.

cd "$(dirname "$0")"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

HF_MODEL="${HF_MODEL:-meta-llama/Meta-Llama-3-8B}"
HN_DIR="${HN_DIR:-/orange/sgao1/sgao1/saved_hns/gated_attn_llama3_8b}"
HN_CKPT="${HN_CKPT:-${HN_DIR}/hn-gated-attn-ckpt-final-0.50.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/orange/sgao1/sgao1/saved_models/tomoe_gated_actual_moe_llama3_8b}"

DYNAMIC_EXPERTS="${DYNAMIC_EXPERTS:-8}"
GATE_RANK="${GATE_RANK:-128}"
GATE_INIT_BIAS="${GATE_INIT_BIAS:-3.0}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
SAVE_SHARD_SIZE="${SAVE_SHARD_SIZE:-8GB}"

if [[ ! -d "${HN_DIR}" ]]; then
  echo "[export] HN_DIR does not exist: ${HN_DIR}" >&2
  exit 1
fi

if [[ ! -f "${HN_CKPT}" ]]; then
  echo "[export] HN_CKPT does not exist: ${HN_CKPT}" >&2
  echo "[export] available checkpoints:" >&2
  find "${HN_DIR}" -maxdepth 1 -type f -name 'hn-gated-attn-ckpt*.pt' -print >&2
  exit 1
fi

echo "[export] HF_MODEL=${HF_MODEL}"
echo "[export] HN_CKPT=${HN_CKPT}"
echo "[export] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[export] DYNAMIC_EXPERTS=${DYNAMIC_EXPERTS}"
echo "[export] GATE_RANK=${GATE_RANK}"
echo "[export] GATE_INIT_BIAS=${GATE_INIT_BIAS}"
echo "[export] TORCH_DTYPE=${TORCH_DTYPE}"
echo "[export] SAVE_SHARD_SIZE=${SAVE_SHARD_SIZE}"

python export_tomoe_gated_actual_moe.py \
  --hf_model "${HF_MODEL}" \
  --hn_path "${HN_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --dynamic_experts "${DYNAMIC_EXPERTS}" \
  --gate_rank "${GATE_RANK}" \
  --gate_init_bias "${GATE_INIT_BIAS}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --save_tokenizer true \
  --low_cpu_mem_usage true \
  --save_shard_size "${SAVE_SHARD_SIZE}"

echo "[export] done: ${OUTPUT_DIR}"
