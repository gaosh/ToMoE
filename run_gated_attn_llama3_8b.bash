#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

HF_MODEL="${HF_MODEL:-meta-llama/Meta-Llama-3-8B}"
OUT_DIR="${OUT_DIR:-}"
DATASET_PATH="${DATASET_PATH:-.}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-12344}"

if [[ -z "${OUT_DIR}" ]]; then
  echo "Set OUT_DIR to the directory where gated-attention hypernetwork checkpoints should be saved." >&2
  echo "Example: OUT_DIR=./outputs/gated-attn bash $0" >&2
  exit 1
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" train_tomoe_gated_attn.py \
  --use_bf16 true \
  --save_interval "${SAVE_INTERVAL:-100000}" \
  --hf_model "${HF_MODEL}" \
  --total_n_step "${TOTAL_N_STEP:-20000}" \
  --kd_loss true \
  --dataset_list '["mix"]' \
  --dataset_path "${DATASET_PATH}" \
  --dataset_seed "${DATASET_SEED:-777}" \
  --use_fsdp "${USE_FSDP:-false}" \
  --dynamic_experts "${DYNAMIC_EXPERTS:-8}" \
  --dynamic_alpha "${DYNAMIC_ALPHA:-3.0}" \
  --load_balance_alpha "${LOAD_BALANCE_ALPHA:-1.0}" \
  --p "${PRUNE_RATIO:-0.5}" \
  --lam "${LAMBDA:-16.0}" \
  --hn_lr "${HN_LR:-1e-3}" \
  --gate_rank "${GATE_RANK:-128}" \
  --gate_init_bias "${GATE_INIT_BIAS:-3.0}" \
  --gate_reg_weight "${GATE_REG_WEIGHT:-0.0}" \
  --gate_reg_type "${GATE_REG_TYPE:-l1}" \
  --out_dir "${OUT_DIR}" \
  "$@"
