#!/bin/bash
#SBATCH --job-name=tomoe-cpt
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-b200
#SBATCH --reservation=sgao1b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:8
#SBATCH --time=336:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

cd /home/sgao1/ToMoE

source /orange/sgao1/sgao1/miniconda3/etc/profile.d/conda.sh
conda activate tomoe_megablocks

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/orange/sgao1/sgao1/saved_models/tomoe_gated_actual_moe_llama3_8b}"
export OUTPUT_DIR="${OUTPUT_DIR:-/orange/sgao1/sgao1/continual_pretrain_outputs/tomoe_gated_llama3_8b}"

export SEQ_LEN="${SEQ_LEN:-8192}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-4e-5}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export WARMUP_STEPS="${WARMUP_STEPS:-1000}"
export MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-20B}"

export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export SAVE_STEPS="${SAVE_STEPS:-5000}"
export SAVE_OPTIMIZER="${SAVE_OPTIMIZER:-0}"
export SAVE_OPTIMIZER_LATEST_ONLY="${SAVE_OPTIMIZER_LATEST_ONLY:-0}"
export SAVE_AT_ITER0="${SAVE_AT_ITER0:-1}"

export NUM_WORKERS="${NUM_WORKERS:-4}"
export FSDP_LAYER_CLS="${FSDP_LAYER_CLS:-LlamaDecoderLayer}"
export MOE_AUX_LOSS_WEIGHT="${MOE_AUX_LOSS_WEIGHT:-0.02}"
export TOMOE_MOE_IMPL="${TOMOE_MOE_IMPL:-grouped_gemm}"
export NPROC_PER_NODE=8
export MASTER_PORT="${MASTER_PORT:-29513}"
export COMPILE_MODEL="${COMPILE_MODEL:-1}"
export COMPILE_MODE="${COMPILE_MODE:-default}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
export SYNC_CUSTOM_CODE="${SYNC_CUSTOM_CODE:-1}"
export LOAD_MODEL_ON_GPU="${LOAD_MODEL_ON_GPU:-1}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SAVE_STEPS=${SAVE_STEPS}"
hostname
nvidia-smi

SAVE_STEPS="${SAVE_STEPS}" bash run_continual_pretrain_fsdp_8gpu.bash
