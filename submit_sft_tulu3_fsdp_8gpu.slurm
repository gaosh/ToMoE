#!/bin/bash
#SBATCH --job-name=tomoe-sft
#SBATCH --account=sgao1
#SBATCH --qos=sgao1
#SBATCH --partition=hpg-b200
#SBATCH --reservation=sgao1b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:8
#SBATCH --time=168:00:00
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

export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/orange/sgao1/sgao1/continual_pretrain_outputs/tomoe_gated_llama3_8b/checkpoint-converted}"
export OUTPUT_DIR="${OUTPUT_DIR:-/orange/sgao1/sgao1/sft_outputs/tomoe_gated_llama3_8b_tulu3}"

export DATASET_NAME="${DATASET_NAME:-allenai/tulu-3-sft-mixture}"
export DATASET_SPLIT="${DATASET_SPLIT:-train}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
export SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-10000}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-7500}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export SAVE_STEPS="${SAVE_STEPS:-5000}"

export NUM_WORKERS="${NUM_WORKERS:-4}"
export FSDP_LAYER_CLS="${FSDP_LAYER_CLS:-LlamaDecoderLayer}"
export MOE_AUX_LOSS_WEIGHT="${MOE_AUX_LOSS_WEIGHT:-0.01}"
export TOMOE_MOE_IMPL="${TOMOE_MOE_IMPL:-grouped_gemm}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-29517}"
export COMPILE_MODEL="${COMPILE_MODEL:-0}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
export LOAD_MODEL_ON_GPU="${LOAD_MODEL_ON_GPU:-1}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
export SYNC_CUSTOM_CODE="${SYNC_CUSTOM_CODE:-1}"
export USE_8BIT_ADAM="${USE_8BIT_ADAM:-1}"
export SAVE_OPTIMIZER="${SAVE_OPTIMIZER:-0}"
export SAVE_OPTIMIZER_LATEST_ONLY="${SAVE_OPTIMIZER_LATEST_ONLY:-0}"
export SAVE_AT_ITER0="${SAVE_AT_ITER0:-0}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
hostname
nvidia-smi

bash scripts/run_sft_tulu3_fsdp_8gpu.bash
