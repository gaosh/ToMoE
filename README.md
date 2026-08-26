# ToMoE (TMLR)

Official implementation of **“ToMoE: Converting Dense Large Language Models to Mixture-of-Experts through Dynamic Structural Pruning.”**

- [Paper](https://openreview.net/forum?id=RFHq46pjb6)
- [Original GitHub repository](https://github.com/gaosh/ToMoE)

## Overview

This repository contains two separate workflows:

1. **Original ToMoE conversion** — train a hypernetwork that learns dynamic structural pruning, then convert a dense LLaMA checkpoint into a pruned MoE checkpoint.
2. **Upcycling and continual pretraining** — train ToMoE with gated attention, export an explicit Hugging Face MoE checkpoint, prepare packed token data, and continue pretraining the exported model with FSDP. Tulu 3 supervised fine-tuning is also available as an optional final stage.

The second workflow does not replace the original ToMoE path. Choose the path that matches the checkpoint you want to produce.

## Previous version

The repository state before the upcycling and continual-pretraining work remains available as the [original ToMoE snapshot](https://github.com/gaosh/ToMoE/tree/a3e1b655dbfd5af0382c67e2a8069301c089f022). Use that permanent link if you only need the earlier hypernetwork-training and pruning implementation.

## What is new

The updated codebase adds:

- Gated-attention ToMoE upcycling for Llama 3.
- Export from a learned hypernetwork checkpoint to an explicit, standalone Hugging Face MoE model.
- Parquet dataset download and fixed-length token packing for continual pretraining.
- Eight-GPU FSDP continual pretraining, checkpoint resume, and optional optimizer-state persistence.
- Perplexity evaluation for exported or continually pretrained checkpoints.
- Streaming Tulu 3 supervised fine-tuning.
- Optional grouped-GEMM expert execution and validation utilities.
- Portable launch scripts collected under `scripts/`, with machine-specific paths supplied explicitly.

## Installation

The supplied environment targets Linux, CUDA 12.8, Python 3.11, and PyTorch 2.7.1.

```bash
conda env create -f environment.yml
conda activate llm-env
```

The default MoE implementation is `naive`. For the faster grouped-GEMM path, install the optional MegaBlocks grouped-GEMM dependencies and set `TOMOE_MOE_IMPL=grouped_gemm`:

```bash
pip install 'megablocks[gg]==0.10.0'
```

Access to gated Hugging Face models such as Llama 2 or Llama 3 requires an accepted model license and an authenticated Hugging Face session.

## Workflow A: Original ToMoE conversion

### 1. Train the hypernetwork

```bash
torchrun --nproc_per_node=1 --master_port=12343 train_tomoe.py \
  --use_bf16 true \
  --save_interval 100000 \
  --dynamic_experts 8 \
  --dynamic_alpha 3.0 \
  --load_balance_alpha 1.0 \
  --hf_model meta-llama/Llama-2-7b-hf \
  --p 0.5 \
  --total_n_step 20000 \
  --lam 16.0 \
  --kd_loss true \
  --dataset_list '["mix"]' \
  --dataset_path /path/to/dataset-cache \
  --dataset_seed 777 \
  --use_fsdp false \
  --out_dir /path/to/hypernetwork-output
```

### 2. Prune and export the model

```bash
python prune_tomoe.py \
  --hf_model meta-llama/Llama-2-7b-hf \
  --hn_path /path/to/hypernetwork-output/hn-ckpt-final-0.50.pt \
  --output_dir /path/to/tomoe-model \
  --dynamic_experts 8 \
  --attn_prune true
```

### 3. Run zero-shot evaluation

Use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
accelerate launch --main_process_port 12323 --num_processes 1 \
  -m lm_eval --model hf \
  --model_args pretrained=/path/to/tomoe-model,dtype=bfloat16,trust_remote_code=true \
  --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande \
  --device cuda:0 \
  --batch_size 32
```

## Workflow B: Upcycling and continual pretraining

The upcycling path is:

```text
dense Llama checkpoint
  -> gated-attention ToMoE training
  -> explicit Hugging Face MoE export
  -> packed-data continual pretraining
  -> optional Tulu 3 SFT and evaluation
```

The launch scripts use environment variables for machine-specific paths. They fail early when a required path is missing.

### 1. Train gated-attention ToMoE

This stage learns the structural-pruning hypernetwork and gated-attention parameters from the dense Llama 3 checkpoint. `DATASET_PATH` is the cache root used by the existing mixed-dataset loader.

```bash
HF_MODEL=meta-llama/Meta-Llama-3-8B \
DATASET_PATH=/path/to/dataset-cache \
OUT_DIR=/path/to/gated-attention-hn \
bash scripts/run_gated_attn_llama3_8b.bash
```

The default final checkpoint is named `hn-gated-attn-ckpt-final-0.50.pt`. Override the training knobs through variables such as `TOTAL_N_STEP`, `DYNAMIC_EXPERTS`, `PRUNE_RATIO`, `GATE_RANK`, and `NPROC_PER_NODE`.

### 2. Export an explicit MoE checkpoint

The exporter materializes the selected expert weights, copies the custom modeling code, and writes a standalone Hugging Face checkpoint.

```bash
HF_MODEL=meta-llama/Meta-Llama-3-8B \
HN_CKPT=/path/to/gated-attention-hn/hn-gated-attn-ckpt-final-0.50.pt \
OUTPUT_DIR=/path/to/tomoe-actual-moe \
bash scripts/run_export_tomoe_gated_actual_moe.bash
```

### 3. Prepare continual-pretraining data

`train_continual_pretrain_fsdp.py` consumes directories of fixed-length `.npy` token shards. If you already have Parquet files with a `text` column, skip the download command and point the packing command at them.

The included presets download FineWeb-Edu `sample-100BT` and OpenWebMath. These are large datasets; choose storage and token limits appropriate for your run.

```bash
python data/dataset_prepare.py \
  --output-root /path/to/raw-data \
  --cache-root /path/to/huggingface-cache \
  --datasets fineweb_edu openwebmath
```

Pack both datasets with the same tokenizer and sequence length used for training:

```bash
MODEL_NAME=meta-llama/Meta-Llama-3-8B \
FINEWEB_DATA_DIR=/path/to/raw-data/fineweb_edu \
FINEWEB_PACKED_DIR=/path/to/packed/fineweb-edu-8192 \
OPENWEBMATH_DATA_DIR=/path/to/raw-data/openwebmath \
OPENWEBMATH_PACKED_DIR=/path/to/packed/openwebmath-8192 \
bash scripts/pretokenize_all.bash
```

The defaults pack up to 20B FineWeb-Edu tokens and 5B OpenWebMath tokens at sequence length 8192. Override `FINEWEB_TARGET_TOKENS`, `OPENWEBMATH_TARGET_TOKENS`, `SEQ_LEN`, or `SHARD_SEQUENCES` for smaller runs.

### 4. Run continual pretraining

`PACKED_DATA_DIRS` is a colon-separated list. The default launcher uses eight processes, FSDP, BF16, and an 8-bit Adam optimizer.

```bash
MODEL_NAME_OR_PATH=/path/to/tomoe-actual-moe \
PACKED_DATA_DIRS=/path/to/packed/fineweb-edu-8192:/path/to/packed/openwebmath-8192 \
OUTPUT_DIR=/path/to/continual-pretraining-output \
NPROC_PER_NODE=8 \
bash scripts/run_continual_pretrain_fsdp_8gpu.bash
```

Use `TOMOE_MOE_IMPL=grouped_gemm` after installing the optional grouped-GEMM dependency. Common overrides include `MAX_TRAIN_TOKENS`, `LEARNING_RATE`, `SAVE_STEPS`, `GRADIENT_ACCUMULATION_STEPS`, `ATTN_IMPLEMENTATION`, and `COMPILE_MODEL`.

Resume from a saved checkpoint with the same data and output settings:

```bash
RESUME_FROM_CHECKPOINT=/path/to/continual-pretraining-output/checkpoint-5000 \
PACKED_DATA_DIRS=/path/to/packed/fineweb-edu-8192:/path/to/packed/openwebmath-8192 \
OUTPUT_DIR=/path/to/continual-pretraining-output \
bash scripts/run_resume_continual_pretrain_fsdp_8gpu.bash
```

If a compiled FSDP checkpoint contains `_orig_mod` prefixes, convert it into a normal Hugging Face checkpoint:

```bash
python convert_ckpt.py \
  --src-dir /path/to/source-checkpoint \
  --dst-dir /path/to/converted-checkpoint \
  --test-load
```

### 5. Evaluate perplexity

```bash
MODEL_NAME_OR_PATH=/path/to/exported-or-continued-model \
DATASETS=wikitext \
bash scripts/run_eval_tomoe_gated_actual_moe_ppl.bash
```

`DATASETS` accepts a comma-separated list supported by `eval_tomoe_gated_actual_moe_ppl.py`. The default evaluation uses a 2048-token block and at most 524,288 tokens.

### 6. Optional Tulu 3 supervised fine-tuning

```bash
MODEL_NAME_OR_PATH=/path/to/continual-pretraining-checkpoint \
OUTPUT_DIR=/path/to/tulu3-sft-output \
DATASET_NAME=allenai/tulu-3-sft-mixture \
NPROC_PER_NODE=8 \
bash scripts/run_sft_tulu3_fsdp_8gpu.bash
```

## Repository layout

- `train_tomoe.py` and `prune_tomoe.py`: original ToMoE hypernetwork training and pruning.
- `train_tomoe_gated_attn.py`: gated-attention upcycling stage.
- `export_tomoe_gated_actual_moe.py`: explicit MoE Hugging Face export.
- `train_continual_pretrain_fsdp.py`: FSDP continual pretraining on packed token shards.
- `train_sft_fsdp.py`: streaming Tulu-style supervised fine-tuning.
- `eval_tomoe_gated_actual_moe_ppl.py`: perplexity evaluation.
- `scripts/`: launchers for upcycling, export, data packing, continual pretraining, resume, evaluation, SFT, and packed-data validation.
- `models/`: dense, dynamic-pruning, gated-attention, and explicit-MoE model definitions.
- `tomoe/`: hypernetwork and pruning helpers.
- `data/`: dataset download, packing, and loading utilities.
- `test/`: distributed data and grouped-GEMM validation scripts.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{
    gao2026tomoe,
    title={ToMoE: Converting Dense Large Language Models to Mixture-of-Experts through Dynamic Structural Pruning},
    author={Shangqian Gao and Ting Hua and Reza Shirkavand and Chi-Heng Lin and Zheng Tang and Zhengao Li and Longge Yuan and Fangyi Li and Zeyu Zhang and Alireza Ganjdanesh and Qian Lou and Jie Xu and Yen-Chang Hsu},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2026},
    url={https://openreview.net/forum?id=RFHq46pjb6},
    note={J2C Certification}
}
```
