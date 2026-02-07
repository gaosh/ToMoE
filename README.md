# ToMoE (TMLR)
Official implementation of "ToMoE: Converting Dense Large Language Models to Mixture-of-Experts through Dynamic Structural Pruning."

Paper: https://openreview.net/pdf?id=RFHq46pjb6

## Overview
This repo provides:
- Hypernetwork training for dynamic structural pruning.
- A conversion script to turn dense LLaMA models into pruned MoE models.
- Model definitions for the pruned MoE variants.

Dependencies are specified in `environment.yml`.

## Key scripts
- `train_tomoe.py`: Train the hypernetwork.
- `prune_tomoe.py`: Convert a dense model to a pruned MoE model.

## Example: Training the hypernetwork
```bash
CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 --master_port=12343 train_tomoe.py \
    --use_bf16 True \
    --save_interval 100000 \
    --dynamic_experts 8 \
    --dynamic_alpha 3.0 \
    --load_balance_alpha 1.0 \
    --hf_model meta-llama/Llama-2-7b-hf \
    --p 0.5 \
    --total_n_step 20000 \
    --lam 16.0 \
    --kd_loss True \
    --dataset_list ['mix'] \
    --dataset_seed 777 --use_fsdp False  --out_dir /path/to/output_dir
```

## Example: Pruning
```bash
python prune_tomoe.py \
    --hf_model meta-llama/Llama-2-7b-hf \
    --hn_path /path/to/hn-ckpt-final.pt \
    --output_dir /path/to/tomoe_model \
    --dynamic_experts 8 \
    --attn_prune true
```

## Example: Zero-Shot Evaluation
Please refer to https://github.com/EleutherAI/lm-evaluation-harness
```bash
accelerate launch --main_process_port 12323 --num_processes 1 \
    -m lm_eval --model hf \
    --model_args pretrained=/path/to/tomoe_model,dtype=bfloat16,trust_remote_code=true \
    --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande \
    --device cuda:0 \
    --batch_size 32
```

## Repo layout
- `models/`: Model definitions (dense + pruned MoE).
- `tomoe/`: Hypernetwork and pruning helper utilities.
- `utils/`: Training/runtime helpers.
- `data/`: Dataset utilities.

## Notes
- The pruning pipeline expects a trained hypernetwork checkpoint.
- For best results, train and prune with the same base model family.

## Citation
If you found this repo useful, please cite:
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

