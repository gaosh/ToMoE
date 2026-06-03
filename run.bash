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
--dataset_seed 777 --use_fsdp False  --out_dir /orange/sgao1/sgao1/saved_hns/hn_prune_llama2_7b > llama2_7b_0.5_e8_new.txt 2>&1 &

accelerate launch \
  --num_processes 8 \
  -m lm_eval \
  --model hf \
  --apply_chat_template \
  --model_args "pretrained=/orange/sgao1/sgao1/continual_pretrain_outputs/tomoe_gated_llama3_8b/checkpoint-converted,dtype=bfloat16,trust_remote_code=True" \
  --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,boolq,sciq \
  --batch_size auto \
  #--output_path eval_results