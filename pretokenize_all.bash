#!/bin/bash

set -e

MODEL_NAME="meta-llama/Meta-Llama-3-8B"

python data/pretokenize_and_pack.py \
    --model_name ${MODEL_NAME} \
    --data_dir /orange/sgao1/sgao1/data/fineweb_edu_100bt \
    --output_dir /orange/sgao1/sgao1/data/fineweb_edu_20b_packed_8192 \
    --seq_len 8192 \
    --target_tokens 20000000000 \
    --shard_sequences 4096 \
    --batch_size 2048 \
    --shuffle_files \
    --seed 42 \
    --add_eos

python data/pretokenize_and_pack.py \
    --model_name ${MODEL_NAME} \
    --data_dir /orange/sgao1/sgao1/openwebmath \
    --output_dir /orange/sgao1/sgao1/data/openwebmath_5b_packed_8192 \
    --seq_len 8192 \
    --target_tokens 5000000000 \
    --shard_sequences 4096 \
    --batch_size 2048 \
    --shuffle_files \
    --seed 43 \
    --add_eos