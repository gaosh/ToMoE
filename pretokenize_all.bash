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
    --shuffle_files \
    --seed 42 \
    --add_eos \
    > fineweb_pretokenize.log 2>&1 &

PID1=$!

python data/pretokenize_and_pack.py \
    --model_name ${MODEL_NAME} \
    --data_dir /orange/sgao1/sgao1/openwebmath \
    --output_dir /orange/sgao1/sgao1/data/openwebmath_5b_packed_8192 \
    --seq_len 8192 \
    --target_tokens 5000000000 \
    --shard_sequences 4096 \
    --shuffle_files \
    --seed 43 \
    --add_eos \
    > openwebmath_pretokenize.log 2>&1 &

PID2=$!

wait $PID1
wait $PID2

echo "All datasets finished."