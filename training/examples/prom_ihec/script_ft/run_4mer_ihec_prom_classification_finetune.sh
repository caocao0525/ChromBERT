#!/bin/bash

# Fixed value
KMER=4

# Default values
MODEL_PATH="../pretrain_result"
DATA_PATH="../ft_data/classification"
OUTPUT_PATH="../ft_result/classification"
EPOCHS=10.0
LR=2e-5
BATCH_SIZE=32
SOURCE=../../..
TOKENIZER_PATH=$SOURCE/src/transformers/dnabert-config/bert-config-18-$KMER

# Parse input arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --data_path) DATA_PATH="$2"; shift ;;
        --output_path) OUTPUT_PATH="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --tokenizer_path) TOKENIZER_PATH="$2"; shift ;;  # optional override
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$MODEL_PATH" || -z "$DATA_PATH" || -z "$OUTPUT_PATH" ]]; then
  echo "❌ Error: --model_path, --data_path, and --output_path are required."
  exit 1
fi

# Run fine-tuning
python ../../run_finetune.py \
    --model_type dna \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name "$TOKENIZER_PATH" \
    --config_name "$TOKENIZER_PATH/config.json" \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --data_dir "$DATA_PATH" \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size=$BATCH_SIZE \
    --per_gpu_train_batch_size=$BATCH_SIZE \
    --learning_rate $LR \
    --num_train_epochs $EPOCHS \
    --output_dir "$OUTPUT_PATH" \
    --evaluate_during_training \
    --logging_steps 100 \
    --save_steps 5000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 8

