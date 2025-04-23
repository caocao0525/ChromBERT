#!/bin/bash

# Default values
KMER=4
TRAIN_FILE="../pretrain_data/pretraining_small.txt"  # Default training file
TEST_FILE="../pretrain_data/pretraining_small.txt"  # Default test file
SOURCE="../../.."
OUTPUT_PATH="../pretrain_result/"
TRAIN_BATCH=5
EVAL_BATCH=3
MAX_STEPS=500  # Default max steps
LEARNING_RATE=2e-4  # Default learning rate
MLM_PROB=0.025  # Default MLM probability

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --train_file) TRAIN_FILE="$2"; shift ;;  # User-defined training file
        --test_file) TEST_FILE="$2"; shift ;;  # User-defined test file
        --max_steps) MAX_STEPS="$2"; shift ;;  # Change max steps
        --learning_rate) LEARNING_RATE="$2"; shift ;;  # Change learning rate
        --mlm_prob) MLM_PROB="$2"; shift ;;  # Change MLM probability
        --train_batch) TRAIN_BATCH="$2"; shift ;;  # Change batch size
        --eval_batch) EVAL_BATCH="$2"; shift ;;  # Change eval batch size
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Ensure train and test files have default values if not provided
if [ -z "$TRAIN_FILE" ]; then
  echo "Using default training file: ../pretrain_data/pretraining_small.txt"
  TRAIN_FILE="../pretrain_data/pretraining_small.txt"
fi

if [ -z "$TEST_FILE" ]; then
  echo "Using default test file: ../pretrain_data/pretraining_small.txt"
  TEST_FILE="../pretrain_data/pretraining_small.txt"
fi

# Run the pretraining script
python ../../run_pretrain.py \
 --output_dir $OUTPUT_PATH \
 --model_type=dna \
 --tokenizer_name=dna$KMER \
 --config_name=$SOURCE/src/transformers/dnabert-config/bert-config-$KMER/config.json \
 --do_train \
 --train_data_file="$TRAIN_FILE" \
 --do_eval \
 --eval_data_file="$TEST_FILE" \
 --mlm \
 --gradient_accumulation_steps 25 \
 --per_gpu_train_batch_size $TRAIN_BATCH \
 --per_gpu_eval_batch_size $EVAL_BATCH \
 --save_steps 50 \
 --save_total_limit 20 \
 --max_steps $MAX_STEPS \
 --evaluate_during_training \
 --logging_steps 50 \
 --line_by_line \
 --learning_rate $LEARNING_RATE \
 --block_size 512  \
 --adam_epsilon 1e-6 \
 --weight_decay 0.01 \
 --beta1 0.9 \
 --beta2 0.98 \
 --mlm_probability $MLM_PROB \
 --warmup_steps 50  \
 --overwrite_output_dir \
 --n_process 24

