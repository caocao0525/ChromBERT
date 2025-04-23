#!/bin/bash

# === RELATIVE CONFIGURATION ===
export KMER=4
export SOURCE=../../..
export OUTPUT_PATH=../pretrain_result
export TOKENIZER_PATH=$SOURCE/src/transformers/dnabert-config/bert-config-18-$KMER
export TRAIN_BATCH=16
export EVAL_BATCH=8
export PERPLEXITY_THRESHOLD=1.1

# Relative path to split chunks
export SPLIT_DIR=../pretrain_data/split_chunks/chunk_*

# Initialize model path
MODEL_PATH=""

# === LOOP OVER CHUNKS ===
for CHUNK in $SPLIT_DIR
do
  echo "[INFO] Processing chunk: $CHUNK"

  if [ -n "$MODEL_PATH" ]; then
    MODEL_ARG="--model_name_or_path $MODEL_PATH --tokenizer_name $MODEL_PATH"
  else
    MODEL_ARG=""
  fi

  OUTPUT_CHUNK_DIR=$OUTPUT_PATH/$(basename "$CHUNK")_output
  mkdir -p "$OUTPUT_CHUNK_DIR"

  # Tokenizer info debug
  echo "[DEBUG] Using tokenizer from $TOKENIZER_PATH"
  python -c "
from transformers import DNATokenizer
tokenizer = DNATokenizer.from_pretrained('$TOKENIZER_PATH')
print(f'Tokenizer vocabulary size: {len(tokenizer.vocab)}')
"

  # Run pretraining
  python ../../run_pretrain.py \
    --output_dir "$OUTPUT_CHUNK_DIR" \
    --model_type=dna \
    --tokenizer_name="$TOKENIZER_PATH" \
    --config_name="$TOKENIZER_PATH/config.json" \
    --do_train \
    --train_data_file="$(realpath "$CHUNK")" \
    --do_eval \
    --eval_data_file="$(realpath "$CHUNK")" \
    --mlm \
    --gradient_accumulation_steps 25 \
    --per_gpu_train_batch_size $TRAIN_BATCH \
    --per_gpu_eval_batch_size $EVAL_BATCH \
    --save_steps 500 \
    --save_total_limit 20 \
    --max_steps 5000 \
    --evaluate_during_training \
    --logging_steps 500 \
    --line_by_line \
    --learning_rate 1e-4 \
    --block_size 512 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.001 \
    --beta1 0.9 \
    --beta2 0.98 \
    --mlm_probability 0.15 \
    --warmup_steps 2000 \
    $MODEL_ARG

  # Check model output
  if [ -f "$OUTPUT_CHUNK_DIR/pytorch_model.bin" ]; then
    echo "[INFO] Model saved in $OUTPUT_CHUNK_DIR"
    MODEL_PATH="$OUTPUT_CHUNK_DIR"
  else
    echo "[ERROR] Model not saved in $OUTPUT_CHUNK_DIR"
    exit 1
  fi

  # === Check Perplexity ===
  PERPLEXITY=$(python -c "
import json, math
try:
    state = json.load(open('$OUTPUT_CHUNK_DIR/trainer_state.json'))
    loss = next((log['eval_loss'] for log in reversed(state['log_history']) if 'eval_loss' in log), None)
    print(math.exp(loss) if loss is not None else 9999)
except Exception as e:
    print(9999)
")

  echo "[INFO] Perplexity after $(basename "$CHUNK"): $PERPLEXITY"

  if (( $(echo "$PERPLEXITY < $PERPLEXITY_THRESHOLD" | bc -l) )); then
    echo "[STOP] Stopping early: perplexity $PERPLEXITY is below threshold $PERPLEXITY_THRESHOLD"
    break
  fi
done

