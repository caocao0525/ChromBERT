# Defaults
DEFAULT_KMER=4
DEFAULT_MODEL_PATH="../ft_result/classification"
DEFAULT_DATA_PATH="../ft_data/classification"
DEFAULT_PREDICTION_PATH="../predict_result"

# Use arguments if provided, otherwise use defaults
KMER=${1:-$DEFAULT_KMER}
MODEL_PATH=${2:-$DEFAULT_MODEL_PATH}
DATA_PATH=${3:-$DEFAULT_DATA_PATH}
PREDICTION_PATH=${4:-$DEFAULT_PREDICTION_PATH}

echo "Running prediction with:"
echo "  KMER = $KMER"
echo "  MODEL_PATH = $MODEL_PATH"
echo "  DATA_PATH = $DATA_PATH"
echo "  PREDICTION_PATH = $PREDICTION_PATH"

# Run script
python ../../run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_visualize \
    --visualize_data_dir $DATA_PATH \
    --visualize_models $KMER \
    --data_dir $DATA_PATH \
    --max_seq_length 81 \
    --per_gpu_pred_batch_size=16 \
    --output_dir $MODEL_PATH \
    --predict_dir $PREDICTION_PATH \
    --n_process 96
