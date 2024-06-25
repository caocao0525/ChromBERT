export KMER=4
export MODEL_PATH=../ft_result
export DATA_PATH=../ft_data
export PREDICTION_PATH=../predict_result

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
        --per_gpu_pred_batch_size=16   \
        --output_dir $MODEL_PATH \
        --predict_dir $PREDICTION_PATH \
        --n_process 96 
