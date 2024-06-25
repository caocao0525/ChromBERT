export KMER=4
export MODEL_PATH=../pretrain_result
export DATA_PATH=../ft_data
export OUTPUT_PATH=../ft_result

python ../../run_finetune.py \
    --model_type dna \
    --tokenizer_name="dna$KMER" \
    --model_name_or_path "$MODEL_PATH" \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --data_dir "$DATA_PATH" \
    --max_seq_length 300 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_train_batch_size=32 \
    --learning_rate 2e-4 \
    --num_train_epochs 10.0 \
    --output_dir "$OUTPUT_PATH" \
    --evaluate_during_training \
    --logging_steps 100 \
    --save_steps 4000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 8
