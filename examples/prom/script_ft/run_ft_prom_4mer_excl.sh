#### Epi all (57 different epigenomes that RNA-seq available)

export KMER=4
export MODEL_PATH=../pretrain_result/4mer/all_epi
export DATA_BASE_PATH=../../../../chromatin_state/database/fine_tune/prom/up2kdown4k/gene_exp/4mer/excl
export OUTPUT_BASE_PATH=../ft_result/up2kdown4k/gene_exp/4mer/excl

declare -a arr=("not_n_rpkm0" "not_n_rpkm30" "rpkm0_n_rpkm20" "rpkm10_n_rpkm20" "rpkm20_n_rpkm30"
                "not_n_rpkm10" "not_n_rpkm50" "rpkm0_n_rpkm30" "rpkm10_n_rpkm30" "rpkm20_n_rpkm50"
                "not_n_rpkm20" "rpkm0_n_rpkm10" "rpkm0_n_rpkm50" "rpkm10_n_rpkm50" "rpkm30_n_rpkm50")

for i in "${arr[@]}"; do
    export DATA_PATH="${DATA_BASE_PATH}/${i}"
    export OUTPUT_PATH="${OUTPUT_BASE_PATH}/${i}"

    python ../../run_finetune.py \
        --model_type dna \
        --tokenizer_name=dna$KMER \
        --model_name_or_path $MODEL_PATH \
        --task_name dnaprom \
        --do_train \
        --do_eval \
        --data_dir $DATA_PATH \
        --max_seq_length 300 \
        --per_gpu_eval_batch_size=32   \
        --per_gpu_train_batch_size=32   \
        --learning_rate 2e-4 \
        --num_train_epochs 10.0 \
        --output_dir $OUTPUT_PATH \
        --evaluate_during_training \
        --logging_steps 100 \
        --save_steps 4000 \
        --warmup_percent 0.1 \
        --hidden_dropout_prob 0.1 \
        --overwrite_output \
        --weight_decay 0.01 \
        --n_process 8
done
