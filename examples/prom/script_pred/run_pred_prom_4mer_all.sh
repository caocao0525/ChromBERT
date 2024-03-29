#### Epi all (57 different epigenomes that RNA-seq available)

export KMER=4
export MODEL_BASE_PATH=../ft_result/up2kdown4k/gene_exp/4mer/all
export DATA_BASE_PATH=../../../../chromatin_state/database/fine_tune/prom/up2kdown4k/gene_exp/4mer/all
export PREDICTION_BASE_PATH=../predict_result/up2kdown4k/gene_exp/4mer/all


declare -a arr=("not_n_rpkm0" "not_n_rpkm30" "rpkm0_n_rpkm20" "rpkm10_n_rpkm20" "rpkm20_n_rpkm30"
                "not_n_rpkm10" "not_n_rpkm50" "rpkm0_n_rpkm30" "rpkm10_n_rpkm30" "rpkm20_n_rpkm50"
                "not_n_rpkm20" "rpkm0_n_rpkm10" "rpkm0_n_rpkm50" "rpkm10_n_rpkm50" "rpkm30_n_rpkm50")

for i in "${arr[@]}"; do
    export MODEL_PATH="${MODEL_BASE_PATH}/${i}"
    export DATA_PATH="${DATA_BASE_PATH}/${i}"
    export PREDICTION_PATH="${PREDICTION_BASE_PATH}/${i}"

    python ../../run_finetune.py \
        --model_type dna \
        --tokenizer_name=dna$KMER \
        --model_name_or_path $MODEL_PATH \
        --task_name dnaprom \
        --do_predict \
        --data_dir $DATA_PATH \
        --max_seq_length 300 \
        --per_gpu_pred_batch_size=128   \
        --output_dir $MODEL_PATH \
        --predict_dir $PREDICTION_PATH \
        --n_process 48
done
