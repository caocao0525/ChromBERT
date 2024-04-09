export KMER=4
export DATA_BASE_PATH=../../../chromatin_state/database/fine_tune/prom/up2kdown4k/gene_exp/4mer/all
export PREDICTION_BASE_PATH=../../examples/prom/predict_result/up2kdown4k/gene_exp/4mer/all
export MOTIF_BASE_PATH=./result/up2kdown4k/gene_exp/4mer/all

declare -a arr=("not_n_rpkm0" "not_n_rpkm30" "rpkm0_n_rpkm20" "rpkm10_n_rpkm20" "rpkm20_n_rpkm30"
                "not_n_rpkm10" "not_n_rpkm50" "rpkm0_n_rpkm30" "rpkm10_n_rpkm30" "rpkm20_n_rpkm50"
                "not_n_rpkm20" "rpkm0_n_rpkm10" "rpkm0_n_rpkm50" "rpkm10_n_rpkm50" "rpkm30_n_rpkm50")

for i in "${arr[@]}"; do
    export DATA_PATH="${DATA_BASE_PATH}/${i}"
    export PREDICTION_PATH="${PREDICTION_BASE_PATH}/${i}"
    export MOTIF_PATH="${MOTIF_BASE_PATH}/${i}"

    python ../find_motifs.py \
        --data_dir $DATA_PATH \
        --predict_dir $PREDICTION_PATH \
        --window_size 12 \
        --min_len 5 \
        --pval_cutoff 0.005 \
        --min_n_motif 2 \
        --align_all_ties \
        --save_file_dir $MOTIF_PATH \
        --verbose
done
