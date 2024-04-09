export KMER=4
export DATA_PATH=../../examples/prom/ft_data
export PREDICTION_PATH=../../examples/prom/predict_result
export MOTIF_PATH=./result

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
