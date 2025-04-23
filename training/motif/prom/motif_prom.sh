#!/bin/bash

# ----------------------
# Default parameter values
# ----------------------
KMER=4
DATA_PATH=../../examples/prom/ft_data
PREDICTION_PATH=../../examples/prom/predict_result
MOTIF_PATH=./result
WINDOW_SIZE=12
MIN_LEN=5
PVAL_CUTOFF=0.005
MIN_N_MOTIF=2

# ----------------------
# Help message
# ----------------------
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options (all are optional):"
    echo "  --window_size <int>      Sliding window size (default: $WINDOW_SIZE)"
    echo "  --min_len <int>          Minimum motif length (default: $MIN_LEN)"
    echo "  --min_n_motif <int>      Minimum number of motif instances (default: $MIN_N_MOTIF)"
    echo "  --data_path <path>       Input data path (default: $DATA_PATH)"
    echo "  --predict_path <path>    Prediction result path (default: $PREDICTION_PATH)"
    echo "  --motif_path <path>      Output motif result path (default: $MOTIF_PATH)"
    echo "  -h, --help               Show this help message and exit"
    echo ""
    echo "Example:"
    echo "  $0 --window_size 15 --min_len 6 --motif_path ./my_output"
}

# ----------------------
# Parse arguments
# ----------------------
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --window_size) WINDOW_SIZE="$2"; shift ;;
        --min_len) MIN_LEN="$2"; shift ;;
        --min_n_motif) MIN_N_MOTIF="$2"; shift ;;
        --data_path) DATA_PATH="$2"; shift ;;
        --predict_path) PREDICTION_PATH="$2"; shift ;;
        --motif_path) MOTIF_PATH="$2"; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "[ERROR] Unknown parameter: $1"; show_help; exit 1 ;;
    esac
    shift
done

# ----------------------
# Confirm parameters
# ----------------------
echo "[INFO] Running motif discovery with:"
echo "       KMER           = $KMER"
echo "       DATA_PATH      = $DATA_PATH"
echo "       PREDICT_PATH   = $PREDICTION_PATH"
echo "       MOTIF_PATH     = $MOTIF_PATH"
echo "       WINDOW_SIZE    = $WINDOW_SIZE"
echo "       MIN_LEN        = $MIN_LEN"
echo "       PVAL_CUTOFF    = $PVAL_CUTOFF"
echo "       MIN_N_MOTIF    = $MIN_N_MOTIF"
echo ""

# ----------------------
# Run motif discovery
# ----------------------
python ../find_motifs.py \
    --data_dir "$DATA_PATH" \
    --predict_dir "$PREDICTION_PATH" \
    --window_size "$WINDOW_SIZE" \
    --min_len "$MIN_LEN" \
    --pval_cutoff "$PVAL_CUTOFF" \
    --min_n_motif "$MIN_N_MOTIF" \
    --align_all_ties \
    --save_file_dir "$MOTIF_PATH" \
    --verbose

