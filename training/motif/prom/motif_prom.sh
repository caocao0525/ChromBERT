# Default values
KMER=4
DATA_PATH=../../examples/prom/ft_data
PREDICTION_PATH=../../examples/prom/predict_result
MOTIF_PATH=./result
WINDOW_SIZE=12
MIN_LEN=5
PVAL_CUTOFF=0.005
MIN_N_MOTIF=2

# Function to display help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --window_size <value>    Set the window size (default: $WINDOW_SIZE)"
    echo "  --min_len <value>        Set the minimum length (default: $MIN_LEN)"
    echo "  --min_n_motif <value>    Set the minimum number of motifs (default: $MIN_N_MOTIF)"
    echo "  -h, --help               Show this help message and exit"
    echo ""
    echo "Example:"
    echo "  $0 --window_size 12 --min_len 5 --min_n_motif 2"
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --window_size) WINDOW_SIZE="$2"; shift ;;
        --min_len) MIN_LEN="$2"; shift ;;
        --min_n_motif) MIN_N_MOTIF="$2"; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Execute the Python script with the specified arguments
python ../find_motifs.py \
    --data_dir $DATA_PATH \
    --predict_dir $PREDICTION_PATH \
    --window_size $WINDOW_SIZE \
    --min_len $MIN_LEN \
    --pval_cutoff $PVAL_CUTOFF \
    --min_n_motif $MIN_N_MOTIF \
    --align_all_ties \
    --save_file_dir $MOTIF_PATH \
    --verbose




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
