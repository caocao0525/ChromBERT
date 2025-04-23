#!/bin/bash

# Default settings
KMER=4
CHUNK_SIZE=100000  # lines per chunk
DEFAULT_INPUT="../pretrain_data/promoter_ihec_all_4mer_wo_4R.txt"
DEFAULT_OUTPUT="../pretrain_data/split_chunks"

# Help message
usage() {
  echo "Usage: $0 [-i INPUT_FILE] [-o OUTPUT_DIR] [-c CHUNK_SIZE]"
  echo ""
  echo "  -i   Input file to shuffle and split (default: $DEFAULT_INPUT)"
  echo "  -o   Output directory to store chunks (default: $DEFAULT_OUTPUT)"
  echo "  -c   Number of lines per chunk (default: $CHUNK_SIZE)"
  echo "  -h   Show this help message"
  exit 1
}

# Parse arguments
while getopts "i:o:c:h" opt; do
  case ${opt} in
    i ) INPUT_FILE=$OPTARG ;;
    o ) OUTPUT_DIR=$OPTARG ;;
    c ) CHUNK_SIZE=$OPTARG ;;
    h ) usage ;;
    * ) usage ;;
  esac
done

# Set defaults if not provided
INPUT_FILE=${INPUT_FILE:-$DEFAULT_INPUT}
OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT}

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
  echo "[ERROR] Input file not found: $INPUT_FILE"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Temp file for shuffled content
SHUFFLED_FILE="$OUTPUT_DIR/shuffled_input.txt"

echo "[INFO] Shuffling input file: $INPUT_FILE"
shuf "$INPUT_FILE" > "$SHUFFLED_FILE"

echo "[INFO] Splitting shuffled file into chunks of $CHUNK_SIZE lines..."
split -l "$CHUNK_SIZE" --numeric-suffixes=1 --suffix-length=3 "$SHUFFLED_FILE" "$OUTPUT_DIR/chunk_"

rm "$SHUFFLED_FILE"

echo "[INFO] Split complete. Chunks saved to: $OUTPUT_DIR"

