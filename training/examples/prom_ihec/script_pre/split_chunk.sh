#!/bin/bash

# Settings
KMER=4
INPUT_FILE=../pretrain_data/promoter_ihec_all_4mer_wo_4R.txt
OUTPUT_DIR=../pretrain_data/split_chunks
CHUNK_SIZE=100000  # lines per chunk

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Temporary shuffled file path
SHUFFLED_FILE="$OUTPUT_DIR/shuffled_input.txt"

# Shuffle the input file
echo "[INFO] Shuffling input file: $INPUT_FILE"
shuf "$INPUT_FILE" > "$SHUFFLED_FILE"

# Split the shuffled file into chunks
echo "[INFO] Splitting shuffled file into chunks of $CHUNK_SIZE lines..."
split -l $CHUNK_SIZE --numeric-suffixes=1 --suffix-length=3 "$SHUFFLED_FILE" "$OUTPUT_DIR/chunk_"

# Clean up shuffled file (optional)
rm "$SHUFFLED_FILE"

# Done
echo "[INFO] Split complete. Chunks saved to: $OUTPUT_DIR"

