import itertools

# Define the alphabet and special tokens
alphabet = "ABCDEFGHIJKLMNOPQR"
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
exclude_4mer = "RRRR"  # The 4-mer to exclude

# Output file
output_file = "4mer_vocabulary.txt"

# Generate all 4-mers
with open(output_file, "w") as f:
    # Write special tokens first
    for token in special_tokens:
        f.write(token + "\n")
    
    # Generate 4-mers and write to file, excluding RRRR
    for word in itertools.product(alphabet, repeat=4):
        word_str = "".join(word)
        if word_str != exclude_4mer:
            f.write(word_str + "\n")

print(f"Vocabulary saved to {output_file}")
