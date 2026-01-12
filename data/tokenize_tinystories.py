from tokenizers import Tokenizer
import os
import numpy as np

def tokenize_data():
    input_file = "data/cleaned/tinystories_clean.txt"
    output_dir = "data/tokenized_stage1" # We overwrite stage 1 data with this high quality data
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "stage1.bin")

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run download_tinystories.py first.")
        return

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file("plagon_tokenizer.json")

    print(f"Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    print("Tokenizing (this may take a moment)...")
    encoded = tokenizer.encode(text)
    ids = np.array(encoded.ids, dtype=np.uint16)

    print(f"Writing {len(ids)} tokens to {output_file}...")
    ids.tofile(output_file)
    
    print(f"✅ Tokenization Complete! Tokens: {len(ids)}")
    print("You are now ready for REAL Deep Pretraining.")

if __name__ == "__main__":
    tokenize_data()
