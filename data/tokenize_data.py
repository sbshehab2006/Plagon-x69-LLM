import os
import glob
import numpy as np
from tokenizers import Tokenizer

DATA_CLEAN = "data/cleaned"
DATA_TOK = "data/tokenized"
TOKENIZER_FILE = "plagon_tokenizer.json"

os.makedirs(DATA_TOK, exist_ok=True)

def tokenize_file(filepath, tokenizer):
    print(f"Tokenizing {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Encode batch? No, simpler for now: just encode whole string
    # Warning: Huge files might crash RAM. For this "reasonable size" constraint it's fine.
    encoded = tokenizer.encode(text)
    ids = np.array(encoded.ids, dtype=np.uint16)
    
    base_name = os.path.basename(filepath).replace('.txt', '.bin')
    out_path = os.path.join(DATA_TOK, base_name)
    
    ids.tofile(out_path)
    print(f"Saved {out_path} ({len(ids)} tokens)")

def main():
    if not os.path.exists(TOKENIZER_FILE):
        print("Tokenizer not found! Run Phase 4 first.")
        return

    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    files = glob.glob(os.path.join(DATA_CLEAN, "*.txt"))
    
    if not files:
        print("No cleaned files found!")
        return

    for f in files:
        tokenize_file(f, tokenizer)
        
    print("Phase 5 (Tokenization) Complete.")

if __name__ == "__main__":
    main()
