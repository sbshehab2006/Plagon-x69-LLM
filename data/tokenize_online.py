import os
import numpy as np
from tokenizers import Tokenizer

def tokenize_online():
    # Input/Output paths
    txt_path = "data/cleaned/online_data.txt"
    bin_dir = "data/tokenized_online"
    bin_path = os.path.join(bin_dir, "online.bin")
    tokenizer_path = "plagon_tokenizer.json"
    
    # Create output directory
    os.makedirs(bin_dir, exist_ok=True)
    
    if not os.path.exists(txt_path):
        print(f"Error: {txt_path} not found.")
        return

    print(f"Tokenizing {txt_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    encoded = tokenizer.encode(text)
    ids = np.array(encoded.ids, dtype=np.uint16)
    
    ids.tofile(bin_path)
    print(f"Saved {bin_path} ({len(ids)} tokens)")

if __name__ == "__main__":
    tokenize_online()
