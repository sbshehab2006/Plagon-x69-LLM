import os
import numpy as np
from tokenizers import Tokenizer

def tokenize_demo():
    txt_path = "data/cleaned/demo.txt"
    bin_path = "data/tokenized/demo.bin"
    tokenizer_path = "plagon_tokenizer.json"
    
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    
    print(f"Tokenizing {txt_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    encoded = tokenizer.encode(text)
    ids = np.array(encoded.ids, dtype=np.uint16)
    
    ids.tofile(bin_path)
    print(f"Saved {bin_path} ({len(ids)} tokens)")

if __name__ == "__main__":
    tokenize_demo()
