
import os
import numpy as np
from tokenizers import Tokenizer

def tokenize_stage1():
    txt_path = "data/cleaned/stage1_pretrain.txt"
    bin_dir = "data/tokenized_stage1"
    bin_path = os.path.join(bin_dir, "stage1.bin")
    tokenizer_path = "plagon_tokenizer.json"
    
    os.makedirs(bin_dir, exist_ok=True)
    
    print(f"Tokenizing {txt_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    encoded = tokenizer.encode(text)
    ids = np.array(encoded.ids, dtype=np.uint16)
    
    ids.tofile(bin_path)
    print(f"Saved {bin_path} ({len(ids)} tokens)")

if __name__ == "__main__":
    tokenize_stage1()
