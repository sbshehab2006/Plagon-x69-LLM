
from tokenizers import ByteLevelBPETokenizer
import os

def train_tokenizer_stage1():
    # Strict matching: Train ONLY on Stage 1 data
    files = ["data/cleaned/stage1_pretrain.txt"]
    vocab_size = 16000 # Small efficient vocab for 10MB data
    
    print(f"Training Tokenizer on: {files}")
    
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=files, vocab_size=vocab_size, min_frequency=2, special_tokens=[
        "<|endoftext|>",
        "<|padding|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>"
    ])
    
    save_path = "plagon_tokenizer.json"
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path} (Vocab: {vocab_size})")

if __name__ == "__main__":
    train_tokenizer_stage1()
