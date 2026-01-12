import glob
import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from tokenizers.implementations import ByteLevelBPETokenizer

def train_tokenizer(
    data_path: str = "data/cleaned/*.txt",
    save_path: str = "plagon_tokenizer.json",
    vocab_size: int = 50257
):
    """
    Trains a Byte-Level BPE Tokenizer for Plagon LLM.
    
    Why Byte-Level BPE?
    1. **Code:** Handles significant whitespace and rare symbols typical in programming languages.
    2. **Multilingual (Bangla):** Byte fallback ensures we never hit <unk> for unseen unicode characters.
       Bangla characters will be decomposed into bytes if not explicitly in vocab, preserving information.
    3. **Compatibility:** Matches GPT-2/GPT-3/Llama style tokenization.
    
    Why Vocab Size 50,257?
    - Standard GPT-2 size.
    - Large enough to cover common English words, Bangla subwords, and Code keywords.
    - Small enough to keep the embedding layer efficient.
    """
    
    print(f"Initializing ByteLevelBPETokenizer...")
    
    # 1. Initialize
    # We use Hugging Face's ByteLevelBPETokenizer wrapper for simplicity
    tokenizer = ByteLevelBPETokenizer()
    
    # 2. Collect Data
    files = glob.glob(data_path)
    if not files:
        print(f"Warning: No files found at {data_path}")
        # Create a dummy file for demonstration so the script runs without error
        dummy_dir = os.path.dirname(data_path)
        os.makedirs(dummy_dir, exist_ok=True)
        dummy_file = os.path.join(dummy_dir, "sample_corpus.txt")
        with open(dummy_file, "w", encoding="utf-8") as f:
            f.write("def hello_world():\n    print('Hello World')\n\nআমার নাম প্লাগন। আমি কোড লিখতে পারি।")
        files = [dummy_file]
        print(f"Created dummy file: {dummy_file}")
    
    print(f"Found {len(files)} files to train on.")
    
    # 3. Train
    print(f"Training tokenizer (vocab_size={vocab_size})...")
    tokenizer.train(
        files,
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=[
            "<|endoftext|>",  # Document separator
            "<|padding|>",    # Padding token
        ]
    )
    
    # 4. Save
    print(f"Saving tokenizer to {save_path}...")
    tokenizer.save(save_path)
    print("Done!")

if __name__ == "__main__":
    train_tokenizer()
