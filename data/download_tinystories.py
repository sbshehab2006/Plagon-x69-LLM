import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    print(f"Downloading {filename}...")
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def prepare_tinystories():
    data_dir = "data/raw"
    os.makedirs(data_dir, exist_ok=True)
    
    # Official TinyStories Dataset URL (HuggingFace)
    # We download valid.txt for quicker experiment (approx 20MB) 
    # Or train.txt for full training (approx 250MB+)
    # Let's start with valid.txt (Verification Split) which is large enough (~18MB) for a good start.
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
    file_path = os.path.join(data_dir, "TinyStories_valid.txt")
    
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping download.")
    else:
        download_file(url, file_path)
    
    # Now we clean it and put it in cleaned folder
    clean_dir = "data/cleaned"
    os.makedirs(clean_dir, exist_ok=True)
    clean_path = os.path.join(clean_dir, "tinystories_clean.txt")
    
    print("Cleaning and preparing data...")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Basic cleanup (TinyStories is already quite clean)
    text = text.replace("<|endoftext|>", "") # Remove special tokens if any
    
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(text)
        
    print(f"✅ Data prepared at: {clean_path}")
    print(f"Total Size: {len(text)/1024/1024:.2f} MB")

if __name__ == "__main__":
    prepare_tinystories()
