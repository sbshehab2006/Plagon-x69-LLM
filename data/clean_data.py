import os
import json
import glob
import re

DATA_RAW = "data/raw"
DATA_CLEANED = "data/cleaned"
os.makedirs(DATA_CLEANED, exist_ok=True)

def clean_text(text):
    # Basic normalization
    return text.replace('\r\n', '\n').strip()

def process_alpaca():
    src = os.path.join(DATA_RAW, "alpaca_data.json")
    if not os.path.exists(src): return
    
    print(f"Processing {src}...")
    with open(src, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    out_path = os.path.join(DATA_CLEANED, "english_instruct.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in data:
            text = f"{item['instruction']}\n{item['input']}\n{item['output']}\n"
            f.write(clean_text(text) + "\n<|endoftext|>\n")
    print(f"Saved {out_path}")

def process_bangla():
    src = os.path.join(DATA_RAW, "bangla_text.txt")
    if not os.path.exists(src): return

    print(f"Processing {src}...")
    with open(src, 'r', encoding='utf-8') as f:
        content = f.read()
    
    out_path = os.path.join(DATA_CLEANED, "bangla.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        # Just writing it as is, maybe split by lines
        f.write(clean_text(content))
    print(f"Saved {out_path}")

def is_valid_code_file(filepath):
    # Filter minified or huge files
    if filepath.endswith('.min.js') or filepath.endswith('.min.css'): return False
    try:
        if os.path.getsize(filepath) > 1024 * 500: return False # Skip > 500KB
        with open(filepath, 'r', encoding='utf-8') as f:
            if '\0' in f.read(1024): return False # Binary check
        return True
    except:
        return False

def process_code_repo(repo_dir, extensions, out_name):
    print(f"Processing repo {repo_dir} for {extensions}...")
    out_path = os.path.join(DATA_CLEANED, out_name)
    count = 0
    with open(out_path, 'w', encoding='utf-8') as outfile:
        for root, _, files in os.walk(repo_dir):
            if '.git' in root: continue
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    full_path = os.path.join(root, file)
                    if is_valid_code_file(full_path):
                        try:
                            with open(full_path, 'r', encoding='utf-8', errors='ignore') as infile:
                                outfile.write(infile.read() + "\n<|endoftext|>\n")
                                count += 1
                        except Exception:
                            pass
    print(f"Saved {out_path} ({count} files)")

if __name__ == "__main__":
    process_alpaca()
    process_bangla()
    process_code_repo(os.path.join(DATA_RAW, "flask_repo"), ['.py'], "python_code.txt")
    process_code_repo(os.path.join(DATA_RAW, "react_repo"), ['.js', '.jsx', '.ts', '.tsx'], "js_code.txt")
    print("Phase 3 (Cleaning) Complete.")
