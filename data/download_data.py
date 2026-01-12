import os
import subprocess
import requests

DATA_RAW = "data/raw"
os.makedirs(DATA_RAW, exist_ok=True)

def download_file(url, filename):
    print(f"Downloading {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(os.path.join(DATA_RAW, filename), 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {filename}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def git_clone(repo_url, dir_name):
    target = os.path.join(DATA_RAW, dir_name)
    if os.path.exists(target):
        print(f"{target} already exists, skipping clone.")
        return
    print(f"Cloning {repo_url}...")
    try:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, target], check=True)
        print(f"Cloned {dir_name}")
    except Exception as e:
        print(f"Failed to clone {repo_url}: {e}")

if __name__ == "__main__":
    # 1. English Instruction Data (Alpaca)
    download_file(
        "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json",
        "alpaca_data.json"
    )

    # 2. Bangla Text (Review/Sentiment data is good for raw text)
    download_file(
        "https://raw.githubusercontent.com/minar09/bangla-sentiment-analysis/master/data/data.txt",
        "bangla_text.txt"
    )

    # 3. Code (Flask - Python)
    git_clone("https://github.com/pallets/flask", "flask_repo")
    
    # 4. Code (React - JS)
    git_clone("https://github.com/facebook/react", "react_repo")

    print("Phase 2 (Collection) Complete.")
