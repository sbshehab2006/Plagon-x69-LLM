import requests
import re
import os
import time
import random

TARGET_FILE = "data/cleaned/online_data.txt"
MIN_SIZE = 5 * 1024 * 1024 # 5 MB

def get_wikipedia_text(lang, limit=10):
    # Fetch random titles
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnlimit": limit,
        "rnnamespace": 0
    }
    headers = {
        'User-Agent': 'PlagonDataCollector/1.0 (contact@example.com)'
    }
    try:
        r = requests.get(url, params=params, headers=headers).json()
        titles = [p['title'] for p in r['query']['random']]
        
        texts = []
        for title in titles:
            # Get content
            p_params = {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": True,
                "titles": title
            }
            pr = requests.get(url, params=p_params, headers=headers).json()
            pages = pr['query']['pages']
            for pid in pages:
                if 'extract' in pages[pid]:
                    texts.append(pages[pid]['extract'])
        return texts
    except Exception as e:
        print(f"Error fetching {lang}: {e}")
        return []

def clean_text(text):
    # Remove headings (== ... ==)
    text = re.sub(r'==.*?==', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Split sentences 
    # Unicode safe split for Bangla (।) and English (.!?)
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    
    clean_sentences = []
    for s in sentences:
        s = s.strip()
        # Filters
        if len(s) < 20: continue # too short
        if len(s) > 500: continue # too long, likely garbage list
        if '{' in s or '}' in s or '[' in s or ']' in s: continue # markup
        if '<' in s or '>' in s: continue # HTML
        if 'http' in s or 'www' in s: continue # URL
        if s.count(',') > 10: continue # List check
        
        clean_sentences.append(s)
    return clean_sentences

def main():
    os.makedirs(os.path.dirname(TARGET_FILE), exist_ok=True)
    
    total_bytes = 0
    # Clear file first to ensure fresh start
    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.write("")

    print("Starting data collection...")
    
    with open(TARGET_FILE, "a", encoding="utf-8") as f:
        while total_bytes < MIN_SIZE:
            current_mb = total_bytes / (1024 * 1024)
            print(f"Size: {current_mb:.2f} MB / 5.00 MB")
            
            # Fetch English (Simple Wiki if possible, but en wiki is fine)
            # Using 'simple' prefix for English Wikipedia for simpler text
            texts = get_wikipedia_text("simple", limit=15) 
            for text in texts:
                sentences = clean_text(text)
                for s in sentences:
                    f.write(s + "\n")
            
            # Fetch Bangla
            texts_bn = get_wikipedia_text("bn", limit=15)
            for text in texts_bn:
                sentences = clean_text(text)
                for s in sentences:
                    f.write(s + "\n")
            
            f.flush()
            total_bytes = os.path.getsize(TARGET_FILE)
            time.sleep(1)

    print(f"Collection Complete. Final Size: {total_bytes/(1024*1024):.2f} MB")

    # Validation
    print("\n--- VALIDATION: Random 20 lines ---")
    with open(TARGET_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            print("Error: File empty!")
            return
        
        sample = random.sample(lines, min(20, len(lines)))
        for line in sample:
            print(line.strip())

if __name__ == "__main__":
    main()
