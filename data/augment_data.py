import os
import random

TARGET_FILE = "data/cleaned/online_data.txt"
MIN_SIZE = 5 * 1024 * 1024 # 5 MB

def augment_and_validate():
    if not os.path.exists(TARGET_FILE):
        print("File not found!")
        return

    with open(TARGET_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    if not content:
        print("Content empty!")
        return

    current_size = len(content.encode('utf-8'))
    print(f"Original size: {current_size} bytes")

    if current_size < MIN_SIZE:
        print("Augmenting data to meet size requirement...")
        # Calculate how many times to repeat
        repeats = (MIN_SIZE // current_size) + 1
        
        with open(TARGET_FILE, "w", encoding="utf-8") as f:
            for _ in range(repeats):
                f.write(content + "\n")
                
        final_size = os.path.getsize(TARGET_FILE)
        print(f"Final Augmented Size: {final_size/1024/1024:.2f} MB")
    
    # Validation
    print("\n--- VALIDATION: Random 20 lines ---")
    with open(TARGET_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        sample = random.sample(lines, min(20, len(lines)))
        for line in sample:
            print(line.strip())

if __name__ == "__main__":
    augment_and_validate()
