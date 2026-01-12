import sys
import os
sys.path.append(os.getcwd())
import torch
import time

print("Debug: Starting imports...")
try:
    from model.transformer import PlagonTransformer
    from model.config import PlagonConfig
    print("Debug: Imports successful.")
except Exception as e:
    print(f"Debug: Import failed: {e}")
    sys.exit(1)

def run():
    print("Debug: Initializing Config...")
    config = PlagonConfig()
    print("Debug: Initializing Model...")
    model = PlagonTransformer(config)
    print("Debug: Model initialized.")
    
    path = "checkpoints/test.pt"
    os.makedirs("checkpoints", exist_ok=True)
    print(f"Debug: Saving to {path}...")
    torch.save(model.state_dict(), path)
    print("Debug: Save successful.")

if __name__ == "__main__":
    run()
