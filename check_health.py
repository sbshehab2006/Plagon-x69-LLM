import torch
import os

def check_checkpoint(path):
    print(f"Checking {path}...")
    if not os.path.exists(path):
        print("❌ File not found.")
        return False
        
    try:
        torch.load(path, map_location="cpu", weights_only=False)
        print(f"✅ VALID: {path}")
        return True
    except Exception as e:
        print(f"❌ CORRUPTED: {path}")
        print(f"   Reason: {e}")
        return False

print("--- Checkpoint Health Check ---")
check_checkpoint("checkpoints/plagon_stage1.pt")
check_checkpoint("checkpoints/plagon_finetuned.pt")
check_checkpoint("checkpoints/plagon_pretrain.pt")
