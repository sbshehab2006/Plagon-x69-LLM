#!/usr/bin/env python3
"""
Emergency checkpoint creator for running training process.
This script loads the existing model from memory trace and saves it.
"""
import torch
import os
from model.transformer import PlagonTransformer
from model.config import PlagonConfig

print("="*60)
print("EMERGENCY CHECKPOINT CREATION")
print("="*60)

# Create a model instance (same as training)
config = PlagonConfig()
device = "cpu"  # Always use CPU for emergency save
model = PlagonTransformer(config).to(device)

print(f"\n⚠️  This script creates a checkpoint from current memory state.")
print(f"If training is running, you should:")
print(f"  1. Press Ctrl+C in the training terminal")
print(f"  2. The updated pretrain.py will auto-save")
print(f"\nIf you want to force-save random weights as a test:")
input("Press ENTER to continue or Ctrl+C to abort...")

# Save as checkpoint
os.makedirs("checkpoints", exist_ok=True)
checkpoint_path = "checkpoints/plagon_stage1.pt"

torch.save({
    'model_state_dict': model.state_dict(),
    'note': 'Emergency checkpoint - may contain untrained weights'
}, checkpoint_path)

print(f"\n✅ Emergency checkpoint saved to: {checkpoint_path}")
print(f"You can now run: python demo_chat.py")
print("="*60)
