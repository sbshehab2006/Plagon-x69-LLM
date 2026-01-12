import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import time
import sys

# Add current directory to path so we can import 'model'
sys.path.append(os.getcwd())

# Import Plagon modules
from model.transformer import PlagonTransformer
from model.config import PlagonConfig

class PlagonPretrainDataset(Dataset):
    """
    Dataset loader for pretraining.
    Expects binary files (*.bin) in data/tokenized/ containing raw token IDs (uint16/int64).
    """
    def __init__(self, data_dir: str, block_size: int):
        self.block_size = block_size
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        self.data = []

        if not self.files:
            print(f"Warning: No .bin files found in {data_dir}. Using random dummy data for demonstration.")
            # Create dummy data: 1MB of tokens
            self.data = [np.random.randint(0, 50257, (1024 * 1024,), dtype=np.uint16)]
        else:
            for f in self.files:
                # Load whole file into memory (assuming fits in RAM, else use np.memmap)
                try:
                    # Assuming data was saved as standard numpy array or raw bytes
                    # Try numpy load first, simple binary read fallback
                    raw = np.fromfile(f, dtype=np.uint16)
                    self.data.append(raw)
                    print(f"Loaded {f}: {len(raw)} tokens")
                except Exception as e:
                    print(f"Error loading {f}: {e}")
        
        if self.data:
            self.data = np.concatenate(self.data)
        else:
            self.data = np.array([], dtype=np.uint16)

    def __len__(self):
        # We need block_size + 1 (for input and target)
        if len(self.data) <= self.block_size:
            return 0
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Grab chunk of length block_size + 1
        chunk = self.data[idx : idx + self.block_size + 1]
        
        # Convert to tensor int64 (Long)
        chunk_tensor = torch.from_numpy(chunk.astype(np.int64))
        
        x = chunk_tensor[:-1] # Input
        y = chunk_tensor[1:]  # Target (Input shifted by 1)
        return x, y

def save_checkpoint(model, optimizer, epoch, step, loss, path="plagon_checkpoint.pt"):
    print(f"Saving checkpoint to {path}...")
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def pretrain():
    # 1. Configuration
    config = PlagonConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Pretraining Plagon LLM on {device}")
    
    # 2. Model
    model = PlagonTransformer(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")
    
    # 3. Data
    # STAGE 1: Pretraining on clean conversational/book data
    data_dir = "data/tokenized_stage1"  # Directory containing stage1.bin
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} missing. Run data prep scripts first.")
        return
        
    dataset = PlagonPretrainDataset(data_dir, config.max_seq_len)
    
    if len(dataset) == 0:
        print("❌ ERROR: Dataset is empty! Check your data files.")
        return
    
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if device == "cuda" else False
    )

    # 4. Optimizer
    # Standard AdamW
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    # 5. Resume logic (SKIP - FORCE FRESH)
    start_epoch = 0
    # checkpoint_path = "checkpoints/plagon_pretrain.pt"
    # Force start from scratch for Stage 1 Correctness
    
    # 6. Training Loop (PRODUCTION-SAFE)
    model.train()
    
    os.makedirs("checkpoints", exist_ok=True)
    CHECKPOINT_PATH = os.path.join("checkpoints", "plagon_stage1.pt")
    
    # Step-based checkpoint configuration
    SAVE_EVERY_STEPS = 50  # Save every 50 steps (frequent for testing)
    total_steps = 0
    epochs = 3
    
    print(f"Starting Stage 1 Pretraining (Vocab: {config.vocab_size}, Seq: {config.max_seq_len})...")
    print(f"Checkpoints will be saved every {SAVE_EVERY_STEPS} steps to: {CHECKPOINT_PATH}")
    
    def emergency_save():
        """Force save current model state"""
        print(f"\n⚠️  EMERGENCY SAVE to {CHECKPOINT_PATH}...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_steps': total_steps,
        }, CHECKPOINT_PATH)
        print("✅ Checkpoint saved successfully!")
    
    # Register Ctrl+C handler
    import signal
    def signal_handler(sig, frame):
        print('\n\n🛑 Training interrupted by user!')
        emergency_save()
        print('Exiting...')
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        for epoch in range(epochs):
            print(f"\n🔄 Starting Epoch {epoch+1}/{epochs}")
            epoch_start_time = time.time()
            
            for batch_idx, (x, y) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)

                # Forward pass
                logits, loss = model(x, targets=y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_steps += 1

                # Logging every 50 steps
                if total_steps % 50 == 0:
                    elapsed = time.time() - epoch_start_time
                    print(f"Epoch {epoch+1} | Global Step {total_steps} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

                # CHECKPOINT SAVE (every N steps)
                if total_steps % SAVE_EVERY_STEPS == 0:
                    print(f"\n💾 Saving checkpoint at step {total_steps}...")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'total_steps': total_steps,
                        'epoch': epoch,
                    }, CHECKPOINT_PATH)
                    print(f"✅ Checkpoint saved!\n")

            # End of epoch save
            print(f"\n✅ Epoch {epoch+1} complete!")
            emergency_save()

        # Final save after all epochs
        print(f"\n🎉 Training Complete!")
        emergency_save()
        
    except KeyboardInterrupt:
        print('\n\n🛑 Training interrupted!')
        emergency_save()
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        emergency_save()

if __name__ == "__main__":
    pretrain()
