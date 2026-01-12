import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from model.transformer import PlagonTransformer
from model.config import PlagonConfig
from train.pretrain import PlagonPretrainDataset

def finetune():
    # 1. Configuration
    config = PlagonConfig()
    config.learning_rate = 1e-5 # Lower LR for finetuning
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Finetuning Plagon LLM on {device}...")
    
    # 2. Model Setup
    model = PlagonTransformer(config).to(device)
    
    # 3. Load Stage 1 Weights
    PRETRAIN_PATH = os.path.join("checkpoints", "plagon_stage1.pt")
    if not os.path.exists(PRETRAIN_PATH):
        print(f"FAILED: Stage 1 checkpoint not found at {PRETRAIN_PATH}")
        return

    print(f"Loading weights from {PRETRAIN_PATH}...")
    checkpoint = torch.load(PRETRAIN_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("✅ Stage 1 Weights loaded.")

    # 4. Generate ROBUST Synthetic Instruction Data (Stage 2)
    print("Generating robust instruction dataset (Chat/Logic/Facts)...")
    tokenizer = None
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file("plagon_tokenizer.json")
    except:
        print("Tokenizer not found!")
        return

    # Base list of diverse instructions
    base_instructions = [
        # GREETINGS
        ("Hi", "Hello! How can I help you?"),
        ("Hello", "Hi there!"),
        ("Hey", "Hello! Nice to meet you."),
        ("Good morning", "Good morning!"),
        ("Good afternoon", "Good afternoon!"),
        ("Good evening", "Good evening!"),
        ("How are you?", "I am doing well, thank you."),
        ("What's up?", "Not much, just here to help you."),
        
        # IDENTITY
        ("Who are you?", "I am Plagon, an AI assistant."),
        ("What is your name?", "My name is Plagon."),
        ("Are you human?", "No, I am an Artificial Intelligence."),
        ("Who made you?", "I was trained by you."),
        
        # FACTS And KNOWLEDGE
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the sun?", "The sun is a star at the center of our solar system."),
        ("What is water?", "Water is a clear liquid essential for life."),
        ("What is AI?", "AI stands for Artificial Intelligence."),
        ("Tell me a fact", "The earth revolves around the sun."),
        
        # SIMPLE REASONING
        ("What is 2+2?", "The answer is 4."),
        ("What is 5+5?", "The answer is 10."),
        ("Is fire hot?", "Yes, fire is hot."),
        ("Is ice hot?", "No, ice is cold."),
        ("Which is bigger, a cat or an elephant?", "An elephant is bigger than a cat."),
        
        # OTHERS
        ("Tell me a joke", "Why did the chicken cross the road? To get to the other side!"),
        ("Thank you", "You are welcome!"),
        ("Bye", "Goodbye! Have a nice day."),
        ("See you", "See you later!"),
        ("Help", "How can I help you today?")
    ]
    
    # Expand dataset to ~2500 samples by repetition and slight variations
    # (In a real scenario, we would load an external dataset like Alpaca)
    instructions = base_instructions * 100 
    
    data_ids = []
    
    for user, assistant in instructions:
        # Chat Format: User: ...\nAssistant: ...
        text = f"User: {user}\nAssistant: {assistant}\n"
        ids = tokenizer.encode(text).ids
        # Pad or truncate to max_seq_len (128)
        if len(ids) > config.max_seq_len:
            ids = ids[:config.max_seq_len]
        data_ids.append(ids)

    print(f"Prepared {len(data_ids)} instruction samples.")
    
    # 5. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-5) # Low LR to preserve knowledge
    
    # 6. Training Loop
    SAVE_EVERY_STEPS = 100
    total_steps = 0
    epochs = 5  # 5 Epochs for solid learning
    
    print(f"\nStarting Stage 2 Chat Training (Target: {epochs} Epochs)...")
    
    # Ensure checkpoints dir exists
    os.makedirs("checkpoints", exist_ok=True)
    CHECKPOINT_PATH = os.path.join("checkpoints", "plagon_finetuned.pt")
    
    model.train()
    
    try:
        for epoch in range(epochs):
            print(f"\n🔄 Finetuning Epoch {epoch+1}/{epochs}")
            total_loss = 0
            
            for i, ids in enumerate(data_ids):
                if len(ids) < 2: continue
                
                headers = torch.tensor([ids], dtype=torch.long, device=device)
                x = headers[:, :-1]
                y = headers[:, 1:]
                
                logits, loss = model(x, targets=y)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_steps += 1
                
                # Logging
                if total_steps % 50 == 0:
                    print(f"Epoch {epoch+1} | Step {total_steps} | Loss: {loss.item():.4f}")
            
                # SAVE checkpoint
                if total_steps % SAVE_EVERY_STEPS == 0:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'stage': 2
                    }, CHECKPOINT_PATH)

            avg_loss = total_loss / len(data_ids)
            print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
            print(f"💾 Saved checkpoint to {CHECKPOINT_PATH}")
            
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted! Saving state...")
        torch.save(model.state_dict(), CHECKPOINT_PATH)

    print("✅ Stage 2 Training Complete!")
    print("You can now run 'python demo_chat.py'")

if __name__ == "__main__":
    finetune()
