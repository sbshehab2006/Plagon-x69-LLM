import torch
from tokenizers import Tokenizer
from model.transformer import PlagonTransformer
from model.config import PlagonConfig
from chat.chat_engine import ChatEngine
import os

def run_chat():
    print("Initializing Plagon 69 Chat System...")
    
    # 1. Load Resources
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check for artifacts (Assuming we are in project root)
    if not os.path.exists("plagon_tokenizer.json"):
        print("Error: plagon_tokenizer.json not found. Run tokenizer/train_tokenizer.py first.")
        return

    tokenizer = Tokenizer.from_file("plagon_tokenizer.json")
    config = PlagonConfig()
    model = PlagonTransformer(config).to(device)
    
    # Load weights - STRICT REQUIREMENT
    # REVERTING TO STAGE 1 (More stable for now)
    checkpoint_paths = [
        "checkpoints/plagon_stage1.pt",      # Stage 1 (Stable)
        "checkpoints/plagon_finetuned.pt",   # Stage 2 (Unstable/Experimental)
        "checkpoints/plagon_pretrain.pt"     # Legacy
    ]
    
    checkpoint_loaded = None
    for checkpoint_path in checkpoint_paths:
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location=device)
            
            # Handle both dict with 'model_state_dict' key and direct state_dict
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
            
            checkpoint_loaded = checkpoint_path
            print(f"✅ Model loaded successfully from: {checkpoint_path}\n")
            break
    
    if checkpoint_loaded is None:
        print("\n" + "="*60)
        print("❌ ERROR: NO TRAINED MODEL FOUND")
        print("="*60)
        print("\nThe model requires a trained checkpoint to run.")
        print("Please train the model first by running:")
        print("  python train/pretrain.py")
        print("\nExpected checkpoint location:")
        for path in checkpoint_paths:
            print(f"  - {path}")
        print("\n" + "="*60)
        return

    # 2. Init Engine
    chat = ChatEngine(model, tokenizer)
    
    # 3. Chat Loop
    print("\n--- Plagon 69 CLI (Type 'quit' to exit) ---\n")
    
    # Optional: Set System Prompt from file
    from chat.system_prompt import SYSTEM_PROMPT
    chat.add_message("system", SYSTEM_PROMPT)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            break
            
        # Add User Message
        chat.add_message("user", user_input)
        
        print("Plagon:", end=" ", flush=True)
        # Generate
        response = chat.generate_response(max_new_tokens=100, temperature=0.7)
        print(response)

if __name__ == "__main__":
    run_chat()
