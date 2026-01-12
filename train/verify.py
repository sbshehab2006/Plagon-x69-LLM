import sys
import os
sys.path.append(os.getcwd())
import torch
from model.transformer import PlagonTransformer
from model.config import PlagonConfig
from chat.chat_engine import ChatEngine
from tokenizers import Tokenizer

def check():
    path = "checkpoints/plagon_pretrain.pt"
    if not os.path.exists(path):
        print(f"FAILED: {path} not found")
        return

    print(f"Loading {path}...")
    config = PlagonConfig()
    model = PlagonTransformer(config)
    
    # Load state dict
    ckpt = torch.load(path, map_location="cpu")
    # Check if full checkpoint or just state_dict
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("Model loaded successfully!")
    
    # Try generation
    try:
        tokenizer = Tokenizer.from_file("plagon_tokenizer.json")
        chat = ChatEngine(model, tokenizer)
        print("Generating test response...")
        chat.add_message("user", "Hello")
        response = chat.generate_response(max_new_tokens=20)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Generation warning: {e}")

if __name__ == "__main__":
    check()
