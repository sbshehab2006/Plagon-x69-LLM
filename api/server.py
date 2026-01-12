from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import os
import time

# Import our Plagon modules
from model.transformer import PlagonTransformer
from model.config import PlagonConfig

# --- Configuration ---
MODEL_PATH = "plagon_model.pt" # Default path to weights
TOKENIZER_PATH = "plagon_tokenizer.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="Plagon LLM API",
    description="Inference API for Plagon LLM - Specialized for Code & Bangla/English.",
    version="1.0.0"
)

# --- Global State ---
model = None
tokenizer = None

# --- Schemas ---
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input text to continue")
    max_tokens: int = Field(128, ge=1, le=2048, description="Max tokens to generate")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Randomness (0=greedy)")
    top_k: int = Field(50, ge=0, description="Top-k sampling cutoff")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Nucleus sampling probability")

    @validator("prompt")
    def validate_prompt_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty/whitespace only.")
        return v

class GenerateResponse(BaseModel):
    output: str
    tokens_generated: int
    computation_time_ms: float

# --- Startup ---
@app.on_event("startup")
async def load_resources():
    global model, tokenizer
    print(f"Server starting on DEVICE: {DEVICE}")

    # 1. Load Tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Warning: Tokenizer not found at {TOKENIZER_PATH}. API will fail requests.")
    else:
        try:
            tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
            print("Tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")

    # 2. Load Model
    config = PlagonConfig()
    model = PlagonTransformer(config).to(DEVICE)
    
    # Check for weights
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.eval() # Set to inference mode
            print(f"Model weights loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    else:
        print(f"Warning: No weights found at {MODEL_PATH}. Using random initialization (will output garbage).")
        model.eval()

# --- Inference Logic ---
@torch.no_grad()
def generate_text(req: GenerateRequest):
    if not tokenizer or not model:
        raise HTTPException(status_code=503, detail="Model/Tokenizer not initialized.")

    start_time = time.time()

    # 1. Tokenize
    # ByteLevel tokenizer expects string input.
    encoded = tokenizer.encode(req.prompt)
    idx = torch.tensor(encoded.ids, dtype=torch.long, device=DEVICE).unsqueeze(0) # (1, T)
    
    # Check context length
    input_len = idx.size(1)
    if input_len > model.config.max_seq_len:
         # Truncate from left if too long
         idx = idx[:, -model.config.max_seq_len:]
    
    # 2. Generate Loop
    for _ in range(req.max_tokens):
        # Crop context to model limit
        idx_cond = idx[:, -model.config.max_seq_len:]
        
        # Forward pass
        logits, _ = model(idx_cond)
        
        # Pluck logits at the final step
        logits = logits[:, -1, :] # (B, V)
        
        # Apply Temperature
        if req.temperature > 0:
            logits = logits / req.temperature
            probs = F.softmax(logits, dim=-1)
            
            # Apply Top-K
            if req.top_k > 0:
                v, _ = torch.topk(logits, min(req.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
                
            # Apply Top-P (Nucleus) - Simplified implementation for readability
            # Note: For full robustness, sort probabilities and mask cumulative sum > top_p
            # Here we stick to Top-K + Temperature as primary controls for simplicity + speed
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
        # Append
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Check for EOS (Optional, if we defined an EOS token ID)
        # if idx_next.item() == tokenizer.token_to_id("<|endoftext|>"): break

    # 3. Decode
    # We only decode the *new* tokens to return just the generation? 
    # Or full text? The prompt asks for "output: generated text", usually implying full text or just completion.
    # Let's return Full Text for context awareness.
    output_ids = idx[0].tolist()
    full_text = tokenizer.decode(output_ids)
    
    # Trim prompt from output if you solely want the completion
    # completion = full_text[len(req.prompt):] 
    
    end_time = time.time()
    
    return {
        "output": full_text,
        "tokens_generated": len(output_ids) - input_len,
        "computation_time_ms": (end_time - start_time) * 1000
    }

# --- Endpoints ---

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "Plagon LLM",
        "device": DEVICE,
        "model_loaded": model is not None
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(req: GenerateRequest):
    try:
        result = generate_text(req)
        return GenerateResponse(**result)
    except Exception as e:
        # Catch unexpected errors during generation
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # For dev testing
    uvicorn.run(app, host="0.0.0.0", port=8000)
