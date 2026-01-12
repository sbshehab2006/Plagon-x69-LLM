import torch
import torch.nn.functional as F

class ChatEngine:
    def __init__(self, model, tokenizer, system_prompt=None):
        self.model = model
        self.tokenizer = tokenizer
        self.history = []  # List of {"role": "user"/"assistant", "content": "..."}
        self.system_prompt = system_prompt
        self.device = next(model.parameters()).device

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})

    def generate_response(self, max_new_tokens=100, temperature=0.7):
        # 1. Build Prompt (Few-Shot Style for Stage 1 Model)
        # This structure teaches the model the pattern: User -> Assistant
        prompt = "Conversation between a User and an AI Assistant.\n\n"
        prompt += "User: Hi\nAssistant: Hello! How can I help you?\n\n"
        prompt += "User: What is your name?\nAssistant: I am Plagon, an AI assistant.\n\n"
        
        # Add recent history (Last 3 turns to keep context short and clean)
        recent_history = self.history[-6:]
        for msg in recent_history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role_label}: {msg['content']}\n"
            
        # The trigger for the model to answer
        prompt += "Assistant:"

        # 2. Encode
        input_ids = self.tokenizer.encode(prompt).ids
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # 3. Generate Loop with STOP LOGIC
        generated_ids = input_tensor.tolist()[0]
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Context Window (Last 128 tokens)
                ctx_tensor = torch.tensor([generated_ids[-128:]], device=self.device)
                
                logits, _ = self.model(ctx_tensor)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                
                # Sample
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated_ids.append(next_token)
                
                # IMMEDIATE STOP CHECK
                # Decode the last few tokens to see if we generated "User:"
                current_text = self.tokenizer.decode(generated_ids)
                new_content = current_text[len(prompt):]
                
                if "User:" in new_content or "\nUser" in new_content:
                    break
                if "Assistant:" in new_content: # Prevent self-loop
                    break

        # 4. Final Cleanup
        full_text = self.tokenizer.decode(generated_ids)
        response = full_text[len(prompt):]
        
        # Cut off any trailing garbage
        for stop in ["User:", "Assistant:", "\nUser", "User", "Human:"]:
            if stop in response:
                response = response.split(stop)[0]
                
        return response.strip()
