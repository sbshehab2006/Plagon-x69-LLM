import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import PlagonConfig

class CausalSelfAttention(nn.Module):
    def __init__(self, config: PlagonConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.n_heads = config.n_heads
        
        # Projections for Query, Key, Value
        self.c_attn = nn.Linear(config.dim, 3 * config.dim)
        # Output projection
        self.c_proj = nn.Linear(config.dim, config.dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Causal mask: ensures token at pos t can only attend to t and before
        self.register_buffer("bias", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                                     .view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x):
        B, T, C = x.size() # Batch, Time, Channels
        
        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape for multi-head: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask (set future positions to -inf)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v # Aggregate values
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble
        
        return self.c_proj(y)

class MLP(nn.Module):
    """Feed-Forward Network"""
    def __init__(self, config: PlagonConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.dim, 4 * config.dim)
        self.act     = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, config: PlagonConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LayerNorm architecture (standard for GPT-2/3)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class PlagonTransformer(nn.Module):
    """
    Plagon LLM: A decoder-only Transformer.
    """
    def __init__(self, config: PlagonConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.dim),  # Token embeddings
            wpe = nn.Embedding(config.max_seq_len, config.dim), # Position embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.dim),
        ))
        
        # Language Model Head
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight tying (improve performance & reduce params)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        if t > self.config.max_seq_len:
            raise ValueError(f"Sequence length {t} exceeds limit {self.config.max_seq_len}")
            
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        # Embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Transformer Blocks
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        
        logits = None
        loss = None
        
        if targets is not None:
            # Training Mode
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # Inference Mode (Optimization: only calculate logits for last step)
            logits = self.lm_head(x[:, [-1], :])
            
        return logits, loss
