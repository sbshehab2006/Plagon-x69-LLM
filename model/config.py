from dataclasses import dataclass

@dataclass
class PlagonConfig:
    """
    Configuration for Plagon LLM.
    """
    # Architecture
    vocab_size: int = 50257     # Standard GPT size, ample for Code + English + Bangla
    max_seq_len: int = 128      # context window (reduced for safety)
    dim: int = 256              # embedding dimension (d_model)
    n_layers: int = 2           # number of transformer blocks
    n_heads: int = 4            # number of attention heads
    
    # Regularization
    dropout: float = 0.1
    
    # Training
    batch_size: int = 4
    learning_rate: float = 3e-4
    
    # Generation
    temperature: float = 0.8
    top_k: int = 40

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
