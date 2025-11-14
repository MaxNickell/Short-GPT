from dataclasses import dataclass


@dataclass
class ShortGPTConfig:
    """
    This class stores every architectural and training-related hyperparameter
    needed to build the model. All components (embeddings, attention,
    RoPE, transformer blocks) will receive this config object.
    """
    # fixed vocabulary size
    vocab_size: int = 32

    # Model architecture
    d_model: int = 256  # hidden dimension
    n_layers: int = 2  # number of transformer blocks
    n_heads: int = 4  # number of attention heads
    d_ff: int = 1024  # feedforward inner dim

    # Token settings
    max_seq_len: int = 512  # max context window
    rope_base: int = 10000  # RoPE frequency base

    # Regularization
    dropout: float = 0.1  # dropout rate