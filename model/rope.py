import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE)
    Applies position-dependent rotations to query/key vectors.
    Used in modern GPT architectures (GPT-J, GPT-NeoX, LLaMA, etc.)
    """

    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even."

        self.dim = dim
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        # Positions 0...T-1
        positions = torch.arange(max_seq_len, dtype=torch.float32)

        # Outer product: (T, dim/2)
        freqs = torch.outer(positions, inv_freq)

        # Precompute cos & sin
        self.register_buffer("cos_cached", freqs.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, :, None, :], persistent=False)

    def forward(self, x, seq_len):
        """
        x: (B, T, H, D)
        seq_len: actual sequence length to apply RoPE to

        Returns rotated x with same shape.
        """
        # Slice cached cos/sin to the needed sequence length
        cos = self.cos_cached[:, :seq_len, :, :]    # (1, T, 1, D/2)
        sin = self.sin_cached[:, :seq_len, :, :]    # (1, T, 1, D/2)

        # Split last dim into pairs
        x_even = x[..., ::2]    # (B, T, H, D/2)
        x_odd  = x[..., 1::2]   # (B, T, H, D/2)

        # Apply rotation
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd  = x_even * sin + x_odd * cos

        # Recombine the pairs
        x_rot = torch.stack([rotated_even, rotated_odd], dim=-1)
        x_rot = x_rot.flatten(-2, -1)   # back to (B, T, H, D)

        return x_rot
