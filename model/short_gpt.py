import torch
import torch.nn as nn
from config import ShortGPTConfig


class ShortGPT(nn.Module):
    """
    Top-level GPT model for shortest-path reasoning.
    """

    def __init__(self, config: ShortGPTConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_emb = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
        )

        # Dropout
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            []  # WIP
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Linear layer to project hidden states to vocabulary size
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        # Weight tying: token_emb and lm_head share parameters
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids):
        """
        input_ids: LongTensor of shape (B, T)
        Returns logits of shape (B, T, vocab_size)
        """
        B, T = input_ids.shape

        # 1. Embed tokens
        x = self.token_emb(input_ids) # (B, T, d_model)

        # 2. Apply dropout
        x = self.drop(x)

        # 3. Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # 4. Final layernorm
        x = self.ln_f(x)

        # 5. Project to logits
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits
