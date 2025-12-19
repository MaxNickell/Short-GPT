"""ShortGPT: Decoder-only transformer for shortest path prediction."""

import torch
import torch.nn as nn

from src.config import ShortGPTConfig
from .block import TransformerBlock
from src.tokenizer import ShortGPTTokenizer


class ShortGPT(nn.Module):
    """
    Decoder-only GPT model for shortest-path reasoning.

    Architecture:
        - Token embeddings (with weight tying to output layer)
        - N transformer blocks (pre-norm architecture)
        - Final layer norm
        - Linear projection to vocabulary
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
            [TransformerBlock(config) for _ in range(config.n_layers)]
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

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: LongTensor of shape (B, T)

        Returns:
            logits: Tensor of shape (B, T, vocab_size)
        """
        B, T = input_ids.shape

        # 1. Embed tokens
        x = self.token_emb(input_ids)  # (B, T, d_model)

        # 2. Apply dropout
        x = self.drop(x)

        # 3. Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # 4. Final layernorm
        x = self.ln_f(x)

        # 5. Project to logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        tokenizer: ShortGPTTokenizer,
        row: dict,
        device: torch.device,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        """
        Generate a path given a graph problem.

        Args:
            tokenizer: Tokenizer for encoding/decoding
            row: Dataset row with graph_repr, origin, destination
            device: Device to run inference on
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (only used if do_sample=True)
            do_sample: If True, sample from distribution; if False, greedy decode

        Returns:
            Full generated sequence as string
        """
        self.eval()
        self.to(device)

        graph_repr = row["graph_repr"]
        origin = row["origin"]
        dest = row["destination"]

        prompt_str = (
            graph_repr
            + "<ORIGIN>" + str(origin)
            + "<DEST>" + str(dest)
            + "<START_PATH>"
        )

        input_ids_list = tokenizer.encode_string(prompt_str)
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            logits = self(input_ids)
            next_logits = logits[:, -1, :]

            if do_sample:
                if temperature != 1.0:
                    next_logits = next_logits / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_id = torch.argmax(next_logits, dim=-1)

            input_ids = torch.cat(
                [input_ids, next_id.unsqueeze(0)], dim=1
            )

            next_token_str = tokenizer.decode([next_id.item()])[0]
            if next_token_str == "<END_PATH>":
                break

            if input_ids.size(1) >= self.config.max_seq_len:
                break

        full_token_ids = input_ids[0].tolist()
        full_tokens = tokenizer.decode(full_token_ids)
        generated_str = "".join(full_tokens)
        return generated_str

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.

        Args:
            non_embedding: If True, exclude embedding parameters
                          (since they're tied to lm_head)

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_emb.weight.numel()
        return n_params

    @classmethod
    def from_pretrained(cls, path: str, config: ShortGPTConfig) -> "ShortGPT":
        """
        Load a pretrained model from a checkpoint.

        Args:
            path: Path to checkpoint file
            config: Model configuration

        Returns:
            Loaded model
        """
        model = cls(config)
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model
