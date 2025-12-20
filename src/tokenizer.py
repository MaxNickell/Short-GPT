import re
from typing import List


class ShortGPTTokenizer:
    """
    Tokenizer for Short-GPT using a fixed vocabulary.
    Responsible ONLY for:
      - managing the vocab
      - converting between strings, tokens, and IDs
    """

    def __init__(self, vocab_size: int = 24):
        # Fixed vocabulary order
        self.tokens: List[str] = [
            "<PAD>",
            "<EDGE>",
            "<BD>",
            "<ORIGIN>",
            "<DEST>",
            "<START_PATH>",
            "<END_PATH>",
            "<TO>",
        ] + [str(i) for i in range(1, 16)]  # node identifiers "0"–"15"

        # lookup dictionaries
        self.token_to_id = {tok: i for i, tok in enumerate(self.tokens)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        # If vocab_size not provided, infer it from tokens
        self.vocab_size = vocab_size if vocab_size is not None else len(self.tokens)

        # Sanity: keep config.vocab_size in sync with len(self.tokens)
        if self.vocab_size != len(self.tokens):
            raise ValueError(
                f"vocab_size={self.vocab_size} does not match len(tokens)={len(self.tokens)}. "
                "Update your config or tokenizer init."
            )

        # Regex that matches the longest token first
        token_patterns = sorted(self.tokens, key=len, reverse=True)
        escaped = [re.escape(t) for t in token_patterns]
        pattern = "|".join(escaped)
        self.lexer_re = re.compile(pattern)

    def encode(self, token_list: List[str]) -> List[int]:
        """Convert a list of string tokens into a list of integer token IDs."""
        return [self.token_to_id[t] for t in token_list]

    def decode(self, id_list: List[int]) -> List[str]:
        """Convert a list of token IDs back into string tokens."""
        return [self.id_to_token[i] for i in id_list]

    def tokenize_string(self, s: str) -> List[str]:
        """Lex a raw string into a list of tokens from the known vocabulary."""
        s = "".join(s.split())

        tokens: List[str] = []
        pos = 0
        n = len(s)

        while pos < n:
            m = self.lexer_re.match(s, pos)
            if not m:
                raise ValueError(
                    f"Unrecognized sequence at position {pos}: {s[pos:pos+30]!r}"
                )
            tok = m.group(0)
            tokens.append(tok)
            pos = m.end()

        return tokens

    def encode_string(self, s: str) -> List[int]:
        """
        Convenience: go directly from raw string to token IDs.
        """
        tokens = self.tokenize_string(s)
        return self.encode(tokens)

    @property
    def pad_token(self) -> str:
        return "<PAD>"

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]
