class ShortGPTTokenizer:
    """
    Tokenizer for Short-GPT using the fixed vocabulary.
    Converts between tokens and integer IDs using a predefined, immutable vocabulary.
    """

    def __init__(self, vocab_size=23):
        # Fixed vocabulary order (exactly 23 tokens)
        self.tokens = [
            "<EDGE>",
            "<BD>",
            "<ORIGIN>",
            "<DEST>",
            "<START_PATH>",
            "<END_PATH>",
            "<TO>",
        ] + [str(i) for i in range(16)]  # node identifiers 0–15
        
        # lookup dictionaries
        self.token_to_id = {tok: i for i, tok in enumerate(self.tokens)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        self.vocab_size = vocab_size

    def encode(self, token_list):
        """Convert a list of string tokens into a list of integer token IDs."""
        return [self.token_to_id[t] for t in token_list]

    def decode(self, id_list):
        """Convert a list of token IDs back into string tokens."""
        return [self.id_to_token[i] for i in id_list]
