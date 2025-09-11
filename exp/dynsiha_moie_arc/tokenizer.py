from collections.abc import Mapping
from itertools import product


class ArcPositionalTokenizer:
    def __init__(self, max_grid_size: int = 30):
        self.max_grid_size = max_grid_size
        self.control_tokens = {
            "<|pad|>": 0,
            "<|bos|>": 1,
            "<|eos|>": 2,
            "<|im_start|>": 3,
            "<|im_end|>": 4,
            "problem": 5,
            "solution": 6,
        }
        
        self.grid_token_offset = len(self.control_tokens)
        self.num_colors = 10

        self.vocab = {**self.control_tokens}
        for i, j, c in product(range(max_grid_size), range(max_grid_size), range(self.num_colors)):
            token_str = f"_{i}_{j}_{c}"
            token_id = self.grid_token_offset + (i * max_grid_size * self.num_colors) + (j * self.num_colors) + c
            self.vocab[token_str] = token_id

        self.inv_vocab: Mapping[int, str] = {v: k for k, v in self.vocab.items()}

    def grid_to_token_id(self, i: int, j: int, c: int) -> int:
        return self.grid_token_offset + (i * self.max_grid_size * self.num_colors) + (j * self.num_colors) + c

    def token_id_to_grid(self, token_id: int) -> tuple[int, int, int] | None:
        if token_id < self.grid_token_offset:
            return None
        
        relative_id = token_id - self.grid_token_offset
        c = relative_id % self.num_colors
        j = (relative_id // self.num_colors) % self.max_grid_size
        i = relative_id // (self.num_colors * self.max_grid_size)
        return i, j, c

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.control_tokens["<|pad|>"]

    @property
    def bos_token_id(self) -> int:
        return self.control_tokens["<|bos|>"]
        
    @property
    def eos_token_id(self) -> int:
        return self.control_tokens["<|eos|>"]

    def encode(self, text: str) -> list[int]:
        # This tokenizer is not designed for general text encoding.
        # It's primarily used for structured serialization.
        # However, a simple split-based encode can be useful for prompts.
        return [self.vocab[token] for token in text.split(" ") if token in self.vocab]

    def decode(self, tokens: list[int]) -> str:
        # Similarly, for debugging and inspection purposes.
        return " ".join([self.inv_vocab.get(token, "<|unk|>") for token in tokens])
