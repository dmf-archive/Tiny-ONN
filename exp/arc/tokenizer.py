from collections.abc import Mapping


class ArcColorTokenizer:
    def __init__(self):
        self.control_tokens = {
            "<|pad|>": 0,
            "<|bos|>": 1,
            "<|eos|>": 2,
            "problem": 3,
            "solution": 4,
            "\n": 5,
            "<im_start>": 6,
            "<im_end>": 7,
        }

        self.color_token_offset = len(self.control_tokens)
        self.num_colors = 10
        self.color_tokens = {str(i): self.color_token_offset + i for i in range(self.num_colors)}

        self.vocab = {**self.control_tokens, **self.color_tokens}
        self.inv_vocab: Mapping[int, str] = {v: k for k, v in self.vocab.items()}

    def color_to_token_id(self, color: int) -> int:
        return self.color_token_offset + color

    def token_id_to_color(self, token_id: int) -> int | None:
        if token_id < self.color_token_offset or token_id >= self.color_token_offset + self.num_colors:
            return None
        return token_id - self.color_token_offset

    @property
    def row_sep_token_id(self) -> int:
        return self.control_tokens["\n"]

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
        return [self.vocab[token] for token in text.split(" ") if token in self.vocab]

    def decode(self, tokens: list[int]) -> str:
        return " ".join([self.inv_vocab.get(token, "<|unk|>") for token in tokens])
