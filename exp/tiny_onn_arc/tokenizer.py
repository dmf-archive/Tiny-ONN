from collections.abc import Mapping


class ArcChatMLTokenizer:
    def __init__(self):
        self.special_tokens = {
            "<|im_start|>": 10,
            "<|im_end|>": 11,
            "problem": 12,
            "solution": 13,
            "<|pad|>": 14,
            "<row_start>": 15,
            "<row_end>": 16,
        }
        self.color_tokens = {str(i): i for i in range(10)}

        self.vocab: Mapping[str, int] = {
            **self.color_tokens,
            **self.special_tokens,
        }

        self.inv_vocab: Mapping[int, str] = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens["<|pad|>"]

    @property
    def bos_token_id(self) -> int:
        return self.special_tokens["<|im_start|>"]
        
    @property
    def eos_token_id(self) -> int:
        return self.special_tokens["<|im_end|>"]

    def encode(self, text: str) -> list[int]:
        tokens = text.replace("\n", " \\n ").split(" ")
        return [self.vocab[token] for token in tokens if token and token in self.vocab]

    def decode(self, tokens: list[int]) -> str:
        text = " ".join([self.inv_vocab.get(token, "") for token in tokens])
        return text.replace(" \\n ", "\n")
