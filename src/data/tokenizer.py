from collections.abc import Mapping

from tokenizers import Tokenizer as HfTokenizer


class ArcColorTokenizer:
    def __init__(self, tokenizer_path: str = "exp/arc_dyntrm/dist/tokenizer.json") -> None:
        self._tokenizer: HfTokenizer = HfTokenizer.from_file(tokenizer_path)
        self.color_token_offset: int = 8
        self.num_colors: int = 10
        self._vocab: Mapping[str, int] = self._tokenizer.get_vocab()
        self.inv_vocab: Mapping[int, str] = {v: k for k, v in self._vocab.items()}

    @property
    def vocab(self) -> Mapping[str, int]:
        return self._vocab

    def color_to_token_id(self, color: int) -> int:
        return self.color_token_offset + color

    def token_id_to_color(self, token_id: int) -> int | None:
        if not (self.color_token_offset <= token_id < self.color_token_offset + self.num_colors):
            return None
        return token_id - self.color_token_offset

    @property
    def row_sep_token_id(self) -> int:
        return self._vocab["\n"]

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self._vocab["<|pad|>"]

    @property
    def bos_token_id(self) -> int:
        return self._vocab["<|bos|>"]

    @property
    def eos_token_id(self) -> int:
        return self._vocab["<|eos|>"]

    def encode(self, text: str) -> list[int]:
        return [self._vocab[token] for token in text.split(" ") if token in self._vocab]

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)
