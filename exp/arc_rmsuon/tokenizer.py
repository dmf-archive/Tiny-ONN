import os
from collections.abc import Mapping

from tokenizers import Tokenizer as HfTokenizer
from transformers import PreTrainedTokenizerFast


class ArcColorTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = {"tokenizer_file": "tokenizer.json"}

    def __init__(self, tokenizer_file=None, **kwargs):
        # 加载底层的 Rust-based tokenizer
        if tokenizer_file is None:
            base_path = os.path.dirname(__file__)
            tokenizer_file = os.path.join(base_path, "dist", "tokenizer.json")

        slow_tokenizer = HfTokenizer.from_file(tokenizer_file)

        # 从加载的 tokenizer 中提取特殊 tokens
        vocab = slow_tokenizer.get_vocab()
        pad_token = "<|pad|>"
        bos_token = "<|bos|>"
        eos_token = "<|eos|>"
        unk_token = "<|unk|>"

        super().__init__(
            tokenizer_object=slow_tokenizer,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            **kwargs,
        )

        if self.pad_token_id is None:
            self.add_special_tokens({'pad_token': pad_token})

        self.color_token_offset = 8
        self.num_colors = 10
        self._vocab = vocab
        self.inv_vocab: Mapping[int, str] = {v: k for k, v in self._vocab.items()}

    def color_to_token_id(self, color: int) -> int:
        return self.color_token_offset + color

    def token_id_to_color(self, token_id: int) -> int | None:
        if not (self.color_token_offset <= token_id < self.color_token_offset + self.num_colors):
            return None
        return token_id - self.color_token_offset

    @property
    def row_sep_token_id(self) -> int:
        return self._vocab["\n"]

    # The `vocab_size` and other properties are now inherited from PreTrainedTokenizerFast
