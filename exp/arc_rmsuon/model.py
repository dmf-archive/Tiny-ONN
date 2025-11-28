import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from torch.nn.attention import SDPBackend, sdpa_kernel

from .config import ModelConfig


class ArcTransformer(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device | str):
        super().__init__()
        self.config = config
        self.device = device

        qwen_config = Qwen3Config(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            intermediate_size=config.hidden_size * config.ffn_scale,
            use_cache=True,
            use_sliding_window=False,
            tie_word_embeddings=False,
            dropout=config.dropout,
        )

        self.model = Qwen3ForCausalLM(qwen_config)
        self.to(device)
        torch.cuda.empty_cache()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list | None = None,
        return_dict: bool = False,
    ):
        # 强制使用高效的SDPA实现，提升attention计算效率
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=(past_key_values is not None),
                return_dict=True,
            )

        if not return_dict:
            return (outputs.logits, outputs.past_key_values)

        return outputs
