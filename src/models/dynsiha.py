import torch
import torch.nn as nn
import torch.nn.functional as F

from src.configs.base import ModelConfig
from src.models.components import SparseProtoLinear


class FlatDynSIHA(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config: ModelConfig = config
        self.d_model: int = config.hidden_size
        self.num_heads: int = config.num_heads
        self.head_dim: int = self.d_model // self.num_heads

        self.embedding: nn.Embedding = nn.Embedding(config.vocab_size, self.d_model)

        self.layers: nn.ModuleList = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(self.d_model),
                "q_spl": SparseProtoLinear(self.d_model, config.latent_attn_expert, self.head_dim),
                "k_spl": SparseProtoLinear(self.d_model, config.latent_attn_expert, self.head_dim),
                "v_spl": SparseProtoLinear(self.d_model, config.latent_attn_expert, self.head_dim),
                "o_proj": nn.Linear(self.d_model, self.d_model, bias=False),
                "ln2": nn.LayerNorm(self.d_model),
                "mlp": nn.Sequential(
                    nn.Linear(self.d_model, self.d_model * config.ffn_scale, bias=False),
                    nn.SiLU(),
                    nn.Linear(self.d_model * config.ffn_scale, self.d_model, bias=False)
                )
            }) for _ in range(config.num_layers)
        ])

        self.lm_head: nn.Linear = nn.Linear(self.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]] | None]:
        B: int
        T: int
        B, T = input_ids.shape
        x: torch.Tensor = self.embedding(input_ids)

        all_routing_info: list[dict[str, torch.Tensor]] = []

        for layer in self.layers:
            residual: torch.Tensor = x
            x = layer["ln1"](x)

            x_proj: torch.Tensor = x.view(B, T, self.num_heads, self.head_dim)

            q: torch.Tensor
            q_info: dict[str, torch.Tensor]
            q, q_info = layer["q_spl"](x_proj)

            k: torch.Tensor
            k_info: dict[str, torch.Tensor]
            k, k_info = layer["k_spl"](x_proj)

            v: torch.Tensor
            v_info: dict[str, torch.Tensor]
            v, v_info = layer["v_spl"](x_proj)

            all_routing_info.extend([q_info, k_info, v_info])

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn_out: torch.Tensor = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)

            x = residual + layer["o_proj"](attn_out)

            residual = x
            x = layer["ln2"](x)
            x = residual + layer["mlp"](x)

        logits: torch.Tensor = self.lm_head(x)

        loss: torch.Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100)

        return {
            "logits": logits,
            "loss": loss,
            "routing_info": all_routing_info
        }
