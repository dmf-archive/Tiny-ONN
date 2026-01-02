import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextForCausalLM,
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextSparseMoeBlock,
    Qwen3NextMLP,
)
from torch.nn.attention import SDPBackend, sdpa_kernel

from .config import ModelConfig


class MLP(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.SiLU):
        super().__init__()
        self.linear1 = nn.Linear(in_features, in_features) # 1x scale
        self.activation = activation()
        self.linear2 = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


class MLPAttention(Qwen3NextAttention):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        
        self.q_proj = MLP(
            config.hidden_size,
            config.num_attention_heads * self.head_dim * 2,
        )
        self.k_proj = MLP(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
        )
        self.v_proj = MLP(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
        )

class CPRRouter(nn.Module):
    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        
        self.proto = nn.Parameter(torch.empty(self.num_experts, config.hidden_size))
        nn.init.normal_(self.proto, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: (batch_size * sequence_length, hidden_dim)
        
        # Normalize both the hidden states and the prototypes
        normalized_hidden_states = F.normalize(hidden_states, p=2, dim=1)
        normalized_proto = F.normalize(self.proto, p=2, dim=1)

        # Calculate cosine similarity
        router_logits = torch.matmul(normalized_hidden_states, normalized_proto.t())
        
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        return routing_weights, selected_experts


class DynMoE(Qwen3NextSparseMoeBlock):
    def __init__(self, config: Qwen3NextConfig):
        super().__init__(config)
        self.gate = CPRRouter(config)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        
        final_hidden_states = torch.zeros_like(hidden_states_reshaped)

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue

            current_states = hidden_states_reshaped[top_x]
            current_routing_weights = routing_weights[top_x, idx, None]
            
            expert_output = self.experts[expert_idx](current_states)
            
            final_hidden_states.index_add_(0, top_x, expert_output * current_routing_weights)

        # According to ADR-0010, we are not using the shared expert for now.
        # We can add it back later if needed.

        return final_hidden_states.view_as(hidden_states), router_logits


class ArcDynSNT(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device | str):
        super().__init__()
        self.config = config
        self.device = device

        qwen_config = Qwen3NextConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            intermediate_size=config.hidden_size * config.ffn_scale,
            use_cache=True,
            tie_word_embeddings=False,
            dropout=config.dropout,
            layer_types=["full_attention"] * config.num_layers,
            num_experts=8,
            num_experts_per_tok=2, # A common value for MoE
            mlp_only_layers=[],
        )

        self.model = Qwen3NextForCausalLM(qwen_config)
        
        for i, layer in enumerate(self.model.model.layers):
            if isinstance(layer, Qwen3NextDecoderLayer):
                if layer.layer_type == "full_attention":
                    layer.self_attn = MLPAttention(qwen_config, i)
                if hasattr(layer, 'mlp') and isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                    layer.mlp = DynMoE(qwen_config)


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
