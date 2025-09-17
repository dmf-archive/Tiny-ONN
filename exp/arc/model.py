import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device: torch.device | None = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin

class SparseProtoLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.proto_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.mu_bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        self.gate_param = nn.Parameter(torch.empty(out_features, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.mu_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.proto_weight, a=math.sqrt(5))
        nn.init.zeros_(self.mu_bias)
        nn.init.constant_(self.gate_param, -0.1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = torch.matmul(x, self.proto_weight.t())
        raw_weights = F.relu(scores - self.gate_param)
        computation_output = F.linear(x, self.mu_weight, self.mu_bias)
        masked_output = computation_output * raw_weights
        return masked_output, computation_output, raw_weights

class DynamicInfiniteHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = config.hidden_size
        self.sbl_qkv = SparseProtoLinear(self.d_model, 3 * self.d_model, dtype=dtype)
        self.sbl_o = SparseProtoLinear(self.d_model, self.d_model, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        m_qkv, c_qkv, rw_qkv = self.sbl_qkv(x)
        q, k, v = torch.split(m_qkv, self.d_model, dim=-1)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        present_key_value = (k, v)

        is_causal = past_key_value is None
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        m_o, c_o, rw_o = self.sbl_o(attn_out)
        
        masked_outputs = [m_qkv, m_o]
        comp_outputs = [c_qkv, c_o]
        raw_weights = [rw_qkv, rw_o]
        sbl_inputs = [x, attn_out]

        return m_o, masked_outputs, comp_outputs, raw_weights, sbl_inputs, present_key_value

class DynamicInfiniteExpert(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        d_ffn = config.hidden_size * config.d_ffn_factor
        self.sbl1 = SparseProtoLinear(config.hidden_size, d_ffn, dtype=dtype)
        self.sbl2 = SparseProtoLinear(d_ffn, config.hidden_size, dtype=dtype)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        m1, c1, rw1 = self.sbl1(x)
        h_act = F.relu(m1)
        m2, c2, rw2 = self.sbl2(h_act)

        masked_outputs = [m1, m2]
        comp_outputs = [c1, c2]
        raw_weights = [rw1, rw2]
        sbl_inputs = [x, h_act]

        return m2, masked_outputs, comp_outputs, raw_weights, sbl_inputs

class MoIETransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, dtype=dtype)
        self.attn = DynamicInfiniteHeadAttention(config, dtype=dtype)
        self.ln2 = nn.LayerNorm(config.hidden_size, dtype=dtype)
        self.ffn = DynamicInfiniteExpert(config, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        attn_in = self.ln1(x)
        attn_out, attn_m, attn_c, attn_rw, attn_inputs, present_key_value = self.attn(
            attn_in, position_embeddings, past_key_value
        )
        x = x + attn_out

        ffn_in = self.ln2(x)
        ffn_out, ffn_m, ffn_c, ffn_rw, ffn_inputs = self.ffn(ffn_in)
        x_out = x + ffn_out

        masked_outputs = attn_m + ffn_m
        comp_outputs = attn_c + ffn_c
        raw_weights = attn_rw + ffn_rw
        sbl_inputs = attn_inputs + ffn_inputs

        return x_out, masked_outputs, comp_outputs, raw_weights, sbl_inputs, present_key_value

class ArcTransformer(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device | str):
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.device = device

        dtype = torch.bfloat16

        self.embedding = nn.Embedding(self.vocab_size, config.hidden_size, dtype=dtype)
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            dtype=dtype,
            device=device
        )
        self.blocks = nn.ModuleList([
            MoIETransformerBlock(config, dtype=dtype) for _ in range(self.num_layers)
        ])
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
    ):
        assert input_ids.max().item() < self.embedding.num_embeddings, "Token ID out of vocab range"
        tok_emb = self.embedding(input_ids)
        x = tok_emb

        seq_len = input_ids.size(1)
        position_embeddings = self.rotary_emb(x, seq_len=seq_len)

        if past_key_values is None:
            past_key_values = [None] * self.num_layers

        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        all_masked_outputs: list[torch.Tensor] = []
        all_comp_outputs: list[torch.Tensor] = []
        all_proto_weights: list[torch.Tensor] = []
        all_sbl_inputs: list[torch.Tensor] = []
        all_block_raw_weights: list[list[torch.Tensor]] = []

        for i, block in enumerate(self.blocks):
            x, masked, comp, raw, sbl_inputs, present_key_value = block(
                x, position_embeddings, past_key_values[i]
            )
            present_key_values.append(present_key_value)
            all_masked_outputs.extend(masked)
            all_comp_outputs.extend(comp)
            all_sbl_inputs.extend(sbl_inputs)
            all_block_raw_weights.append(raw)
            for module in block.modules():
                if isinstance(module, SparseProtoLinear):
                    all_proto_weights.append(module.proto_weight)

        logits = self.lm_head(x)

        return logits, all_masked_outputs, all_comp_outputs, all_proto_weights, all_sbl_inputs, all_block_raw_weights, present_key_values
