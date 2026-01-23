import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union
from .router import MLPRouter, VectorizedExpertMLP
from transformers.cache_utils import Cache

class DynSIHARMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class DynSIHARotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class DynSIHAAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_experts: int,
        top_k: int,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.layer_idx = layer_idx
        
        self.q_router = MLPRouter(self.head_dim, num_experts, top_k, dtype=dtype)
        self.k_router = MLPRouter(self.head_dim, num_experts, top_k, dtype=dtype)
        self.v_router = MLPRouter(self.head_dim, num_experts, top_k, dtype=dtype)
        
        self.q_experts = VectorizedExpertMLP(num_experts, self.head_dim, self.head_dim, dtype=dtype)
        self.k_experts = VectorizedExpertMLP(num_experts, self.head_dim, self.head_dim, dtype=dtype)
        self.v_experts = VectorizedExpertMLP(num_experts, self.head_dim, self.head_dim, dtype=dtype)
        
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Cache]]:
        B, T, C = x.shape
        x_heads = x.view(B, T, self.num_heads, self.head_dim)
        
        qw, qe, ql = self.q_router(x_heads)
        kw, ke, kl = self.k_router(x_heads)
        vw, ve, vl = self.v_router(x_heads)
        
        q = self.q_experts(x_heads, qw, qe).transpose(1, 2) # [B, num_heads, T, head_dim]
        k = self.k_experts(x_heads, kw, ke).transpose(1, 2)
        v = self.v_experts(x_heads, vw, ve).transpose(1, 2)
        
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            actual_layer_idx = layer_idx if layer_idx is not None else self.layer_idx
            k, v = past_key_value.update(k, v, actual_layer_idx, cache_kwargs)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=True if attention_mask is None and q.shape[2] > 1 else False
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(attn_output)
        
        routing_info = {"q_logits": ql, "k_logits": kl, "v_logits": vl}
        return output, routing_info, past_key_value

class DynSIHAMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        ffn_scale: int = 4,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.router = MLPRouter(hidden_size, num_experts, top_k, dtype=dtype)
        self.experts = VectorizedExpertMLP(
            num_experts,
            hidden_size,
            hidden_size * ffn_scale,
            dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w, e, l = self.router(x)
        return self.experts(x, w, e), l

class DynSIHABlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_experts: int,
        top_k: int,
        layer_idx: Optional[int] = None,
        ffn_scale: int = 4,
        rms_norm_eps: float = 1e-6,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.ln1 = DynSIHARMSNorm(hidden_size, eps=rms_norm_eps)
        self.attn = DynSIHAAttention(hidden_size, num_heads, num_experts, top_k, layer_idx=layer_idx, dtype=dtype)
        self.ln2 = DynSIHARMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = DynSIHAMLP(hidden_size, num_experts, top_k, ffn_scale, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Cache]]:
        attn_in = self.ln1(x)
        attn_out, attn_routing, past_key_value = self.attn(
            attn_in,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
            layer_idx=layer_idx
        )
        x = x + attn_out
        
        mlp_in = self.ln2(x)
        mlp_out, mlp_routing = self.mlp(mlp_in)
        x = x + mlp_out
        
        return x, {**attn_routing, "mlp_logits": mlp_routing}, past_key_value
