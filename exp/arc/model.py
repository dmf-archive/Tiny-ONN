
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


@torch.jit.script
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

@torch.jit.script
def apply_rope(q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor, hidden_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = position_ids.max() + 1
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, hidden_size, 2, device=q.device).float() / hidden_size))
    t = torch.arange(seq_len, device=q.device).type_as(inv_freq)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos()[position_ids, :].unsqueeze(2)
    sin = emb.sin()[position_ids, :].unsqueeze(2)

    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q, k = apply_rope(q, k, position_ids, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present_kv = (k, v)

        is_causal = past_kv is None
        import torch.nn.attention as attention
        with attention.sdpa_kernel(attention.SDPBackend.MATH):
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.o_proj(attn_output), present_kv

class MLP(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        intermediate_size = config.hidden_size * config.ffn_scale
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32, use_checkpoint: bool = False):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-5, dtype=dtype)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-5, dtype=dtype)
        self.attention = Attention(config, dtype=dtype)
        self.mlp = MLP(config, dtype=dtype)
        self.use_checkpoint = use_checkpoint

    def _forward_impl(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        position_ids = position_ids.long()

        residual = x
        x_norm = self.input_layernorm(x)
        attn_output, present_kv = self.attention(x_norm, position_ids, past_kv)
        x = residual + attn_output

        residual = x
        x_norm = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x_norm)
        x = residual + mlp_output

        return x, present_kv

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        if self.use_checkpoint and self.training:
            return checkpoint(
                self._forward_impl,
                x, position_ids, past_kv,
                use_reentrant=False,
                preserve_rng_state=False
            )
        else:
            return self._forward_impl(x, position_ids, past_kv)


class ArcEmbedding(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.color_embedding = nn.Embedding(config.vocab_size, self.hidden_size, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype not in [torch.long, torch.int32, torch.int64]:
            input_ids = input_ids.long()
        return self.color_embedding(input_ids)

class ArcTransformer(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device | str):
        super().__init__()
        self.config, self.device = config, device
        dtype = torch.float32
        self.embedding = ArcEmbedding(config, dtype=dtype)
        self.blocks = nn.ModuleList([TransformerBlock(config, dtype=dtype, use_checkpoint=config.use_checkpoint) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-5, dtype=dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)

        self.to(device)
        torch.cuda.empty_cache()

    def forward(
        self,
        input_ids: torch.Tensor,
        coords: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list | None = None,
        return_dict: bool = False,
    ):
        x = self.embedding(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + x.shape[1], device=x.device, dtype=torch.long
            ).unsqueeze(0)
        else:
            position_ids = position_ids.long()

        presents = []
        past_key_values = past_key_values if past_key_values is not None else [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            x, present_kv = block(
                x,
                position_ids=position_ids,
                past_kv=past_key_values[i],
            )
            presents.append(present_kv)

        x = self.norm(x)
        logits = self.lm_head(x)

        if not return_dict:
            return (logits, presents)

        return {
            "logits": logits,
            "past_key_values": presents,
        }
