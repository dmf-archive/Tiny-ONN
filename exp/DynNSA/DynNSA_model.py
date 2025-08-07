import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import PretrainedConfig
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from typing import Optional
from Dense_model import Config, DEVICE, DTYPE
from local_attention import LocalAttention

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

class DynamicSparseAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.sliding_window = LocalAttention(
            dim=self.head_dim,
            window_size=config.selection_block_size,
            causal=True,
            autopad=True,
            use_rotary_pos_emb=False
        )

        self.compress_block_size = config.selection_block_size
        compress_mlp = nn.Sequential(
            nn.Linear(self.compress_block_size * self.head_dim, self.head_dim),
            nn.GELU(),
        )
        self.k_compress = compress_mlp
        self.v_compress = compress_mlp

        self.k_scale = nn.Parameter(torch.ones(1))
        self.k_bias = nn.Parameter(torch.zeros(1))
        
        self.to_strategy_combine = nn.Sequential(
            nn.Linear(config.hidden_size, 3 * self.num_heads),
            nn.Sigmoid(),
            Rearrange('b n (h s) -> b h n s', h=self.num_heads)
        )
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        B, T, _ = hidden_states.shape
        S = self.config.selection_block_size
        H = self.num_heads

        q_raw, k_raw, v_raw = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        q = rearrange(q_raw, 'b t (h d) -> b h t d', h=H)
        k = rearrange(k_raw, 'b t (h d) -> b h t d', h=H)
        v = rearrange(v_raw, 'b t (h d) -> b h t d', h=H)
        
        sliding_window_attn_out = self.sliding_window(q, k, v)
        
        padding = (S - T % S) % S
        if padding > 0:
            k_padded = F.pad(k, (0, 0, 0, padding), 'constant', 0)
            v_padded = F.pad(v, (0, 0, 0, padding), 'constant', 0)
            q_padded = F.pad(q, (0, 0, 0, padding), 'constant', 0)
        else:
            k_padded, v_padded, q_padded = k, v, q
        
        padded_T = T + padding
        num_blocks = padded_T // S

        k_blocked = rearrange(k_padded, 'b h (n s) d -> (b h n) (s d)', s=S)
        v_blocked = rearrange(v_padded, 'b h (n s) d -> (b h n) (s d)', s=S)

        k_compressed = self.k_compress(k_blocked).view(B, H, num_blocks, -1)
        v_compressed = self.v_compress(v_blocked).view(B, H, num_blocks, -1)
        
        importance_scores = torch.einsum('b h i d, b h j d -> b h i j', q_padded, k_compressed) / (self.head_dim ** 0.5)
        compressed_attn_probs = F.softmax(importance_scores, dim=-1)
        compressed_attn_out = torch.einsum('b h i j, b h j d -> b h i d', compressed_attn_probs, v_compressed)
        
        block_attn_scores = rearrange(importance_scores, 'b h (i s) j -> b h i s j', s=S).mean(dim=-2)
        
        block_attn_probs = F.softmax(block_attn_scores, dim=-1)
        entropy = -torch.sum(block_attn_probs * torch.log(block_attn_probs + 1e-9), dim=-1)
        max_entropy = torch.log(torch.tensor(num_blocks, device=DEVICE))
        normalized_entropy = entropy / (max_entropy + 1e-9)

        k_logit = self.k_scale * normalized_entropy + self.k_bias
        k_ratio = torch.sigmoid(k_logit)
        dynamic_k = self.config.max_selected_blocks * k_ratio
        dynamic_k = torch.clamp(dynamic_k, min=1, max=self.config.max_selected_blocks).long()
        avg_k = dynamic_k.float().mean()
        
        top_k_indices = torch.topk(block_attn_scores, self.config.max_selected_blocks, dim=-1).indices
        mask = torch.zeros_like(block_attn_scores, dtype=torch.bool)
        arange_k = torch.arange(self.config.max_selected_blocks, device=DEVICE)[None, None, None, :]
        k_mask = arange_k < dynamic_k.unsqueeze(-1)
        
        top_k_to_set = torch.masked_select(top_k_indices, k_mask)
        batch_indices, head_indices, block_q_indices, _ = torch.where(k_mask)
        if top_k_to_set.numel() > 0:
            mask[batch_indices, head_indices, block_q_indices, top_k_to_set] = True

        diag_mask = torch.eye(num_blocks, device=DEVICE, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        mask = mask | diag_mask
        sparse_mask = torch.repeat_interleave(mask, S, dim=2).repeat_interleave(S, dim=3)
        
        attn_weights = torch.einsum('b h i d, b h j d -> b h i j', q_padded, k_padded) / (self.head_dim ** 0.5)
        if attention_mask is not None:
             attn_weights.masked_fill_(~attention_mask[:,:,:padded_T,:padded_T], max_neg_value(attn_weights))
        attn_weights.masked_fill_(~sparse_mask, max_neg_value(attn_weights))
        
        final_attn_probs = F.softmax(attn_weights, dim=-1)
        fine_attn_out = torch.einsum('b h i j, b h j d -> b h i d', final_attn_probs, v_padded)

        fine_attn_out = fine_attn_out[:, :, :T, :]
        compressed_attn_out = compressed_attn_out[:, :, :T, :]

        strategy_weights = self.to_strategy_combine(hidden_states)
        combined_out = torch.stack([sliding_window_attn_out, compressed_attn_out, fine_attn_out], dim=-1)
        out = torch.einsum('b h n d s, b h n s -> b h n d', combined_out, strategy_weights)

        out = self.merge_heads(out)
        output = self.o_proj(out)
        
        entropy_loss = F.mse_loss(k_ratio, normalized_entropy)
        sparsity_loss = k_ratio.mean()
        aux_loss = self.config.w_entropy * entropy_loss + self.config.w_sparse * sparsity_loss
        return output, avg_k, aux_loss

class DynNSAModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.attn = DynamicSparseAttention(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        causal_mask = torch.tril(torch.ones(T, T, device=DEVICE)).view(1, 1, T, T).bool()
        
        padding = (self.config.selection_block_size - T % self.config.selection_block_size) % self.config.selection_block_size
        if padding > 0:
            causal_mask = torch.tril(torch.ones(T + padding, T + padding, device=DEVICE)).view(1, 1, T + padding, T + padding).bool()

        hidden_states = self.embedding(input_ids)
        attn_output, avg_k, sparsity_loss = self.attn(self.ln1(hidden_states), attention_mask=causal_mask)
        hidden_states = hidden_states + attn_output
        ffn_output = self.ffn(self.ln2(hidden_states))
        hidden_states = hidden_states + ffn_output
        logits = self.lm_head(hidden_states)
        
        return logits, avg_k, sparsity_loss