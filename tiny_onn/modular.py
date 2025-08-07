from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from local_attention import LocalAttention
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import TinyOnnConfig


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


@dataclass
class CausalLMOutputWithAux(CausalLMOutputWithPast):
    aux_outputs: list[dict[str, torch.Tensor]] | None = None

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        return (scores > 0).to(scores.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class TinyOnnAttention(nn.Module):
    def __init__(self, config: TinyOnnConfig):
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
            use_rotary_pos_emb=False,
        )

        compress_mlp = nn.Sequential(
            nn.Linear(config.selection_block_size * self.head_dim, self.head_dim),
            nn.GELU(),
        )
        self.k_compress = compress_mlp
        self.v_compress = compress_mlp

        self.k_scale = nn.Parameter(torch.ones(1, self.num_heads, 1))
        self.k_bias = nn.Parameter(torch.zeros(1, self.num_heads, 1))

        self.to_strategy_combine = nn.Sequential(
            nn.Linear(config.hidden_size, 3 * self.num_heads),
            nn.Sigmoid(),
            Rearrange('b n (h s) -> b h n s', h=self.num_heads),
        )
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        B, T, _ = hidden_states.shape
        S = self.config.selection_block_size
        H = self.num_heads

        q_raw, k_raw, v_raw = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        q = rearrange(q_raw, 'b t (h d) -> b h t d', h=H)
        k = rearrange(k_raw, 'b t (h d) -> b h t d', h=H)
        v = rearrange(v_raw, 'b t (h d) -> b h t d', h=H)

        sliding_window_attn_out = self.sliding_window(q, k, v)

        padding = (S - T % S) % S
        k_padded, v_padded, q_padded = (
            (F.pad(t, (0, 0, 0, padding), 'constant', 0) if padding > 0 else t)
            for t in (k, v, q)
        )

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
        max_entropy = torch.log(torch.tensor(num_blocks, device=hidden_states.device))
        normalized_entropy = entropy / (max_entropy + 1e-9)

        k_logit = self.k_scale * normalized_entropy + self.k_bias
        k_ratio = torch.sigmoid(k_logit)
        dynamic_k = torch.clamp(self.config.max_selected_blocks * k_ratio, min=1, max=self.config.max_selected_blocks).long()
        avg_k = dynamic_k.float().mean()

        top_k_indices = torch.topk(block_attn_scores, self.config.max_selected_blocks, dim=-1).indices
        mask = torch.zeros_like(block_attn_scores, dtype=torch.bool)
        
        arange_k = torch.arange(self.config.max_selected_blocks, device=hidden_states.device)[None, None, None, :]
        k_mask = arange_k < dynamic_k.unsqueeze(-1)
        
        top_k_to_set = torch.masked_select(top_k_indices, k_mask)
        batch_indices, head_indices, block_q_indices, _ = torch.where(k_mask)
        if top_k_to_set.numel() > 0:
            mask[batch_indices, head_indices, block_q_indices, top_k_to_set] = True

        mask |= torch.eye(num_blocks, device=hidden_states.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
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

        aux_outputs = {
            "avg_k": avg_k,
            "k_ratio": k_ratio,
            "normalized_entropy": normalized_entropy,
        }
        return output, aux_outputs

class TinyOnnExpert(nn.Module):
    def __init__(self, config: TinyOnnConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.w2 = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))

class TinyOnnGate(nn.Module):
    def __init__(self, config: TinyOnnConfig):
        super().__init__()
        self.config = config
        self.sim_matrix = nn.Parameter(torch.randn(config.hidden_size, config.num_experts_per_layer))
        self.gates = nn.Parameter(torch.zeros(config.num_experts_per_layer))

    def forward(self, x: torch.Tensor):
        logits = torch.matmul(F.normalize(x, dim=-1), F.normalize(self.sim_matrix, dim=0))
        gate_thresholds = torch.sigmoid(self.gates)
        pre_activation_logits = logits - gate_thresholds

        gated_logits = F.relu(pre_activation_logits)
        activation_mask = STEFunction.apply(gated_logits)

        num_active_experts = torch.sum(activation_mask, dim=1)
        inactive_tokens_mask = num_active_experts == 0
        if self.training and inactive_tokens_mask.any():
            k_fallback = self.config.num_experts_per_layer // 2
            topk_expert_indices = torch.topk(logits[inactive_tokens_mask], k=k_fallback, dim=1).indices

            batch_indices = torch.where(inactive_tokens_mask)[0].unsqueeze(1).expand(-1, k_fallback)

            activation_mask[batch_indices, topk_expert_indices] = 1.0

        gated_logits_masked = torch.where(activation_mask > 0, gated_logits, max_neg_value(gated_logits))
        active_expert_probs = F.softmax(gated_logits_masked, dim=-1)

        return active_expert_probs, pre_activation_logits, activation_mask

class TinyOnnMoE(nn.Module):
    def __init__(self, config: TinyOnnConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts_per_layer
        self.experts = nn.ModuleList([TinyOnnExpert(config) for _ in range(self.num_experts)])
        self.gate = TinyOnnGate(config)

    def forward(self, hidden_states: torch.Tensor):
        B, T, C = hidden_states.shape
        flat_hs = hidden_states.view(-1, C)

        routing_weights, pre_act_logits, activation_mask = self.gate(flat_hs)
        token_indices, expert_indices = torch.where(activation_mask > 0)

        full_expert_outputs = torch.zeros(B * T, self.num_experts, C, device=hidden_states.device, dtype=hidden_states.dtype)

        if token_indices.numel() > 0:
            dispatched_outputs = torch.empty_like(flat_hs[token_indices])
            for i in range(self.num_experts):
                if (mask := expert_indices == i).any():
                    dispatched_outputs[mask] = self.experts[i](flat_hs[token_indices][mask])
            full_expert_outputs.index_put_((token_indices, expert_indices), dispatched_outputs)

        final_output = torch.einsum('te,tec->tc', routing_weights, full_expert_outputs).view(B, T, C)

        aux_outputs = {
            "full_expert_outputs": full_expert_outputs,
            "pre_act_logits": pre_act_logits,
            "activation_mask": activation_mask,
        }
        return final_output, aux_outputs


class TinyOnnDecoderLayer(nn.Module):
    def __init__(self, config: TinyOnnConfig):
        super().__init__()
        self.self_attn = TinyOnnAttention(config)
        self.mlp = TinyOnnMoE(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output, attn_aux = self.self_attn(hidden_states, attention_mask=attention_mask)

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        mlp_output, moe_aux = self.mlp(hidden_states)

        hidden_states = residual + mlp_output

        aux_outputs = {"attn": attn_aux, "moe": moe_aux}
        return hidden_states, aux_outputs

class TinyOnnModel(PreTrainedModel):
    config_class = TinyOnnConfig

    def __init__(self, config: TinyOnnConfig):
        super().__init__(config)
        self.padding_idx = -1
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([TinyOnnDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None):
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is None:
            B, T = input_ids.shape
            padding = (self.config.selection_block_size - T % self.config.selection_block_size) % self.config.selection_block_size
            padded_T = T + padding
            attention_mask = torch.tril(torch.ones(padded_T, padded_T, device=hidden_states.device)).view(1, 1, padded_T, padded_T).bool()

        all_aux_outputs = []
        for decoder_layer in self.layers:
            hidden_states, aux_outputs = decoder_layer(hidden_states, attention_mask)
            all_aux_outputs.append(aux_outputs)

        hidden_states = self.norm(hidden_states)
        return hidden_states, all_aux_outputs

class TinyOnnForCausalLM(PreTrainedModel):
    config_class = TinyOnnConfig

    def __init__(self, config: TinyOnnConfig):
        super().__init__(config)
        self.model = TinyOnnModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None, **kwargs):
        hidden_states, aux_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)

        return CausalLMOutputWithAux(
            logits=logits,
            aux_outputs=aux_outputs,
        )
