import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple

from .config import ModelConfig

class SparseBayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.gate_param = nn.Parameter(torch.empty(out_features, dtype=dtype))
        self.mu_bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.mu_weight, a=math.sqrt(5))
        nn.init.normal_(self.sigma_weight, mean=0.0, std=0.5)
        nn.init.constant_(self.gate_param, -0.1)
        nn.init.zeros_(self.mu_bias)

    def forward(self, x: torch.Tensor, prior_std: float, kl_epsilon: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        x_reshaped = x.view(-1, self.in_features)
        
        keys = self.mu_weight * F.softplus(self.sigma_weight)
        scores = torch.matmul(x_reshaped, keys.t()) / math.sqrt(self.in_features)
        raw_weights = F.relu(scores - self.gate_param.unsqueeze(0))

        computation_output = F.linear(x_reshaped, self.mu_weight, self.mu_bias)
        masked_output = computation_output * raw_weights
        
        new_shape = list(original_shape[:-1]) + [self.out_features]
        output = masked_output.view(new_shape)

        sigma_q = F.softplus(self.sigma_weight)
        var_q = sigma_q.pow(2)
        var_p = torch.full_like(sigma_q, prior_std).pow(2)

        kl_div = 0.5 * (torch.log(var_p / (var_q + kl_epsilon)) + (var_q + self.mu_weight.pow(2)) / var_p - 1)
        kl_loss = kl_div.mean()
        
        return output, masked_output, kl_loss

class DynamicInfiniteHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = config.hidden_size
        self.sbl_qkv = SparseBayesianLinear(self.d_model, 3 * self.d_model, dtype=dtype)
        self.sbl_o = SparseBayesianLinear(self.d_model, self.d_model, dtype=dtype)

    def forward(self, x: torch.Tensor, prior_std: float, kl_epsilon: float) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        qkv, m_qkv, kl_qkv = self.sbl_qkv(x, prior_std, kl_epsilon)
        q, k, v = torch.split(qkv, self.d_model, dim=-1)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y, m_o, kl_o = self.sbl_o(attn_out, prior_std, kl_epsilon)
        total_kl = kl_qkv + kl_o
        return y, [m_qkv, m_o], total_kl

class DynamicInfiniteExpert(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        d_ffn = config.hidden_size * config.d_ffn_factor
        self.sbl1 = SparseBayesianLinear(config.hidden_size, d_ffn, dtype=dtype)
        self.sbl2 = SparseBayesianLinear(d_ffn, config.hidden_size, dtype=dtype)

    def forward(self, x: torch.Tensor, prior_std: float, kl_epsilon: float) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        h, m1, kl1 = self.sbl1(x, prior_std, kl_epsilon)
        h_act = F.silu(h)
        y, m2, kl2 = self.sbl2(h_act, prior_std, kl_epsilon)
        total_kl = kl1 + kl2
        return y, [m1, m2], total_kl

class MoIETransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, dtype=dtype)
        self.attn = DynamicInfiniteHeadAttention(config, dtype=dtype)
        self.ln2 = nn.LayerNorm(config.hidden_size, dtype=dtype)
        self.ffn = DynamicInfiniteExpert(config, dtype=dtype)

    def forward(self, x: torch.Tensor, prior_std: float, kl_epsilon: float) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        attn_in = self.ln1(x)
        attn_out, attn_masked_tuple, attn_kl = self.attn(attn_in, prior_std, kl_epsilon)
        x = x + attn_out
        
        ffn_in = self.ln2(x)
        ffn_out, ffn_masked_tuple, ffn_kl = self.ffn(ffn_in, prior_std, kl_epsilon)
        x = x + ffn_out
        
        total_kl = attn_kl + ffn_kl
        masked_outputs = attn_masked_tuple + ffn_masked_tuple
        return x, masked_outputs, total_kl

class ArcTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dtype = torch.bfloat16 # Hardcode for now as per project spec

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.blocks = nn.ModuleList([
            MoIETransformerBlock(config, dtype=dtype) for _ in range(config.num_layers)
        ])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, dtype=dtype)

    def forward(self, input_ids: torch.Tensor, prior_std: float, kl_epsilon: float):
        assert input_ids.max().item() < self.embedding.num_embeddings, "Token ID out of vocab range"
        tok_emb = self.embedding(input_ids)
        x = tok_emb
        
        all_masked_outputs = []
        total_kl_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x, masked_outputs_from_block, kl_from_block = block(x, prior_std, kl_epsilon)
            all_masked_outputs.extend(masked_outputs_from_block)
            total_kl_loss += kl_from_block
            
        logits = self.lm_head(x)
        return logits, all_masked_outputs, total_kl_loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, top_p: float, eos_token_id: int):
        self.eval()
        
        for _ in range(max_new_tokens):
            # The forward pass doesn't need prior_std or kl_epsilon for inference
            logits, _, _ = self.forward(input_ids, prior_std=1.0, kl_epsilon=1e-9)
            
            # Get logits for the last token
            last_token_logits = logits[:, -1, :]
            
            # Apply top-p filtering
            sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            last_token_logits[:, indices_to_remove] = -float("Inf")
            
            # Sample from the filtered distribution
            probs = F.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the new token
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
            # Stop if EOS is generated
            if next_token.item() == eos_token_id:
                break
                
        self.train()
        return input_ids
