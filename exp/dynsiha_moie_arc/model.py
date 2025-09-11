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
        nn.init.normal_(self.sigma_weight, mean=2.0, std=0.5)
        nn.init.constant_(self.gate_param, -0.1)
        nn.init.zeros_(self.mu_bias)

    def forward(self, x: torch.Tensor, prior_std: float, kl_epsilon: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        x_reshaped = x.view(-1, self.in_features)

        keys = self.mu_weight * F.softplus(self.sigma_weight)
        scores = torch.matmul(x_reshaped, keys.t()) / math.sqrt(self.in_features)
        raw_weights = F.relu(scores - self.gate_param.unsqueeze(0))

        computation_output = F.linear(x_reshaped, self.mu_weight, self.mu_bias)
        masked_output = computation_output * raw_weights

        new_shape = list(original_shape[:-1]) + [self.out_features]
        output = masked_output.view(new_shape)

        # Unified KL divergence with zero-mean prior
        mu_q = self.mu_weight
        sigma_q = F.softplus(self.sigma_weight)
        var_q = sigma_q.pow(2)
        
        mu_p = torch.zeros_like(mu_q)
        var_p = torch.full_like(sigma_q, prior_std).pow(2)

        kl_div = 0.5 * (torch.log(var_p / (var_q + kl_epsilon)) + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1)
        kl_loss = kl_div.mean()

        if torch.isnan(kl_loss).any() or torch.isinf(kl_loss).any():
            print(f"SBL KL Loss is NaN/Inf!")
            print(f"var_p: {var_p.mean().item()}, var_q: {var_q.mean().item()}, mu_norm: {self.mu_weight.pow(2).mean().item()}")

        return output, masked_output, computation_output, raw_weights, kl_loss

    def update_priors(self) -> None:
        """
        A method to be called after each optimizer step to update the priors.
        This is now handled inside forward to be safer with JIT and graph tracing.
        Kept as a placeholder in case of future design changes.
        """
        pass

class DynamicInfiniteHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = config.hidden_size
        self.sbl_qkv = SparseBayesianLinear(self.d_model, 3 * self.d_model, dtype=dtype)
        self.sbl_o = SparseBayesianLinear(self.d_model, self.d_model, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        prior_std: float,
        kl_epsilon: float,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        qkv, m_qkv, c_qkv, rw_qkv, kl_qkv = self.sbl_qkv(x, prior_std, kl_epsilon)
        q, k, v = torch.split(qkv, self.d_model, dim=-1)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        present_key_value = (k, v)

        is_causal = past_key_value is None
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        y, m_o, c_o, rw_o, kl_o = self.sbl_o(attn_out, prior_std, kl_epsilon)

        total_kl_loss = kl_qkv + kl_o
        masked_outputs: list[torch.Tensor] = [m_qkv, m_o]
        comp_outputs: list[torch.Tensor] = [c_qkv, c_o]
        raw_weights: list[torch.Tensor] = [rw_qkv, rw_o]

        return y, masked_outputs, comp_outputs, raw_weights, total_kl_loss, present_key_value

class DynamicInfiniteExpert(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        d_ffn = config.hidden_size * config.d_ffn_factor
        self.sbl1 = SparseBayesianLinear(config.hidden_size, d_ffn, dtype=dtype)
        self.sbl2 = SparseBayesianLinear(d_ffn, config.hidden_size, dtype=dtype)

    def forward(self, x: torch.Tensor, prior_std: float, kl_epsilon: float) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        h, m1, c1, rw1, kl1 = self.sbl1(x, prior_std, kl_epsilon)
        h_act = F.relu(h)
        y, m2, c2, rw2, kl2 = self.sbl2(h_act, prior_std, kl_epsilon)

        total_kl_loss = kl1 + kl2
        masked_outputs: list[torch.Tensor] = [m1, m2]
        comp_outputs: list[torch.Tensor] = [c1, c2]
        raw_weights: list[torch.Tensor] = [rw1, rw2]

        return y, masked_outputs, comp_outputs, raw_weights, total_kl_loss

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
        prior_std: float,
        kl_epsilon: float,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_in = self.ln1(x)
        attn_out, attn_m, attn_c, attn_rw, attn_kl, present_key_value = self.attn(
            attn_in, position_embeddings, prior_std, kl_epsilon, past_key_value
        )
        x = x + attn_out

        ffn_in = self.ln2(x)
        ffn_out, ffn_m, ffn_c, ffn_rw, ffn_kl = self.ffn(ffn_in, prior_std, kl_epsilon)
        x = x + ffn_out

        layer_entropy = torch.tensor(0.0, device=x.device)
        # The with torch.no_grad() is not JIT-compatible
        # with torch.no_grad():
        #     probs = F.softmax(x, dim=-1)
        #     layer_entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()

        total_kl_loss = attn_kl + ffn_kl
        masked_outputs = attn_m + ffn_m
        comp_outputs = attn_c + ffn_c
        raw_weights = attn_rw + ffn_rw

        return x, masked_outputs, comp_outputs, raw_weights, total_kl_loss, layer_entropy, present_key_value

class ArcTransformer(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device | str):
        super().__init__()
        self.config = config # Keep for non-JIT access if needed
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size

        self.config.kl_prior_epsilon = 1e-9 # Used in recalculate_kl_prior
        dtype = torch.bfloat16 # Hardcode for now as per project spec

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
        prior_std: float,
       kl_epsilon: float,
       past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
    ):
        assert input_ids.max().item() < self.embedding.num_embeddings, "Token ID out of vocab range"
        tok_emb = self.embedding(input_ids)
        x = tok_emb # No more pos_emb

        seq_len = input_ids.size(1)
        position_embeddings = self.rotary_emb(x, seq_len=seq_len)

        if past_key_values is None:
            # JIT-compatible way to initialize a list of Nones with the correct type hint
            pkv: list[tuple[torch.Tensor, torch.Tensor] | None] = []
            for _ in range(self.num_layers):
                pkv.append(None)
            past_key_values = pkv

        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        all_masked_outputs: list[torch.Tensor] = []
        all_comp_outputs: list[torch.Tensor] = []
        all_raw_weights: list[torch.Tensor] = []
        layer_entropies: list[torch.Tensor] = []
        total_kl_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for i, block in enumerate(self.blocks):
            x, masked_outputs, comp_outputs, raw_weights, kl, layer_entropy, present_key_value = block(
                x, position_embeddings, prior_std, kl_epsilon, past_key_values[i]
            )
            present_key_values.append(present_key_value)
            all_masked_outputs.extend(masked_outputs)
            all_comp_outputs.extend(comp_outputs)
            all_raw_weights.extend(raw_weights)
            total_kl_loss += kl
            layer_entropies.append(layer_entropy)

        logits = self.lm_head(x)

        layer_taus = torch.stack(layer_entropies)

        # JIT requires consistent return types. We will return a list of tuples.
        return logits, all_masked_outputs, all_comp_outputs, all_raw_weights, total_kl_loss, layer_taus, present_key_values

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int, pad_token_id: int | None = None, use_cache: bool = True):
        self.eval()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            if past_key_values is None or not use_cache:
                model_input = input_ids
            else:
                model_input = input_ids[:, -1:]

            logits, _, _, _, _, _, pkv = self.forward(
                model_input, prior_std=1.0, kl_epsilon=self.config.kl_prior_epsilon, past_key_values=past_key_values if use_cache else None
            )
            if use_cache:
                past_key_values = pkv

            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        
        self.train()
        return input_ids

    @torch.no_grad()
    def dfs_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int,
        threshold: float
    ) -> list[tuple[float, torch.Tensor]]:
        self.eval()
        
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.num_layers
        valid_sequences: list[tuple[float, torch.Tensor]] = []
        max_len = input_ids.shape[1] + max_new_tokens

        def _explore(tokens: torch.Tensor, score: float):
            nonlocal past_key_values
            if tokens.shape[1] >= max_len or tokens[0, -1] == eos_token_id:
                if tokens[0, -1] == eos_token_id:
                    valid_sequences.append((score, tokens.clone()))
                return

            len_before_recursion = tokens.shape[1]
            model_input = tokens if past_key_values[0] is None else tokens[:, -1:]

            logits, _, _, _, _, _, past_key_values = self.forward(
                model_input, prior_std=1.0, kl_epsilon=1e-9, past_key_values=past_key_values
            )

            next_token_logits = logits[:, -1, :]
            next_token_log_prob = F.log_softmax(next_token_logits, dim=-1)
            
            top_k_log_probs, top_k_indices = torch.topk(next_token_log_prob, k=50, dim=-1)

            for i in range(top_k_indices.shape[1]):
                token_id = top_k_indices[0, i]
                log_prob = top_k_log_probs[0, i].item()
                next_score = score + log_prob

                if next_score >= threshold:
                    next_tokens = torch.cat([tokens, token_id.view(1, 1)], dim=1)
                    _explore(next_tokens, next_score)

            # KV Cache Pruning / Backtracking
            if past_key_values[0] is not None:
                trimmed_pkv: list[tuple[torch.Tensor, torch.Tensor] | None] = []
                for k_v_pair in past_key_values:
                    if k_v_pair is not None:
                        k, v = k_v_pair
                        k = k[:, :len_before_recursion, :]
                        v = v[:, :len_before_recursion, :]
                        trimmed_pkv.append((k, v))
                    else:
                        trimmed_pkv.append(None)
                past_key_values = trimmed_pkv

        _explore(input_ids.clone(), 0.0)

        self.train()
        return sorted(valid_sequences, key=lambda x: x[0], reverse=True)
