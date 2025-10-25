import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GenerationConfig, ModelConfig
from .data import GridDeserializer, GridSerializer


class ExpertMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.w1 = nn.Linear(input_size, output_size, bias=False, dtype=dtype)
        self.w2 = nn.Linear(output_size, input_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w1_out = self.w1(x)
        silu_out = F.silu(w1_out)
        w2_out = self.w2(silu_out)
        return w2_out, w1_out

class DynSIHA(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = config.hidden_size
        self.physical_num_heads = config.physical_num_heads
        self.head_dim = self.d_model // self.physical_num_heads
        self.latent_attn_expert = config.latent_attn_expert

        self.proto_q = nn.Parameter(torch.empty(self.latent_attn_expert, self.head_dim, dtype=dtype))
        self.gate_q = nn.Parameter(torch.empty(self.latent_attn_expert, dtype=dtype))
        self.proto_k = nn.Parameter(torch.empty(self.latent_attn_expert, self.head_dim, dtype=dtype))
        self.gate_k = nn.Parameter(torch.empty(self.latent_attn_expert, dtype=dtype))
        self.proto_v = nn.Parameter(torch.empty(self.latent_attn_expert, self.head_dim, dtype=dtype))
        self.gate_v = nn.Parameter(torch.empty(self.latent_attn_expert, dtype=dtype))

        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.proto_q, std=0.02)
        nn.init.normal_(self.gate_q, mean=0.0, std=0.02)
        nn.init.normal_(self.proto_k, std=0.02)
        nn.init.normal_(self.gate_k, mean=0.0, std=0.02)
        nn.init.normal_(self.proto_v, std=0.02)
        nn.init.normal_(self.gate_v, mean=0.0, std=0.02)

    def _compose(self, x_proj: torch.Tensor, proto: torch.Tensor, gate: torch.Tensor, expert_library: nn.ModuleList, captured_raw_inputs: list | None = None, captured_raw_output_grads: list | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, H, D_h = x_proj.shape
        S = B * T * H
        P = self.latent_attn_expert

        x_reshaped = x_proj.reshape(S, D_h)

        raw_logits = torch.einsum('bthd,pd->bthp', x_proj, proto) / math.sqrt(self.head_dim)
        raw_logits = raw_logits - gate.unsqueeze(0).unsqueeze(0)
        routing_logits = F.relu(raw_logits)

        routing_logits_flat = routing_logits.reshape(S, P)
        active_mask = routing_logits_flat > 1e-6
        token_indices, expert_indices = torch.where(active_mask)

        if token_indices.numel() == 0:
            synthetic_outputs = torch.zeros_like(x_proj)
            raw_outputs_for_return = torch.zeros_like(routing_logits)
            active_inputs = torch.empty((0, D_h), dtype=x_proj.dtype, device=x_proj.device)
            w1_out_active = torch.empty((0, D_h), dtype=x_proj.dtype, device=x_proj.device)
            active_token_indices = torch.empty((0,), dtype=token_indices.dtype, device=token_indices.device)
            active_expert_indices = torch.empty((0,), dtype=expert_indices.dtype, device=expert_indices.device)
            if self.training and captured_raw_inputs is not None:
                captured_raw_inputs.append(x_proj.clone().detach())
            return synthetic_outputs, routing_logits, raw_outputs_for_return, active_inputs, w1_out_active, active_token_indices, active_expert_indices

        active_inputs = x_reshaped[token_indices]
        active_routing_logits = routing_logits_flat[token_indices, expert_indices]

        sorted_expert_indices, perm_indices = torch.sort(expert_indices)
        sorted_inputs = active_inputs[perm_indices]

        expert_counts = torch.bincount(sorted_expert_indices, minlength=P)
        expert_offsets = torch.cumsum(expert_counts, dim=0) - expert_counts

        expert_outputs_sorted = torch.empty_like(sorted_inputs)
        w1_out_active_sorted = torch.empty_like(sorted_inputs)

        for i in range(P):
            if expert_counts[i] > 0:
                start, end = expert_offsets[i], expert_offsets[i] + expert_counts[i]
                expert_in = sorted_inputs[start:end]
                expert_out, w1_out = expert_library[i](expert_in)
                expert_outputs_sorted[start:end] = expert_out
                w1_out_active_sorted[start:end] = w1_out

        gated_outputs_sorted = expert_outputs_sorted * active_routing_logits[perm_indices].unsqueeze(-1)

        inverse_perm_indices = torch.argsort(perm_indices)
        unpermuted_outputs = gated_outputs_sorted[inverse_perm_indices]
        unpermuted_w1_out = w1_out_active_sorted[inverse_perm_indices]

        raw_outputs = torch.zeros(S, P, D_h, dtype=x_proj.dtype, device=x_proj.device)
        
        flat_indices = token_indices * P + expert_indices
        index_for_scatter = flat_indices.unsqueeze(-1).expand(-1, D_h)
        
        raw_outputs.view(S * P, D_h).scatter_add_(0, index_for_scatter, unpermuted_outputs)

        synthetic_outputs_flat = raw_outputs.sum(dim=1)
        synthetic_outputs = synthetic_outputs_flat.view(B, T, H, D_h)
        
        if self.training:
            if captured_raw_inputs is not None:
                captured_raw_inputs.append(x_proj.clone().detach())
            if captured_raw_output_grads is not None and raw_outputs.requires_grad:
                raw_outputs.register_hook(lambda grad: captured_raw_output_grads.append(grad.clone().detach()))
        
        raw_outputs_for_return = raw_outputs.norm(dim=-1)

        return synthetic_outputs, routing_logits, raw_outputs_for_return, active_inputs, unpermuted_w1_out, token_indices, expert_indices

    def forward(self, x: torch.Tensor, expert_library: nn.ModuleList, position_ids: torch.Tensor, captured_raw_inputs: list | None = None, captured_raw_output_grads: list | None = None) -> tuple[torch.Tensor, list, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        B, T, _ = x.shape
        x_proj = x.view(B, T, self.physical_num_heads, self.head_dim)

        q_synthetic, q_logits, q_raw, q_active_inputs, q_w1_out_active, q_token_indices, q_expert_indices = self._compose(x_proj, self.proto_q, self.gate_q, expert_library, captured_raw_inputs, captured_raw_output_grads)
        k_synthetic, k_logits, k_raw, k_active_inputs, k_w1_out_active, k_token_indices, k_expert_indices = self._compose(x_proj, self.proto_k, self.gate_k, expert_library, captured_raw_inputs, captured_raw_output_grads)
        v_synthetic, v_logits, v_raw, v_active_inputs, v_w1_out_active, v_token_indices, v_expert_indices = self._compose(x_proj, self.proto_v, self.gate_v, expert_library, captured_raw_inputs, captured_raw_output_grads)

        q, k, v = q_synthetic.transpose(1, 2), k_synthetic.transpose(1, 2), v_synthetic.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        output = self.o_proj(attn_out)

        all_routing_logits = [q_logits, k_logits, v_logits]
        all_raw_outputs = [q_raw, k_raw, v_raw]

        all_active_inputs = [q_active_inputs, k_active_inputs, v_active_inputs]
        all_w1_out_active = [q_w1_out_active, k_w1_out_active, v_w1_out_active]
        all_token_indices = [q_token_indices, k_token_indices, v_token_indices]
        all_expert_indices = [q_expert_indices, k_expert_indices, v_expert_indices]

        return output, all_routing_logits, all_raw_outputs, all_active_inputs, all_w1_out_active, all_token_indices, all_expert_indices

class DynTRMBlock(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = DynSIHA(config, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        attn_experts: nn.ModuleList,
        position_ids: torch.Tensor | None = None,
        captured_raw_inputs: list | None = None,
        captured_raw_output_grads: list | None = None,
    ) -> tuple[torch.Tensor, list, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:

        attn_input = self.ln1(x)
        attn_output, attn_routing_logits, attn_raw_outputs, attn_active_inputs, attn_w1_out_active, attn_token_indices, attn_expert_indices = self.attn(attn_input, attn_experts, position_ids, captured_raw_inputs, captured_raw_output_grads)
        x = x + attn_output

        return x, attn_routing_logits, attn_raw_outputs, attn_active_inputs, attn_w1_out_active, attn_token_indices, attn_expert_indices

class ArcEmbedding(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.color_embedding = nn.Embedding(config.vocab_size, self.hidden_size, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.color_embedding(input_ids)

class ArcTransformer(nn.Module):
    def __init__(self, config: ModelConfig, generation_config: GenerationConfig, device: torch.device | str):
        super().__init__()
        self.config = config
        self.generation_config = generation_config
        self.device = device
        dtype = torch.float32

        self.embedding = ArcEmbedding(config, dtype=dtype)

        self.attn_expert_library = nn.ModuleList(
            [ExpertMLP(config.hidden_size // config.physical_num_heads, config.hidden_size // config.physical_num_heads, dtype=dtype) for _ in range(config.latent_attn_expert)]
        )

        self.block = DynTRMBlock(config, dtype=dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        return_dict: bool = False,
        captured_raw_inputs: list | None = None,
        captured_raw_output_grads: list | None = None,
        **kwargs,
    ):
        x = self.embedding(input_ids)

        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)

        all_hidden_states, all_routing_logits = [], []
        all_raw_outputs: list[torch.Tensor] = []
        all_active_inputs: list[torch.Tensor] = []
        all_w1_out_active: list[torch.Tensor] = []
        all_token_indices: list[torch.Tensor] = []
        all_expert_indices: list[torch.Tensor] = []

        current_step = 0
        while current_step < self.config.max_refinement_steps:
            x, layer_routing_logits, layer_raw_outputs, layer_active_inputs, layer_w1_out_active, layer_token_indices, layer_expert_indices = self.block(
                x, self.attn_expert_library,
                position_ids=position_ids,
                captured_raw_inputs=captured_raw_inputs,
                captured_raw_output_grads=captured_raw_output_grads
            )
            all_hidden_states.append(x)
            all_routing_logits.extend(layer_routing_logits)
            all_raw_outputs.extend(layer_raw_outputs)
            all_active_inputs.extend(layer_active_inputs)
            all_w1_out_active.extend(layer_w1_out_active)
            all_token_indices.extend(layer_token_indices)
            all_expert_indices.extend(layer_expert_indices)

            current_step += 1

        logits = self.lm_head(x)

        if not return_dict:
            return (logits, all_hidden_states, all_routing_logits, all_raw_outputs, all_active_inputs, all_w1_out_active, all_token_indices, all_expert_indices)

        return {
            "logits": logits,
            "all_hidden_states": all_hidden_states,
            "all_routing_logits": all_routing_logits,
            "all_raw_outputs": all_raw_outputs,
            "all_active_inputs": all_active_inputs,
            "all_w1_out_active": all_w1_out_active,
            "all_token_indices": all_token_indices,
            "all_expert_indices": all_expert_indices,
        }

    @torch.no_grad()
    def generate(self, serializer: GridSerializer, deserializer: GridDeserializer, task_data: dict[str, Any]) -> tuple[torch.Tensor, list[int], list[float]]:
        self.eval()

        prompt_ids = serializer.serialize_for_inference(task_data)[0]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        output_grid = task_data["test"][0].get("output", [])
        num_rows = len(output_grid)
        num_pixels = sum(len(row) for row in output_grid)
        max_new_tokens = num_pixels + num_rows + 2

        probabilities = []

        for _ in range(max_new_tokens):
            position_ids = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0)

            outputs = self.forward(input_ids, position_ids=position_ids, return_dict=True)

            logits = outputs["logits"]
            next_token_logits = logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(next_token_probs, dim=-1, keepdim=True)

            prob = next_token_probs[0, next_token.item()].item()
            probabilities.append(prob)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == serializer.tokenizer.eos_token_id or next_token.item() == serializer.tokenizer.vocab["<im_end>"]:
                break

        generated_tokens = input_ids[0, len(prompt_ids):].tolist()
        pred_grid = deserializer.deserialize(generated_tokens)
        return pred_grid, generated_tokens, probabilities
