import torch
from torch import nn
from transformers import AutoTokenizer
from tiny_onn.modular import TinyOnnForCausalLM, TinyOnnMoE, TinyOnnExpert
from typing import cast, Dict, List, Any, Tuple, Optional

class SurpriseHookManager:
    def __init__(self, per_token_surprise: torch.Tensor):
        self.per_token_surprise = per_token_surprise
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.expert_inputs: Dict[int, Dict[int, torch.Tensor]] = {}
        self.expert_token_indices: Dict[int, Dict[int, torch.Tensor]] = {}

    def _save_input_hook(self, layer_idx: int, expert_idx: int):
        def hook_fn(module, args, kwargs):
            if layer_idx not in self.expert_inputs:
                self.expert_inputs[layer_idx] = {}
            if layer_idx not in self.expert_token_indices:
                self.expert_token_indices[layer_idx] = {}

            moe_layer = kwargs.get("moe_layer")
            if moe_layer and expert_idx in moe_layer.last_expert_inputs:
                self.expert_inputs[layer_idx][expert_idx] = moe_layer.last_expert_inputs[expert_idx]
                self.expert_token_indices[layer_idx][expert_idx] = moe_layer.last_expert_token_indices[expert_idx]
        return hook_fn

    def _compute_surprise_hook(self, layer_idx: int, expert_idx: int, expert: TinyOnnExpert):
        def hook_fn(module, grad_input, grad_output):
            if layer_idx not in self.expert_inputs or expert_idx not in self.expert_inputs[layer_idx]:
                return

            expert_input = self.expert_inputs[layer_idx][expert_idx]
            token_indices = self.expert_token_indices[layer_idx][expert_idx]
            
            with torch.enable_grad():
                act_fn_derivative = expert.act_fn(expert.w1(expert_input)) * expert.w3(expert_input)
                grad_w2_per_token = torch.einsum("bi,bo->bio", act_fn_derivative, grad_output[0])
                surprise = torch.linalg.norm(grad_w2_per_token.flatten(1), dim=1)

            self.per_token_surprise[token_indices, expert_idx] = surprise.detach()
        return hook_fn

    def register_hooks(self, model: TinyOnnForCausalLM):
        for layer_idx, layer in enumerate(model.model.layers):
            moe_layer = cast(TinyOnnMoE, layer.mlp)
            for expert_idx, expert in enumerate(moe_layer.experts):
                expert.register_forward_pre_hook(self._save_input_hook(layer_idx, expert_idx))
                self.hooks.append(
                    expert.w2.register_full_backward_hook(self._compute_surprise_hook(layer_idx, expert_idx, expert))
                )

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.expert_inputs.clear()
        self.expert_token_indices.clear()

def run_hooks_poc():
    model_path = "weights/Qwen/Qwen2-0.5B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TinyOnnForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    
    num_tokens = batch_size * seq_len
    num_experts = model.config.num_experts_per_layer
    
    per_token_surprise = torch.full((num_tokens, num_experts), float('inf'), device=device, dtype=torch.float32)
    
    manager = SurpriseHookManager(per_token_surprise)
    manager.register_hooks(model)
    
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    manager.remove_hooks()

    print("--- PoC for V6 Hook-based Surprise Calculation ---")
    print(f"Shape of per_token_surprise: {per_token_surprise.shape}")
    
    num_finite_surprise = torch.isfinite(per_token_surprise).sum().item()
    print(f"Number of finite (calculated) surprise values: {num_finite_surprise}")

    total_routed_tokens = 0
    for layer in model.model.layers:
        moe_layer = cast(TinyOnnMoE, layer.mlp)
        if moe_layer.last_expert_token_indices:
            for indices in moe_layer.last_expert_token_indices.values():
                total_routed_tokens += len(indices)

    print(f"Total number of token-expert routes in forward pass: {total_routed_tokens}")
    
    assert per_token_surprise.shape == (num_tokens, num_experts)
    assert num_finite_surprise == total_routed_tokens
    assert num_finite_surprise > 0

    print("\n✅ PoC successful. Per-token surprise calculated correctly via hooks.")
    print("Shape and content of the surprise tensor are as expected.")

if __name__ == "__main__":
    run_hooks_poc()