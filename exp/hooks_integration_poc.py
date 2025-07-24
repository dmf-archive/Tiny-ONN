import torch
from torch import nn
from transformers import AutoTokenizer
from tiny_onn.modular import TinyOnnForCausalLM, TinyOnnMoE, TinyOnnExpert
from typing import cast, Dict, List, Any, Tuple, Optional
import time

class SurpriseHookManager:
    def __init__(self, per_token_surprise: torch.Tensor):
        self.per_token_surprise = per_token_surprise
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.expert_inputs: Dict[str, torch.Tensor] = {}
        self.expert_token_indices: Dict[str, torch.Tensor] = {}

    def _get_key(self, layer_idx: int, expert_idx: int) -> str:
        return f"layer_{layer_idx}_expert_{expert_idx}"

    def _save_input_hook(self, layer_idx: int, expert_idx: int, moe_layer: "TinyOnnMoE"):
        def hook_fn(module, args):
            key = self._get_key(layer_idx, expert_idx)
            if expert_idx in moe_layer.last_expert_inputs:
                self.expert_inputs[key] = moe_layer.last_expert_inputs[expert_idx].detach().clone()
                self.expert_token_indices[key] = moe_layer.last_expert_token_indices[expert_idx].clone()
        return hook_fn

    def _compute_surprise_hook(self, layer_idx: int, expert_idx: int, expert: "TinyOnnExpert"):
        def hook_fn(module, grad_input, grad_output):
            key = self._get_key(layer_idx, expert_idx)
            if key not in self.expert_inputs:
                return

            expert_input = self.expert_inputs[key]
            expert_input.requires_grad = True
            token_indices = self.expert_token_indices[key]
            
            with torch.enable_grad():
                expert_output = expert(expert_input)
                expert_output.backward(grad_output[0])

            if expert_input.grad is not None:
                surprise = torch.linalg.norm(expert_input.grad.flatten(start_dim=1), dim=1)
                self.per_token_surprise[token_indices, expert_idx] = surprise.detach()
        return hook_fn

    def register_hooks(self, model: "TinyOnnForCausalLM"):
        from tiny_onn.modular import TinyOnnMoE, TinyOnnExpert
        for layer_idx, layer in enumerate(model.model.layers):
            moe_layer = cast(TinyOnnMoE, layer.mlp)
            for expert_idx, expert in enumerate(moe_layer.experts):
                self.hooks.append(
                    expert.register_forward_pre_hook(self._save_input_hook(layer_idx, expert_idx, moe_layer))
                )
                self.hooks.append(
                    expert.register_full_backward_hook(self._compute_surprise_hook(layer_idx, expert_idx, expert))
                )

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.expert_inputs.clear()
        self.expert_token_indices.clear()

def run_hooks_poc():
    model_path = "weights/Tiny-ONN-0.6B-Hyper-SMoE"
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
    
    start_time = time.time()
    loss.backward()
    end_time = time.time()
    
    manager.remove_hooks()
    
    num_finite_surprise = torch.isfinite(per_token_surprise).sum().item()
    total_routed_tokens_in_forward = 0
    for layer in model.model.layers:
        moe_layer = cast(TinyOnnMoE, layer.mlp)
        if moe_layer.last_expert_token_indices:
            for indices in moe_layer.last_expert_token_indices.values():
                total_routed_tokens_in_forward += len(indices)

    print("--- PoC for V6 Hook-based Surprise Calculation ---")
    print(f"Time for backward pass + surprise calculation: {end_time - start_time:.4f} seconds")
    print(f"Shape of per_token_surprise: {per_token_surprise.shape}")
    print(f"Number of finite (calculated) surprise values: {num_finite_surprise}")
    print(f"Total number of token-expert routes in forward pass: {total_routed_tokens_in_forward}")
    
    assert per_token_surprise.shape == (num_tokens, num_experts)
    assert num_finite_surprise > 0, "No surprise values were calculated!"
    assert num_finite_surprise == total_routed_tokens_in_forward, "Mismatch between calculated surprises and routed tokens!"
    
    print("\n✅ PoC successful. Per-token surprise calculated correctly via hooks.")

if __name__ == "__main__":
    run_hooks_poc()