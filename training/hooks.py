from collections import defaultdict

import torch
import torch.nn as nn


class GradientInterceptor:
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []
        self.surprises: dict[int, torch.Tensor] = defaultdict(list)
        self.token_info: dict[str, torch.Tensor] = {}
        self._attach_hooks()

    def _forward_hook(
        self, module: nn.Module, input_tuple: tuple[torch.Tensor], output: torch.Tensor
    ):
        if torch.is_grad_enabled():
            module._saved_input = input_tuple[0].detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor],
        grad_output: tuple[torch.Tensor],
    ):
        if not hasattr(module, "_saved_input"):
            return

        input_tensor = module._saved_input
        grad_output_tensor = grad_output[0]

        per_token_grad = torch.einsum("bi,bo->boi", input_tensor, grad_output_tensor)
        surprise = torch.linalg.vector_norm(per_token_grad, dim=(1, 2))

        expert_id = self._find_expert_id(module)
        if expert_id is not None:
            self.surprises[expert_id].append(surprise)

        del module._saved_input

    def _find_expert_id(self, module_to_find: nn.Module) -> int:
        for i, expert in enumerate(self.model.model.layers[0].mlp.experts):
            if expert is module_to_find:
                return i
        return None

    def _attach_hooks(self):
        for layer in self.model.model.layers:
            for _, expert_module in enumerate(layer.mlp.experts):
                handle = expert_module.register_full_backward_hook(self._backward_hook)
                self.handles.append(handle)
                expert_module.register_forward_hook(self._forward_hook)

    def get_surprises(self) -> dict[int, torch.Tensor]:
        # Consolidate surprise tensors for each expert
        consolidated_surprises = {
            expert_id: torch.cat(tensors)
            for expert_id, tensors in self.surprises.items()
        }
        return consolidated_surprises

    def clear(self):
        self.surprises.clear()

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
