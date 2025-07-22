from collections import defaultdict

import torch
import torch.nn as nn


class GradientInterceptor:
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.surprises: dict[nn.Module, list[torch.Tensor]] = defaultdict(list)
        self.expert_to_id: dict[nn.Module, int] = {}
        self._attach_hooks()

    def _forward_hook(
        self,
        module: nn.Module,
        input_tuple: tuple[torch.Tensor],
        output: torch.Tensor,
    ):
        if torch.is_grad_enabled() and isinstance(input_tuple[0], torch.Tensor):
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

        per_token_grad_norms = torch.einsum(
            "bi,bi->b", input_tensor, grad_output_tensor
        )
        surprise = torch.linalg.vector_norm(per_token_grad_norms, dim=-1)

        self.surprises[module].append(surprise)

        del module._saved_input

    def _attach_hooks(self):
        expert_id_counter = 0
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for layer in self.model.model.layers:
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                    for expert_module in layer.mlp.experts:
                        self.expert_to_id[expert_module] = expert_id_counter
                        expert_id_counter += 1
                        handle = expert_module.register_full_backward_hook(
                            self._backward_hook
                        )
                        self.handles.append(handle)
                        expert_module.register_forward_hook(self._forward_hook)

    def get_surprises(self) -> dict[nn.Module, torch.Tensor]:
        return {
            expert: torch.cat(tensors, dim=0)
            for expert, tensors in self.surprises.items()
            if tensors
        }

    def clear(self):
        self.surprises.clear()

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
