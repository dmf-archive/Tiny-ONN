from functools import partial

import torch
import torch.nn as nn


class ScannerEngine:
    def __init__(self, model: nn.Module, target_modules: list[type[nn.Module]] | None = None):
        self._model = model
        self._target_modules = target_modules or [nn.Linear]
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.activations: dict[str, torch.Tensor] = {}
        self.gradients: dict[str, torch.Tensor] = {}
        self._param_to_name: dict[int, str] = {
            id(p): n for n, p in model.named_parameters()
        }

    def _forward_hook(self, module: nn.Module, input: tuple[torch.Tensor], output: torch.Tensor, param_name: str):
        self.activations[param_name] = output.detach().cpu()

    def _tensor_hook(self, grad: torch.Tensor, name: str):
        self.gradients[name] = grad.detach().cpu()

    def _attach_hooks(self):
        for name, module in self._model.named_modules():
            if any(isinstance(module, t) for t in self._target_modules):
                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        full_param_name = f"{name}.{param_name}"
                        self._handles.append(
                            module.register_forward_hook(
                                partial(self._forward_hook, param_name=full_param_name)
                            )
                        )
                        self._handles.append(
                            param.register_hook(
                                partial(self._tensor_hook, name=full_param_name)
                            )
                        )

    def _remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def get_collected_data(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return self.activations, self.gradients

    def clear_collected_data(self):
        self.activations.clear()
        self.gradients.clear()

    def __enter__(self):
        self.clear_collected_data()
        self._attach_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._remove_hooks()
