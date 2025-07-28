from torch import nn

from tiny_onn.modular import TinyOnnExpert, TinyOnnMoE


class ExpertManager:
    def __init__(self, model: nn.Module, regeneration_interval: int):
        self.model = model
        self.regeneration_interval = regeneration_interval

    def check_and_regenerate(self, step: int):
        if step > 0 and step % self.regeneration_interval == 0:
            self._regenerate_experts()

    def _regenerate_experts(self):
        for module in self.model.modules():
            if isinstance(module, TinyOnnMoE):
                dead_experts_indices = (module.routing_records == 0).nonzero(
                    as_tuple=True
                )[0]

                if dead_experts_indices.numel() > 0:
                    for expert_idx in dead_experts_indices:
                        expert_to_reset = module.experts[expert_idx.item()]
                        self._reinit_expert(expert_to_reset)

                module.reset_routing_records()

    def _reinit_expert(self, expert: TinyOnnExpert):
        for param in expert.parameters():
            if param.dim() > 1:
                nn.init.kaiming_uniform_(param, a=5**0.5)
