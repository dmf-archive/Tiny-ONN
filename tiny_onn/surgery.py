
from .config import TinyOnnConfig
from .model import HierarchicalMoE, TinyOnnForCausalLM


def perform_surgery(
    model: TinyOnnForCausalLM, config: TinyOnnConfig
) -> TinyOnnForCausalLM:
    model.config = config

    for layer in model.model.layers:
        layer.mlp = HierarchicalMoE(config).to(model.device, dtype=model.dtype)

    return model
