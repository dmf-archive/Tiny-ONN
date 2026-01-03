from .flat_dynsiha.config import FlatDynSIHAConfig
from .flat_dynsiha.modular import FlatDynSIHAForCausalLM

MODEL_REGISTRY = {
    "flat_dynsiha": FlatDynSIHAForCausalLM,
}


def get_model(config):
    model_type = getattr(config, "model_type", None)
    if model_type is None and hasattr(config, "dict"):
        model_type = config.model_type

    model_cls = MODEL_REGISTRY.get(model_type)
    if not model_cls:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type == "flat_dynsiha" and not isinstance(config, FlatDynSIHAConfig):
        hf_config = FlatDynSIHAConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            num_physical_heads=config.physical_num_heads,
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            intermediate_size=config.hidden_size * config.ffn_scale,
        )
        return model_cls(hf_config)

    return model_cls(config)
