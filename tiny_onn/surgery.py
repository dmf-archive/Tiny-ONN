from transformers import AutoModelForCausalLM
from .config import TinyOnnConfig
from .modular import TinyOnnForCausalLM


def perform_surgery(
    base_model_name: str, cache_dir: str, **kwargs
) -> TinyOnnForCausalLM:
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, cache_dir=cache_dir, trust_remote_code=True
    )
    
    base_config = base_model.config
    
    tiny_onn_config_dict = base_config.to_dict()
    tiny_onn_config_dict.update(kwargs)
    
    config = TinyOnnConfig.from_dict(tiny_onn_config_dict)
    
    model = TinyOnnForCausalLM(config)
    
    model.load_state_dict(base_model.state_dict(), strict=False)
    
    return model
