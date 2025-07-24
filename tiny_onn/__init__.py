from transformers import AutoConfig, AutoModelForCausalLM

from .config import TinyOnnConfig
from .modular import TinyOnnForCausalLM

AutoConfig.register("tiny_onn", TinyOnnConfig)
AutoModelForCausalLM.register(TinyOnnConfig, TinyOnnForCausalLM)

__all__ = ["TinyOnnConfig", "TinyOnnForCausalLM"]
