from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from .config import TinyOnnConfig
from .modular import TinyOnnForCausalLM

AutoConfig.register("tiny_onn", TinyOnnConfig)
AutoModelForCausalLM.register(TinyOnnConfig, TinyOnnForCausalLM)
AutoTokenizer.register(TinyOnnConfig, Qwen2Tokenizer, Qwen2TokenizerFast)

__all__ = ["TinyOnnConfig", "TinyOnnForCausalLM"]
