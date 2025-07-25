import os
import shutil
import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer, Qwen3Config

# Ensure scripts can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.perform_surgery import perform_surgery


@pytest.fixture(scope="session")
def tiny_test_model_and_tokenizer():
    """
    Creates a tiny, randomly initialized Qwen3 model, performs surgery on it,
    and returns the resulting TinyOnn model and tokenizer.
    This is a session-scoped fixture to avoid re-creating the model for every test.
    """
    # 1. Create a tiny config for a standard Qwen3 model
    base_config = Qwen3Config(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=256, # Standard MLP size
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=1000,
    )

    # 2. Create a random base model from this config
    # We use TinyOnnModel here for convenience as it's already a Qwen3Model subclass
    from transformers import Qwen3ForCausalLM
    base_model = Qwen3ForCausalLM(base_config)

    # 3. Use a real tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", cache_dir="weights")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Perform surgery on the tiny base model object
    tiny_onn_model, _ = perform_surgery(
        base_model_name="Qwen/Qwen3-0.6B", # Only used for tokenizer and base config props
        cache_dir="weights",
        base_model=base_model,
        num_experts_per_layer=4,
        num_experts_per_tok=2,
    )

    yield tiny_onn_model, tokenizer

    # Teardown: clean up any created directories if necessary
    test_output_dir = Path("tests/test_output")
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)
