import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch
from transformers import AutoTokenizer

from scripts.perform_surgery import perform_surgery


@pytest.fixture(scope="module")
def surgical_model_and_tokenizer():
    base_model_name = "Qwen/Qwen3-0.6B"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    cache_dir = os.path.join(os.path.dirname(script_dir), "weights")
    os.makedirs(cache_dir, exist_ok=True)

    model, tokenizer = perform_surgery(base_model_name, cache_dir=cache_dir)
    return model, tokenizer


def test_surgery_and_generation(surgical_model_and_tokenizer):
    model, tokenizer = surgical_model_and_tokenizer
    assert model is not None
    assert tokenizer is not None

    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)

        prompt = "The quick brown fox jumps over the lazy dog"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(**inputs, max_new_tokens=20)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\n--- Generated Text ---\n{generated_text}\n----------------------\n")

    except Exception as e:
        pytest.fail(f"Generation failed with an exception: {e}")
