from typing import Any, cast

import torch
from transformers import (
    BatchEncoding,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def run_forward_pass_and_capture_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 256,
    scan_mode: str = "per_token",
) -> tuple[str, list[dict[str, Any]], torch.Tensor, int]:

    inputs_raw = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    if not isinstance(inputs_raw, BatchEncoding):
        raise TypeError(f"Expected BatchEncoding, but got {type(inputs_raw)}")

    inputs = cast(BatchEncoding, inputs_raw).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    full_sequence_ids, per_token_data = model.generate(
        **inputs,
        generation_config=generation_config,
        custom_generate="scanner/engine",
        trust_remote_code=True,
        tokenizer=tokenizer,
        scan_mode=scan_mode,
    )

    final_response = tokenizer.decode(
        full_sequence_ids[prompt_len:], skip_special_tokens=True
    )

    return final_response, per_token_data, full_sequence_ids, prompt_len
