from typing import Any

import torch
from transformers import PreTrainedTokenizer

from .backward_pass import run_per_token_backward_pass
from .forward_pass import run_forward_pass_and_capture_activations


def run_fmri_scan(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    user_message: str,
    history: list[dict],
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> tuple[str, list[dict[str, Any]], torch.Tensor, int]:
    print("LOG: --- [START] fMRI Scan: Per-Token Forward & Backward Pass ---")

    (
        final_response,
        per_token_data,
        full_sequence_ids,
        prompt_len,
    ) = run_forward_pass_and_capture_activations(
        model, tokenizer, user_message, history, temperature=temperature, top_p=top_p
    )

    if per_token_data:
        per_token_data = run_per_token_backward_pass(
            model, full_sequence_ids, per_token_data, prompt_len
        )

    print("LOG: --- [END] fMRI Scan ---")

    return final_response, per_token_data, full_sequence_ids, prompt_len


__all__ = [
    "run_fmri_scan",
]
