
from typing import Any

import torch
from transformers import PreTrainedTokenizer

from .backward_pass import run_per_token_backward_pass, run_full_sequence_backward_pass
from .forward_pass import run_forward_pass_and_capture_activations


def run_fmri_scan(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    messages: list[dict],
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 256,
    scan_mode: str = "per_token",
) -> tuple[str, list[dict[str, Any]], torch.Tensor, int]:
    print(f"LOG: --- [START] fMRI Scan: {scan_mode.capitalize()} Mode ---")

    (
        final_response,
        per_token_data,
        full_sequence_ids,
        prompt_len,
    ) = run_forward_pass_and_capture_activations(
        model,
        tokenizer,
        messages,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        scan_mode=scan_mode,
    )

    if per_token_data:
        if scan_mode == "per_token":
            per_token_data = run_per_token_backward_pass(
                model, full_sequence_ids, per_token_data, prompt_len
            )
        elif scan_mode == "full_sequence":
            per_token_data = run_full_sequence_backward_pass(
                model, full_sequence_ids, per_token_data, prompt_len
            )
        else:
            print(f"WARNING: Unknown scan_mode '{scan_mode}'. No backward pass performed.")

    print(f"LOG: --- [END] fMRI Scan: {scan_mode.capitalize()} Mode ---")

    return final_response, per_token_data, full_sequence_ids, prompt_len


__all__ = [
    "run_fmri_scan",
]
