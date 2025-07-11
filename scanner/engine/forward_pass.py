from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from bitsandbytes.nn import Linear4bit
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from common.models.pruned_layers import PrunedQwen3DecoderLayer


def _patched_forward(module, capture_state, original_forward, *args, **kwargs):
    x = args[0]
    if (activation_data := capture_state.get("capture_dict")) is None:
        return original_forward(x, **kwargs)

    param_name = module.param_name
    num_features = x.shape[-1]
    block_size = 64
    num_blocks = (num_features + block_size - 1) // block_size

    padded_x = F.pad(x, (0, num_blocks * block_size - num_features))
    reshaped_x = padded_x.view(*x.shape[:-1], num_blocks, block_size)
    block_norms = torch.norm(reshaped_x.float(), p=2, dim=-1)

    activation_data[param_name] = {"activation": block_norms.flatten().tolist()}
    if hasattr(module.weight, "quant_state") and module.weight.quant_state:
        absmax = module.weight.quant_state.absmax.tolist()
        activation_data[param_name]["weight"] = (
            absmax if len(absmax) == num_blocks else [absmax[0]] * num_blocks
        )

    return original_forward(x, **kwargs)


def _top_p_filtering(logits, top_p=0.9, filter_value=-float("Inf")):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = filter_value
    return logits


def _sample_next_token(logits, temperature, top_p):
    if temperature > 0:
        logits = logits / temperature
        logits = _top_p_filtering(logits, top_p=top_p)
        return torch.multinomial(F.softmax(logits, dim=-1), 1)
    return torch.argmax(logits, dim=-1).unsqueeze(-1)


def run_forward_pass_and_capture_activations(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    user_message: str,
    history: list[dict],
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> tuple[str, list[dict[str, Any]], torch.Tensor, int]:
    messages = [*history, {"role": "user", "content": user_message}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer([text], return_tensors="pt").to(model.device)["input_ids"]

    hook_modules = []
    for name, module in model.named_modules():
        if isinstance(module, Linear4bit):
            parent_name = ".".join(name.split(".")[:-2])
            parent_module = model.get_submodule(parent_name) if parent_name else None
            if isinstance(parent_module, PrunedQwen3DecoderLayer) and (
                (parent_module.prune_self_attn and "self_attn" in name)
                or (parent_module.prune_mlp and "mlp" in name)
            ):
                continue
            module.param_name = name
            hook_modules.append(module)

    generated_ids, per_token_block_data = [], []
    model.eval()
    capture_state: dict[str, dict[str, Any] | None] = {"capture_dict": None}
    original_forwards = {m: m.forward for m in hook_modules}

    try:
        for module in hook_modules:
            module.forward = partial(
                _patched_forward, module, capture_state, module.forward
            )

        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_id = _sample_next_token(
                outputs.logits[:, -1, :], temperature, top_p
            )

            for _ in tqdm(range(512), desc="Generating Tokens (fMRI Forward)"):
                capture_state["capture_dict"] = {}
                outputs = model(
                    input_ids=next_token_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                per_token_block_data.append(capture_state["capture_dict"])
                past_key_values = outputs.past_key_values
                next_token_id = _sample_next_token(
                    outputs.logits[:, -1, :], temperature, top_p
                )
                token_id = next_token_id.item()
                generated_ids.append(token_id)
                if token_id == tokenizer.eos_token_id:
                    break
    finally:
        for module, original_forward in original_forwards.items():
            module.forward = original_forward
            delattr(module, "param_name")

    final_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_sequence_ids = torch.cat(
        [input_ids[0], torch.tensor(generated_ids, device=model.device.type)]
    )
    return final_response, per_token_block_data, full_sequence_ids, input_ids.shape[1]
