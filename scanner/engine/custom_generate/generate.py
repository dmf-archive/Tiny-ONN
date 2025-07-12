import torch
import torch.nn.functional as F
from bitsandbytes.nn import Linear4bit
from functools import partial
from typing import Any, List

def _capture_block_activations_hook(
    name: str, data_dict_for_token: dict, module: torch.nn.Module, inp: tuple, out: Any
):
    if name not in data_dict_for_token:
        data_dict_for_token[name] = {}

    x = inp[0]
    num_features = x.shape[-1]
    block_size = getattr(module, "blocksize", 64)
    num_blocks = (num_features + block_size - 1) // block_size

    token_slice = x[:, -1, :]
    padded_x = F.pad(token_slice, (0, num_blocks * block_size - num_features))
    reshaped_x = padded_x.view(x.shape[0], num_blocks, block_size)
    block_norms = torch.norm(reshaped_x.float(), p=2, dim=-1).squeeze(0).tolist()

    data_dict_for_token[name]["activation"] = block_norms


def generate(
    model,
    input_ids: torch.Tensor,
    generation_config=None,
    **kwargs,
) -> tuple[torch.Tensor, List[dict]]:
    
    tokenizer = kwargs.pop("tokenizer")
    per_token_data: List[dict] = []
    
    generation_config = generation_config or model.generation_config
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    max_new_tokens = generation_config.max_new_tokens
    
    generated_ids = []
    
    hook_modules = []
    for name, module in model.named_modules():
        if isinstance(module, Linear4bit):
            setattr(module, "param_name", name)
            hook_modules.append(module)

    current_ids = input_ids
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    position_ids = (attention_mask.long().cumsum(-1) - 1)
    position_ids.masked_fill_(attention_mask == 0, 1)
    cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
    past_key_values = None

    for _ in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(
            current_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
        )

        handles = []
        current_token_data_dict: dict[str, Any] = {}
        
        try:
            for module in hook_modules:
                handles.append(
                    module.register_forward_hook(
                        partial(
                            _capture_block_activations_hook,
                            module.param_name,
                            current_token_data_dict,
                        )
                    )
                )

            with torch.no_grad():
                outputs = model(**model_inputs, return_dict=True, use_cache=True)
            
            per_token_data.append(current_token_data_dict)

        finally:
            for handle in handles:
                handle.remove()

        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        
        # Update variables for the next iteration
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        position_ids = torch.tensor([[cache_position[-1] + 1]], device=model.device, dtype=torch.long)
        cache_position = torch.tensor([cache_position[-1] + 1], device=model.device)


        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[:, indices_to_remove] = -float("Inf")

        probs = F.softmax(next_token_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated_ids.append(next_token.item())
        current_ids = next_token

        if next_token.item() == tokenizer.eos_token_id:
            break
            
    for module in hook_modules:
        if hasattr(module, "param_name"):
            delattr(module, "param_name")

    final_ids = torch.cat([input_ids.squeeze(0), torch.tensor(generated_ids, device=model.device, dtype=torch.long)])
    return final_ids, per_token_data
