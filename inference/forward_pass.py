import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from inference.block_level_capture import get_and_clear_block_level_data


def run_forward_pass(model, tokenizer: AutoTokenizer, user_message: str, history: list):
    """
    Runs the forward pass for a given input and captures block-level data.
    Returns generated tokens and per-token block-level data.
    """
    messages = []
    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        if assistant is not None:
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device) # type: ignore
    
    input_ids = model_inputs.input_ids
    
    generated_ids: list[int] = []
    per_token_block_data = []
    past_key_values = None
    model.eval()

    with torch.no_grad():
        # First, process the prompt to get the initial KV cache
        prompt_outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = prompt_outputs.past_key_values
        next_token_logits = prompt_outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # Start the generation loop
        for _ in tqdm(range(256), desc="[Unified] Generating & Logging"):
            # Clear and get block-level data for the previous token before processing the current one
            # This ensures we capture data for the token that *just* finished its forward pass
            if len(generated_ids) > 0: # Only collect after the first token is generated
                per_token_block_data.append(get_and_clear_block_level_data())
            
            # Use KV cache for efficient single-token forward pass
            outputs = model(input_ids=next_token_id, past_key_values=past_key_values, use_cache=True)
            
            # Update for next iteration
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # Append generated token
            generated_ids.append(int(next_token_id.item()))
            
            if next_token_id.item() == tokenizer.eos_token_id: # type: ignore
                break
            torch.cuda.empty_cache() # Clear CUDA cache after each token generation
    
    # Capture data for the very last generated token
    if len(generated_ids) > 0:
        per_token_block_data.append(get_and_clear_block_level_data())

    return generated_ids, per_token_block_data, input_ids
