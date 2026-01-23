import torch
import torch.nn as nn
from src.models.dynsiha.flat.configuration_flat_dynsiha import FlatDynSIHAConfig
from src.models.dynsiha.flat.modeling_flat_dynsiha import FlatDynSIHAForCausalLM
from src.models.dynsiha.recursive.configuration_recursive_dynsiha import RecursiveDynSIHAConfig
from src.models.dynsiha.recursive.modeling_recursive_dynsiha import RecursiveDynSIHAForCausalLM
from transformers.cache_utils import DynamicCache

def test_flat_dynsiha_kv_cache():
    print("Testing FlatDynSIHA KV Cache consistency...")
    config = FlatDynSIHAConfig(
        vocab_size=32,
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=4,
        num_experts=4,
        top_k=2,
    )
    model = FlatDynSIHAForCausalLM(config)
    model.eval()

    input_ids = torch.randint(0, 32, (1, 10))
    
    # 1. Full forward pass
    with torch.no_grad():
        full_outputs = model(input_ids, use_cache=True)
        full_logits = full_outputs.logits
        full_cache = full_outputs.past_key_values

    # 2. Incremental forward pass
    past_key_values = DynamicCache()
    incremental_logits = []
    
    with torch.no_grad():
        for i in range(10):
            curr_input_ids = input_ids[:, i:i+1]
            outputs = model(curr_input_ids, past_key_values=past_key_values, use_cache=True)
            incremental_logits.append(outputs.logits)
            past_key_values = outputs.past_key_values

    incremental_logits = torch.cat(incremental_logits, dim=1)
    
    # Compare
    diff = torch.abs(full_logits - incremental_logits).max().item()
    print(f"Max logit difference: {diff:.2e}")
    
    if diff < 1e-5:
        print("✅ FlatDynSIHA KV Cache test passed!")
    else:
        print("❌ FlatDynSIHA KV Cache test failed!")

def test_recursive_dynsiha_forward():
    print("\nTesting RecursiveDynSIHA Forward and ACT structure...")
    config = RecursiveDynSIHAConfig(
        vocab_size=32,
        hidden_size=64,
        max_refinement_steps=3,
        num_heads=4,
        num_experts=4,
        top_k=2,
    )
    model = RecursiveDynSIHAForCausalLM(config)
    model.eval()

    input_ids = torch.randint(0, 32, (1, 5))
    
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        print(f"Logits shape: {outputs.logits.shape}")
        if outputs.past_key_values is not None:
            # In RecursiveDynSIHA, we treat steps as layers in cache
            seq_len = outputs.past_key_values.get_seq_length()
            print(f"Cache seq_len: {seq_len}")
            
    print("✅ RecursiveDynSIHA Forward test completed!")

if __name__ == "__main__":
    test_flat_dynsiha_kv_cache()
    test_recursive_dynsiha_forward()
