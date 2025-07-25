import json
from pathlib import Path


def calculate_qwen3_params(config):
    """Calculates parameters for a standard Qwen3 dense model."""
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    vocab_size = config["vocab_size"]
    num_hidden_layers = config["num_hidden_layers"]
    num_kv_heads = config["num_key_value_heads"]
    num_attn_heads = config["num_attention_heads"]
    head_dim = hidden_size // num_attn_heads

    embedding_params = vocab_size * hidden_size

    q_proj_params = hidden_size * (num_attn_heads * head_dim)
    k_proj_params = hidden_size * (num_kv_heads * head_dim)
    v_proj_params = hidden_size * (num_kv_heads * head_dim)
    o_proj_params = (num_attn_heads * head_dim) * hidden_size
    attn_params_per_layer = q_proj_params + k_proj_params + v_proj_params + o_proj_params

    mlp_params_per_layer = (hidden_size * intermediate_size) * 2 + (intermediate_size * hidden_size)

    norm_params = num_hidden_layers * (hidden_size * 2) + hidden_size

    total_attn_params = num_hidden_layers * attn_params_per_layer
    total_mlp_params = num_hidden_layers * mlp_params_per_layer
    total_params = embedding_params + total_attn_params + total_mlp_params + norm_params

    return {
        "total": total_params,
        "embedding": embedding_params,
        "attention": total_attn_params,
        "mlp": total_mlp_params,
    }


def calculate_tiny_onn_params(config):
    """Calculates parameters for our Tiny-ONN SMoE model."""
    hidden_size = config["hidden_size"]
    vocab_size = config["vocab_size"]
    num_hidden_layers = config["num_hidden_layers"]
    num_kv_heads = config["num_key_value_heads"]
    num_attn_heads = config["num_attention_heads"]
    head_dim = hidden_size // num_attn_heads
    num_experts = config["num_experts_per_layer"]
    moe_intermediate_size = config["moe_intermediate_size"]

    embedding_params = vocab_size * hidden_size

    q_proj_params = hidden_size * (num_attn_heads * head_dim)
    k_proj_params = hidden_size * (num_kv_heads * head_dim)
    v_proj_params = hidden_size * (num_kv_heads * head_dim)
    o_proj_params = (num_attn_heads * head_dim) * hidden_size
    attn_params_per_layer = q_proj_params + k_proj_params + v_proj_params + o_proj_params
    total_attn_params = num_hidden_layers * attn_params_per_layer

    backbone_params = embedding_params + total_attn_params

    original_intermediate_size = moe_intermediate_size * num_experts
    mlp_params_per_layer = (hidden_size * original_intermediate_size) * 2 + (original_intermediate_size * hidden_size)
    total_moe_params = num_hidden_layers * mlp_params_per_layer

    gating_params = num_hidden_layers * (hidden_size * num_experts)

    total_moe_and_gating_params = total_moe_params + gating_params

    norm_params = num_hidden_layers * (hidden_size * 2) + hidden_size

    total_params = backbone_params + total_moe_and_gating_params + norm_params

    return {
        "total": total_params,
        "backbone": backbone_params,
        "experts_and_gating": total_moe_and_gating_params,
        "single_expert": (mlp_params_per_layer / num_experts),
    }

def main():
    qwen3_config_path = Path("weights/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/config.json")
    tiny_onn_config_path = Path("weights/Tiny-ONN-0.6B-Hyper-SMoE/config.json")

    with open(qwen3_config_path) as f:
        qwen3_config = json.load(f)

    with open(tiny_onn_config_path) as f:
        tiny_onn_config = json.load(f)

    qwen3_params = calculate_qwen3_params(qwen3_config)
    tiny_onn_params = calculate_tiny_onn_params(tiny_onn_config)

    print("--- Qwen3-0.6B ---")
    print(f"Total Parameters: {qwen3_params['total'] / 1e9:.3f} B")
    print(f"  - Embedding: {qwen3_params['embedding'] / 1e6:.2f} M")
    print(f"  - Attention: {qwen3_params['attention'] / 1e6:.2f} M")
    print(f"  - MLP: {qwen3_params['mlp'] / 1e6:.2f} M")


    print("\n--- Tiny-ONN-0.6B-Hyper-SMoE ---")
    print(f"Total Parameters: {tiny_onn_params['total'] / 1e9:.3f} B")
    print(f"  - Shared Backbone: {tiny_onn_params['backbone'] / 1e6:.2f} M")
    print(f"  - Experts + Gating: {tiny_onn_params['experts_and_gating'] / 1e6:.2f} M")
    print(f"  - Single Expert: {tiny_onn_params['single_expert'] / 1e6:.3f} M")

if __name__ == "__main__":
    main()
