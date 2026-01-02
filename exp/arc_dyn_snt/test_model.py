import torch
from .model import ArcDynSNT
from .config import ModelConfig

def test_model_initialization():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use a small configuration for testing
    config = ModelConfig(
        vocab_size=16,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        ffn_scale=2,
        dropout=0.1,
    )

    try:
        model = ArcDynSNT(config, device)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Model initialization failed: {e}")
        return

    # Test forward pass
    try:
        input_ids = torch.randint(0, config.vocab_size, (1, 10)).to(device)
        # When return_dict=False, the model returns a tuple
        outputs = model(input_ids, return_dict=False)
        logits = outputs[0]
        print("Forward pass successful.")
        print("Output logits shape:", logits.shape)
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    test_model_initialization()