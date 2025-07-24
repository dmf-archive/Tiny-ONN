import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from tiny_onn.modular import TinyOnnForCausalLM
from training.config import load_config


def main():
    config_path = "configs/meta_train_v1.yaml"
    config = load_config(Path(config_path))
    
    model = TinyOnnForCausalLM.from_pretrained(
        config.model.model_path, trust_remote_code=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dummy_input_ids = torch.randint(0, model.config.vocab_size, (2, 16), device=device)
    dummy_labels = torch.randint(0, model.config.vocab_size, (2, 16), device=device)
    
    outputs = model(input_ids=dummy_input_ids, labels=dummy_labels)
    loss = outputs.loss
    
    expert_params = [
        p for n, p in model.named_parameters() if "mlp.experts.0" in n and p.requires_grad
    ]
    
    grads = torch.autograd.grad(loss, expert_params, allow_unused=True)
    
    print("--- Gradients for Expert 0 ---")
    for name, grad in zip([n for n, p in model.named_parameters() if "mlp.experts.0" in n], grads):
        if grad is not None:
            print(f"Param: {name}, Grad Norm: {torch.linalg.norm(grad).item()}")
        else:
            print(f"Param: {name}, Grad: None")

if __name__ == "__main__":
    main()