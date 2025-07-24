
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from tiny_onn.modular import TinyOnnForCausalLM
from training.config import TrainConfig
from training.engine import TrainerEngine


def test_meta_learning_grad_flow(tiny_test_model_and_tokenizer):
    model, _ = tiny_test_model_and_tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)
    model.train()

    # Create dummy dataloader, optimizers, and config
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    expert_params = [p for n, p in model.named_parameters() if "experts" in n]
    router_params = [p for n, p in model.named_parameters() if "gate" in n]
    optimizer_experts = AdamW(expert_params, lr=1e-3)
    optimizer_router = AdamW(router_params, lr=1e-3)

    config_dict = {
        "model": {"base_model_name": "dummy", "teacher_model_name": None, "model_path": "dummy"},
        "data": {
            "train_path": "dummy", "eval_path": "dummy", "dataset_name": "dummy",
            "dataset_subset": "dummy", "validation_split_percentage": 10, "max_seq_length": 32,
        },
        "training": {
            "output_dir": "tests/test_output", "num_train_epochs": 1,
            "per_device_train_batch_size": 2, "per_device_eval_batch_size": 2,
            "dataloader_num_workers": 0, "expert_learning_rate": 1e-4, "gate_learning_rate": 1e-4,
            "weight_decay": 0.01, "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0, "lr_scheduler_warmup_steps": 10, "pi_alpha": 0.5, "pi_gamma": 0.1,
        },
        "logging": {
            "rolling_checkpoint_count": 1, "log_interval": 1, "eval_interval": 1, "checkpoint_interval": 1,
        },
        "system": {"device": "cpu", "seed": 42}
    }
    config = TrainConfig.from_dict(config_dict)
    
    trainer = TrainerEngine(
        config=config, model=model, optimizer_experts=optimizer_experts,
        optimizer_router=optimizer_router, train_dataloader=dataloader,
        eval_dataloader=dataloader, device=device
    )

    batch = {"input_ids": input_ids, "labels": labels}
    trainer._hyper_step(batch)
    
    gate_grad_found = False
    for layer in model.model.layers:
        gate = layer.mlp.gate
        if hasattr(gate, 'weight') and gate.weight.grad is not None:
            print(f"Gate gradient for layer {layer.layer_idx}: {gate.weight.grad.norm()}")
            if torch.any(gate.weight.grad != 0):
                gate_grad_found = True
                break
    
    assert gate_grad_found, "Gate should have received non-zero gradients!"
    print("\n✅ PoC successful. Router (gate) received gradients from router_loss.")
