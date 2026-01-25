import torch

from .models.dynsiha.flat.configuration_flat_dynsiha import FlatDynSIHAConfig
from .models.dynsiha.flat.modeling_flat_dynsiha import FlatDynSIHAForCausalLM
from .tasks.arc.config import ModelConfig, TrainConfig
from .tasks.arc.trainer import ARCTrainer


def main():
    # 1. Setup Config
    model_cfg = ModelConfig(
        vocab_size=17,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        max_seq_len=2048
    )

    train_cfg = TrainConfig(
        model=model_cfg,
        lr=1e-4,
        batch_size=4,
        num_epochs=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        data_dir="data/ARC-AGI-2/data"
    )

    # 2. Initialize Model
    flat_config = FlatDynSIHAConfig(
        vocab_size=model_cfg.vocab_size,
        hidden_size=model_cfg.hidden_size,
        num_hidden_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        num_experts=16,
        top_k=4,
        max_position_embeddings=model_cfg.max_seq_len
    )
    model = FlatDynSIHAForCausalLM(flat_config)

    # 3. Initialize Trainer and Start Training
    trainer = ARCTrainer(model, train_cfg)
    trainer.train()

if __name__ == "__main__":
    main()
