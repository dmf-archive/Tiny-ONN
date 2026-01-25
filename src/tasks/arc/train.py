import torch

from src.models.dynsiha.flat.configuration_flat_dynsiha import FlatDynSIHAConfig
from src.models.dynsiha.flat.modeling_flat_dynsiha import FlatDynSIHAForCausalLM
from src.tasks.arc.config import ModelConfig, TrainConfig
from src.tasks.arc.trainer import ARCTrainer


def main():
    # 1. Setup Config
    model_cfg = ModelConfig(
        vocab_size=17,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
    )
    train_cfg = TrainConfig(
        model=model_cfg,
        lr=1e-4,
        batch_size=2,
        num_epochs=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        data_dir="data/ARC-AGI-2/data"
    )

    # 2. Initialize Model
    fds_config = FlatDynSIHAConfig(
        vocab_size=train_cfg.model.vocab_size,
        hidden_size=train_cfg.model.hidden_size,
        num_hidden_layers=train_cfg.model.num_layers,
        num_heads=train_cfg.model.num_heads,
        num_experts=4,
        top_k=2,
        max_position_embeddings=train_cfg.model.max_seq_len,
    )
    model = FlatDynSIHAForCausalLM(fds_config)

    # 3. Initialize Trainer
    trainer = ARCTrainer(model, train_cfg)

    # 4. Smoke Test: Run one epoch
    trainer.train()

if __name__ == "__main__":
    main()
