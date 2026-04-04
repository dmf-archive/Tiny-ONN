import argparse

import torch

from src.models.dynsiha.flat.configuration_flat_dynsiha import FlatDynSIHAConfig
from src.models.dynsiha.flat.modeling_flat_dynsiha import FlatDynSIHAForCausalLM
from src.models.dynsiha.recursive.configuration_recursive_dynsiha import (
    RecursiveDynSIHAConfig,
)
from src.models.dynsiha.recursive.modeling_recursive_dynsiha import (
    RecursiveDynSIHAForCausalLM,
)
from src.tasks.arc.config import TrainConfig
from src.tasks.arc.trainer import ARCTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny-ONN ARC Training Entry Point")
    parser.add_argument("--model_type", type=str, default="recursive", choices=["flat", "recursive"], help="Model architecture type")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per sequence")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--data_dir", type=str, default="data/ARC-AGI-2/data", help="Path to ARC data")
    parser.add_argument("--use_sia", action="store_true", help="Enable SIA (Detached Gradient) mode for recursive models")
    parser.add_argument("--use_cache_in_train", action="store_true", help="Enable KV cache during training for recursive models")
    parser.add_argument("--use_act_inference", action="store_true", help="Enable ACT early-exit during inference")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model_type == "recursive":
        config = RecursiveDynSIHAConfig(
            vocab_size=17,
            hidden_size=256,
            max_refinement_steps=6,
            max_inference_steps=8,
            num_heads=8,
            num_experts=32,
            top_k=4,
            use_sia=args.use_sia,
            use_cache_in_train=args.use_cache_in_train,
            use_act_inference=args.use_act_inference,
            max_position_embeddings=4096
        )
        model = RecursiveDynSIHAForCausalLM(config)
    else:
        config = FlatDynSIHAConfig(
            vocab_size=17,
            hidden_size=512,
            num_hidden_layers=6,
            num_heads=8,
            num_experts=16,
            top_k=4,
            max_position_embeddings=2048
        )
        model = FlatDynSIHAForCausalLM(config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing = True

    train_config = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        num_epochs=args.num_epochs,
        device=args.device,
        data_dir=args.data_dir,
        w_meta=0.1,
        cost_alpha=0.5,
        eval_steps=200,
        save_steps=1000,
        max_checkpoints=3,
        memory_threshold_gb=8.0,
        compute_rmi_itjd=True
    )

    trainer = ARCTrainer(model, train_config)
    trainer.train()


if __name__ == "__main__":
    main()
