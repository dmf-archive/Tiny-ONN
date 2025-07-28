import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from bitsandbytes.optim import AdamW8bit
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

from tiny_onn.modular import TinyOnnForCausalLM, CausalLMOutputWithAux
from training.config import FullConfig, load_config
from training.data import get_dataloaders
from training.expert_manager import ExpertManager
from training.losses import compute_load_balancing_loss, compute_smk_weighted_loss
from training.observer import UnifiedObserver


def main(config_path: str):
    config: FullConfig = load_config(Path(config_path))
    device = torch.device("cuda")
    output_dir = Path(config.observer.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = TinyOnnForCausalLM.from_pretrained(
        config.model.model_path, trust_remote_code=True
    ).to(device, dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_path,
        model_max_length=config.data.max_seq_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataloader, _ = get_dataloaders(
        data_config=config.data,
        tokenizer=tokenizer,
        batch_size=config.train.per_device_train_batch_size,
        num_workers=0,
    )

    expert_params = [p for n, p in model.named_parameters() if "experts" in n and p.requires_grad]
    gate_params = [p for n, p in model.named_parameters() if "gate" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "gate" not in n and "experts" not in n and p.requires_grad]

    optimizer_expert = AdamW8bit(
        [
            {"params": expert_params, "lr": config.train.learning_rate * 0.1},
            {"params": other_params, "lr": config.train.learning_rate},
        ],
        betas=(config.train.adam_beta1, config.train.adam_beta2), eps=config.train.adam_epsilon, weight_decay=config.train.weight_decay
    )
    optimizer_gate = AdamW8bit(
        [{"params": gate_params, "lr": config.train.learning_rate}],
        betas=(config.train.adam_beta1, config.train.adam_beta2), eps=config.train.adam_epsilon, weight_decay=config.train.weight_decay
    )

    num_training_steps = len(train_dataloader) * config.train.num_train_epochs
    lr_scheduler_expert = get_scheduler("linear", optimizer=optimizer_expert, num_warmup_steps=config.train.lr_scheduler_warmup_steps, num_training_steps=num_training_steps)
    lr_scheduler_gate = get_scheduler("linear", optimizer=optimizer_gate, num_warmup_steps=config.train.lr_scheduler_warmup_steps, num_training_steps=num_training_steps)

    observer = UnifiedObserver(output_dir, device)
    expert_manager = ExpertManager(model, config.train.expert_regeneration_interval)
    global_step = 0

    model.train()
    for epoch in range(config.train.num_train_epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            step_start_time = time.time()
            
            # === Phase 1: Forward Pass & Main Task Learning ===
            optimizer_expert.zero_grad()
            optimizer_gate.zero_grad()

            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.pop("labels")
            surprise_cache: List[Dict[str, Any]] = []
            inputs["surprise_cache"] = surprise_cache
            
            fwd_pass_start_time = time.time()
            outputs: CausalLMOutputWithAux = model(**inputs)
            main_loss = F.cross_entropy(outputs.logits.view(-1, model.config.vocab_size), labels.view(-1))
            main_loss.backward()
            optimizer_expert.step()
            fwd_pass_time = time.time() - fwd_pass_start_time

            # === Phase 2: Gating Meta-Learning ===
            gating_loss_start_time = time.time()
            
            # --- Surprise Matrix Construction ---
            num_layers = model.config.num_hidden_layers
            num_tokens, num_experts = next(iter(outputs.aux_outputs)).get("raw_logits").shape
            surprise_matrix = torch.full((num_layers, num_tokens, num_experts), float("inf"), device=device, dtype=torch.bfloat16)
            for entry in surprise_cache:
                surprise_matrix[entry["layer_idx"], entry["token_idx"], entry["expert_idx"]] = entry["norm"]
            flat_surprise_matrix = surprise_matrix.view(-1, num_experts)
            
            # --- Gating Loss Calculation ---
            all_activated_scores = torch.cat([d["activated_scores"] for d in outputs.aux_outputs], dim=0)
            all_routing_weights = torch.cat([d["routing_weights"] for d in outputs.aux_outputs], dim=0)
            all_raw_logits = torch.cat([d["raw_logits"] for d in outputs.aux_outputs], dim=0)

            loss_smk = compute_smk_weighted_loss(flat_surprise_matrix, all_activated_scores)
            loss_balance = compute_load_balancing_loss(all_routing_weights, all_activated_scores)

            with torch.no_grad():
                main_loss_item = main_loss.item()
                valid_surprise = flat_surprise_matrix[torch.isfinite(flat_surprise_matrix)]
                surprise_val = valid_surprise.mean().item() if valid_surprise.numel() > 0 else 0.0
                tau = max(torch.distributions.Categorical(logits=outputs.logits.float()).entropy().mean().item(), 1e-9)
                cost = (1 - config.observer.pi_gamma) * (main_loss_item / tau) + config.observer.pi_gamma * surprise_val
                pi_score = torch.exp(torch.tensor(-config.observer.pi_alpha * cost)).item()

            lambda_balance = 1.0 - torch.sigmoid(torch.tensor((pi_score - config.train.pi_threshold) * config.train.pi_sensitivity)).item()
            gating_loss_total = (1.0 - lambda_balance) * loss_smk + lambda_balance * loss_balance
            
            gating_loss_total.backward()
            optimizer_gate.step()
            gating_loss_time = time.time() - gating_loss_start_time

            # === Phase 3: System Maintenance & Observation ===
            maintenance_start_time = time.time()
            lr_scheduler_expert.step()
            lr_scheduler_gate.step()
            expert_manager.check_and_regenerate(global_step)
            maintenance_time = time.time() - maintenance_start_time

            obs_start_time = time.time()
            with torch.no_grad():
                main_acc = (outputs.logits.argmax(-1) == labels).float().mean().item()
                metrics = {
                    "main_loss": main_loss_item,
                    "smk_loss": loss_smk.item(),
                    "balance_loss": loss_balance.item(),
                    "gating_loss": gating_loss_total.item(),
                    "main_acc": main_acc,
                    "surprise": surprise_val,
                    "tau": float(tau),
                    "pi_score": pi_score,
                    "lambda": lambda_balance,
                }
                
                selected_experts_mask = (all_activated_scores > 0)
                selected_experts = torch.where(selected_experts_mask)[1] + (torch.where(selected_experts_mask)[0] // num_tokens) * num_experts
                
                optimal_experts_indices = torch.argmin(flat_surprise_matrix, dim=-1)
                optimal_experts_mask = torch.isfinite(flat_surprise_matrix.min(dim=-1).values)
                optimal_experts = optimal_experts_indices[optimal_experts_mask] + torch.where(optimal_experts_mask)[0] * num_experts
                
                expert_data = {"selected_experts": selected_experts, "optimal_experts": optimal_experts}
                observer.log_metrics_and_expert_data(metrics, expert_data, global_step)

                timings = {
                    "forward_pass_and_expert_step": fwd_pass_time,
                    "gating_loss_and_step": gating_loss_time,
                    "maintenance": maintenance_time,
                    "observation": time.time() - obs_start_time,
                    "total_step_time": time.time() - step_start_time,
                }
                observer.log_timing(timings, global_step)

            observer.plot_dashboards(global_step, model.config)
            global_step += 1

    observer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args.config)
