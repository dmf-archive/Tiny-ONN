import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from bitsandbytes.optim import AdamW8bit
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

from tiny_onn.modular import CausalLMOutputWithAux, TinyOnnForCausalLM
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

            optimizer_expert.zero_grad()
            optimizer_gate.zero_grad()

            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.pop("labels")
            surprise_cache: list[dict[str, Any]] = []
            hidden_states_cache: list[torch.Tensor] = []
            inputs["surprise_cache"] = surprise_cache
            inputs["hidden_states_cache"] = hidden_states_cache

            fwd_pass_start_time = time.time()
            outputs: CausalLMOutputWithAux = model(**inputs)
            assert outputs.logits is not None
            main_loss = F.cross_entropy(outputs.logits.view(-1, model.config.vocab_size), labels.view(-1))
            main_loss.backward()
            optimizer_expert.step()
            fwd_pass_time = time.time() - fwd_pass_start_time

            gating_loss_start_time = time.time()

            num_layers = model.config.num_hidden_layers
            if not outputs.aux_outputs or not surprise_cache:
                continue

            num_tokens = hidden_states_cache[0].shape[1]
            num_experts = model.config.num_experts_per_layer

            surprise_matrix_l_t_e = torch.full((num_layers, num_tokens, num_experts), float("inf"), device=device, dtype=torch.bfloat16)

            layer_indices = torch.tensor([d['layer_idx'] for d in surprise_cache], device=device)
            token_indices = torch.tensor([d['token_idx'] for d in surprise_cache], device=device)
            expert_indices = torch.tensor([d['expert_idx'] for d in surprise_cache], device=device)
            norms = torch.tensor([d['norm'] for d in surprise_cache], device=device, dtype=torch.bfloat16)
            surprise_matrix_l_t_e[layer_indices, token_indices, expert_indices] = norms
            surprise_matrix = surprise_matrix_l_t_e.permute(1, 0, 2)

            surprise_for_loss = surprise_matrix.clone().detach()

            hidden_dim = model.config.hidden_size
            stacked_hidden_states = torch.stack(hidden_states_cache, dim=0)
            recomputed_scores_list = []
            for i in range(num_layers):
                h_states = stacked_hidden_states[i]
                flat_h_states = h_states.view(-1, hidden_dim)
                _, _, activated_scores = model.model.layers[i].mlp.gate(flat_h_states)
                recomputed_scores_list.append(activated_scores)
            all_activated_scores = torch.stack(recomputed_scores_list, dim=1)

            routing_mask = all_activated_scores > 0
            masked_scores = torch.where(routing_mask, all_activated_scores, torch.tensor(float('-inf'), device=all_activated_scores.device, dtype=all_activated_scores.dtype))
            recomputed_routing_weights = F.softmax(masked_scores, dim=-1, dtype=torch.float)

            loss_smk = compute_smk_weighted_loss(surprise_for_loss, all_activated_scores)
            loss_balance = compute_load_balancing_loss(recomputed_routing_weights, all_activated_scores)

            with torch.no_grad():
                main_loss_item = main_loss.item()
                valid_surprise = surprise_matrix[torch.isfinite(surprise_matrix)]
                surprise_val = valid_surprise.mean().item() if valid_surprise.numel() > 0 else 0.0
                assert outputs.logits is not None
                tau = max(torch.distributions.Categorical(logits=outputs.logits.float()).entropy().mean().item(), 1e-9)
                cost = (1 - config.observer.pi_gamma) * (main_loss_item / tau) + config.observer.pi_gamma * surprise_val
                pi_score = torch.exp(torch.tensor(-config.observer.pi_alpha * cost)).item()

            lambda_balance = 1.0 - torch.sigmoid(torch.tensor((pi_score - config.train.pi_threshold) * config.train.pi_sensitivity)).item()
            gating_loss_total = (1.0 - lambda_balance) * loss_smk + lambda_balance * loss_balance
            gating_loss_total.backward()
            optimizer_gate.step()
            gating_loss_time = time.time() - gating_loss_start_time

            maintenance_start_time = time.time()
            lr_scheduler_expert.step()
            lr_scheduler_gate.step()
            expert_manager.check_and_regenerate(global_step)
            maintenance_time = time.time() - maintenance_start_time

            obs_start_time = time.time()
            with torch.no_grad():
                main_acc = (outputs.logits.argmax(-1) == labels).float().mean().item()

                optimal_expert_indices = torch.argmin(surprise_matrix, dim=-1)
                topk_indices = torch.topk(all_activated_scores, k=1, dim=-1).indices.squeeze(-1)
                gating_acc = (topk_indices == optimal_expert_indices).float().mean().item()

                num_activated_experts = (all_activated_scores > 0).sum(dim=-1).float()
                avg_k = num_activated_experts.mean().item()
                global_avg_k = num_activated_experts.sum(dim=1).mean().item()

                log_probs_gating = F.log_softmax(masked_scores, dim=-1)
                safe_surprise_matrix = torch.where(torch.isinf(surprise_matrix), torch.full_like(surprise_matrix, 1e9), surprise_matrix)
                log_probs_surprise = F.log_softmax(-safe_surprise_matrix, dim=-1)
                gating_kld = F.kl_div(log_probs_gating, log_probs_surprise, reduction='batchmean', log_target=True)

                metrics = {
                    "main_loss": main_loss_item,
                    "smk_loss": loss_smk.item(),
                    "balance_loss": loss_balance.item(),
                    "gating_loss": gating_loss_total.item(),
                    "main_acc": main_acc,
                    "gating_acc": gating_acc,
                    "gating_kld": gating_kld.item(),
                    "avg_k": avg_k,
                    "global_avg_k": global_avg_k,
                    "surprise": surprise_val,
                    "tau": float(tau),
                    "pi_score": pi_score,
                    "lambda": lambda_balance,
                }

                expert_data = {
                    "activated_scores": all_activated_scores.clone().detach(),
                    "surprise_matrix": surprise_matrix.clone().detach(),
                }
                observer.log_metrics_and_expert_data(metrics, expert_data, global_step)

                timings = {
                    "forward_pass_and_expert_step": fwd_pass_time,
                    "gating_loss_and_step": gating_loss_time,
                    "maintenance": maintenance_time,
                    "observation": time.time() - obs_start_time,
                    "total_step_time": time.time() - step_start_time,
                }
                observer.log_timing(timings, global_step)

            observer.plot_all_dashboards(global_step, model.config)
            global_step += 1

    observer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args.config)
