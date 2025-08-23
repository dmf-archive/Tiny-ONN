import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import Config
from .data import get_arc_dataloaders
from .model import ExpertID, TinyOnnForArcReconstruction
from .observer import Observer


def track_and_remake_experts(model: TinyOnnForArcReconstruction, activation_tracker: dict, console: Console):
    console.print("\n[bold cyan]Checking for inactive experts to remake...[/bold cyan]")
    remake_count = 0
    
    # --- OFL SMHA Experts ---
    if config.use_object_finder:
        ofl_layer = model.model.object_finder_layer
        ofl_tracker = activation_tracker.get(("smha", -1), torch.zeros(ofl_layer.max_experts, device=config.DEVICE))
        inactive_ofl_experts = torch.where(ofl_tracker == 0)[0]
        for expert_idx in inactive_ofl_experts:
            expert_idx = expert_idx.item()
            with torch.no_grad():
                nn.init.xavier_uniform_(ofl_layer.q_proj[expert_idx])
                nn.init.xavier_uniform_(ofl_layer.k_proj[expert_idx])
                nn.init.xavier_uniform_(ofl_layer.v_proj[expert_idx])
                nn.init.xavier_uniform_(ofl_layer.o_proj[expert_idx])
                ofl_layer.gating_network.sim_matrix.data[:, expert_idx].normal_()
                ofl_layer.gating_network.gates.data[expert_idx] = 0.0
            remake_count += 1
            console.print(f"  - Remade OFL (SMHA) expert E{expert_idx}")

    # --- MoE Experts in Blocks ---
    for layer_idx, layer in enumerate(model.model.layers):
        moe_layer = layer.moe_layer
        moe_tracker = activation_tracker.get(("moe", layer_idx), torch.zeros(moe_layer.max_experts, device=config.DEVICE))
        inactive_moe_experts = torch.where(moe_tracker == 0)[0]
        for expert_idx in inactive_moe_experts:
            expert_idx = expert_idx.item()
            with torch.no_grad():
                nn.init.kaiming_uniform_(moe_layer.w1[expert_idx], a=math.sqrt(5))
                nn.init.kaiming_uniform_(moe_layer.w2[expert_idx], a=math.sqrt(5))
                moe_layer.gating_network.sim_matrix.data[:, expert_idx].normal_()
                moe_layer.gating_network.gates.data[expert_idx] = 0.0
            remake_count += 1
            console.print(f"  - Remade MoE expert L{layer_idx}/E{expert_idx}")

    if remake_count == 0:
        console.print("  - All experts are active.")
    else:
        console.print(f"[bold green]Total experts remade: {remake_count}[/bold green]")

def get_latest_checkpoint(path: Path) -> Path | None:
    checkpoints = list(path.glob("*.pt"))
    return max(checkpoints, key=os.path.getctime) if checkpoints else None

def calculate_gating_loss_and_metrics(
    config: Config, main_loss: torch.Tensor, aux_outputs: dict[ExpertID, dict]
) -> tuple[torch.Tensor, dict[str, float]]:
    metrics = defaultdict(list)
    total_gating_loss = torch.tensor(0.0, device=config.DEVICE)

    for expert_id, cache in aux_outputs.items():
        type, layer_idx, _ = expert_id
        B, T = cache["B"], cache["T"]
        
        # --- Metrics Calculation (for all layers) ---
        gate_cache = cache["gate_cache"]
        logits = gate_cache["logits"].view(B * T, -1)
        mask = gate_cache["activation_mask"].view(B * T, -1)
        metrics[f"{type}_avg_k"].append(mask.float().sum(1).mean().item())

        # --- Surprise and Gating Loss Calculation (only for MoE layers) ---
        if type == "moe":
            # Surprise is grad of main_loss w.r.t per-expert outputs
            grads, = torch.autograd.grad(outputs=main_loss, inputs=cache["final_output"], retain_graph=True)
            surprise = torch.linalg.norm(grads.view(B * T, -1), dim=1)
            
            # Gate accuracy metric
            metrics[f"{type}_gate_acc"].append((logits.argmax(-1) == surprise.argmin(-1)).float().mean().item())
            
            # Gating loss
            target_p = F.softmax(-surprise.unsqueeze(-1).expand_as(logits), dim=-1)
            log_pred_p = F.log_softmax(logits, dim=-1)
            g_loss = (-target_p * log_pred_p).sum(dim=-1).mean()
            total_gating_loss += g_loss

    processed_metrics = {k: sum(v) / len(v) for k, v in metrics.items() if v}
    return total_gating_loss, processed_metrics

def calculate_accuracy_metrics(generated_ids: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor) -> dict[str, float]:
    mask = labels != -100
    if not mask.any(): return {"tok_acc": 0.0, "grid_acc": 0.0}
    tok_acc = (logits.argmax(-1)[mask] == labels[mask]).float().mean().item()
    grid_acc = 0.0
    for i in range(labels.shape[0]):
        input_len = (labels[i] == -100).sum()
        gen_output = generated_ids[i, input_len:]
        label_output = labels[i, input_len:]
        len_match = min(len(gen_output), len(label_output))
        if (gen_output[:len_match] == label_output[:len_match]).all():
            grid_acc += 1
    return {"tok_acc": tok_acc, "grid_acc": grid_acc / labels.shape[0]}

def run_evaluation(model: TinyOnnForArcReconstruction, eval_loader: DataLoader, observer: Observer, config: Config):
    model.eval()
    total_tok_acc, total_grid_acc, count = 0, 0, 0
    with torch.no_grad():
        for i, (input_ids, labels, attention_mask) in enumerate(eval_loader):
            if i >= config.EVAL_BATCHES: break
            input_len = (labels[0] == -100).sum()
            max_new = labels.shape[1] - input_len
            generated_ids = model.generate(input_ids, attention_mask, max_new_tokens=max_new)
            logits, _ = model(generated_ids, attention_mask=attention_mask)
            metrics = calculate_accuracy_metrics(generated_ids, labels, logits)
            total_tok_acc += metrics["tok_acc"]
            total_grid_acc += metrics["grid_acc"]
            count += 1
            if i == 0: # Visualize first batch
                observer.visualize_batch(input_ids, labels, generated_ids)
    avg_tok_acc = total_tok_acc / count if count > 0 else 0
    avg_grid_acc = total_grid_acc / count if count > 0 else 0
    observer.log_eval_results({"tok_acc": avg_tok_acc, "grid_acc": avg_grid_acc})
    model.train()

def main():
    config = Config()
    device = torch.device(config.DEVICE)
    console = Console()
    observer = Observer(console, config)
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)
    train_loader, eval_loader = get_arc_dataloaders(config)
    model = TinyOnnForArcReconstruction(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    start_epoch, global_step = 0, 0
    if ckpt_path := get_latest_checkpoint(checkpoint_dir):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        console.print(f"Resumed from step {global_step}")

    last_log_time = time.time()
    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        expert_activation_tracker = defaultdict(lambda: torch.zeros(config.max_moe_experts, device=device))

        for batch_idx, (input_ids_batch, labels_batch, attention_mask_batch) in enumerate(train_loader):
            batch_losses = defaultdict(list)
            batch_metrics = defaultdict(list)
            
            for i in range(input_ids_batch.shape[0]):
                optimizer.zero_grad()
                input_ids, labels, attention_mask = input_ids_batch[i:i+1], labels_batch[i:i+1], attention_mask_batch[i:i+1]
                
                if config.TRAINING_MODE == 0:
                    logits, aux_outputs = model(input_ids, attention_mask=attention_mask)
                    generated_ids_for_acc = logits.argmax(-1)
                else:
                    input_len = (labels[0] == -100).sum()
                    max_new = labels.shape[1] - input_len
                    generated_ids = model.generate(input_ids[:, :input_len], attention_mask[:, :input_len], max_new_tokens=max_new)
                    logits, aux_outputs = model(generated_ids.detach(), attention_mask=F.pad(attention_mask, (0, generated_ids.shape[1] - attention_mask.shape[1]), "constant", 1))
                    generated_ids_for_acc = generated_ids

                main_loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1), ignore_index=-100)
                if torch.isnan(main_loss) or not aux_outputs: continue

                gating_loss, gate_metrics = calculate_gating_loss_and_metrics(config, main_loss, aux_outputs)
                total_loss = main_loss + gating_loss
                
                if not torch.isnan(total_loss):
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
                    optimizer.step()
                
                for expert_id, cache in aux_outputs.items():
                    key = (expert_id[0], expert_id[1])
                    expert_activation_tracker[key] += cache["gate_cache"]["activation_mask"].sum(dim=0).squeeze(0)

                batch_losses["main"].append(main_loss.item())
                batch_losses["gating"].append(gating_loss.item())
                acc_metrics = calculate_accuracy_metrics(generated_ids_for_acc, labels, logits)
                for k, v in acc_metrics.items(): batch_metrics[k].append(v)
                for k, v in gate_metrics.items(): batch_metrics[k].append(v)
                
            global_step += input_ids_batch.shape[0]

            if batch_idx > 0 and batch_idx % config.LOG_INTERVAL == 0:
                avg_main_loss = sum(batch_losses["main"]) / len(batch_losses["main"])
                avg_gating_loss = sum(batch_losses["gating"]) / len(batch_losses["gating"])
                avg_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items() if v}
                ips = (config.LOG_INTERVAL * config.BATCH_SIZE) / (time.time() - last_log_time)
                last_log_time = time.time()
                observer.log_step(epoch, global_step, {"main": avg_main_loss, "gating": avg_gating_loss}, avg_metrics, ips)

            if global_step > 0 and global_step % config.EVAL_INTERVAL == 0:
                run_evaluation(model, eval_loader, observer, config)
                ckpt_path = checkpoint_dir / f"ckpt_step_{global_step}.pt"
                torch.save({'epoch': epoch, 'global_step': global_step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)
                checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=os.path.getctime)
                if len(checkpoints) > config.MAX_CHECKPOINTS: os.remove(checkpoints[0])
        
        track_and_remake_experts(model, dict(expert_activation_tracker), console)

if __name__ == "__main__":
    main()
