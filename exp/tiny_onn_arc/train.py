import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Set

import torch
import torch.nn.functional as F
from rich.console import Console
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import Config
from .data import get_arc_dataloaders
from .model import DynONNForArc, DynSMHALayer, DynamicMoELayer
from .observer import Observer
from .tokenizer import ArcTokenizer
from .utils.aug_score import generate_candidates, calculate_augmented_score
from .utils.batch_transforms import apply_batch_augmentations, apply_batch_color_remap


def get_latest_checkpoint(path: Path) -> Path | None:
    checkpoints = list(path.glob("*.pt"))
    return max(checkpoints, key=os.path.getctime) if checkpoints else None


def get_expert_param_names(model: DynONNForArc) -> Set[str]:
    expert_names: Set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, (DynSMHALayer, DynamicMoELayer)):
            for param_name, _ in module.named_parameters():
                expert_names.add(f"{name}.{param_name}")
    return expert_names


def calculate_gating_loss_and_metrics(
    aux_caches: list[dict[str, Any]], config: Config
) -> tuple[torch.Tensor, dict[str, float]]:
    metrics: dict[str, list[float]] = defaultdict(list)
    total_gating_loss = torch.tensor(0.0, device=config.DEVICE)
    if not aux_caches:
        return total_gating_loss, {}

    for cache in aux_caches:
        expert_type = cache["type"]
        logits, mask = cache["logits"], cache["activation_mask"].float()
        B_T, E = logits.shape
        w_div = getattr(config, f"w_{expert_type}_diversity")
        w_sparse = getattr(config, f"w_{expert_type}_sparsity")
        min_experts = getattr(config, f"min_{'attention' if expert_type == 'smha' else 'moe'}_experts")

        avg_k = mask.sum(dim=1).mean()
        sparsity_loss = F.mse_loss(avg_k, torch.tensor(min_experts, device=avg_k.device, dtype=avg_k.dtype))
        total_gating_loss += w_sparse * sparsity_loss
        metrics[f"{expert_type}_sparsity_loss"].append(sparsity_loss.item())

        gating_net = cache["gating_net_ref"]
        sim_matrix = torch.matmul(F.normalize(gating_net.sim_matrix, dim=0).T, F.normalize(gating_net.sim_matrix, dim=0))
        identity = torch.eye(E, device=sim_matrix.device)
        diversity_loss = torch.norm(sim_matrix - identity, p="fro")
        total_gating_loss += w_div * diversity_loss
        metrics[f"{expert_type}_diversity_loss"].append(diversity_loss.item())
        metrics[f"{expert_type}_avg_k"].append(avg_k.item())

    return total_gating_loss, {k: sum(v) / len(v) for k, v in metrics.items()}


def calculate_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    
    h_pred, w_pred = preds.shape[1], preds.shape[2]
    h_targ, w_targ = targets.shape[1], targets.shape[2]
    
    if h_pred != h_targ or w_pred != w_targ:
        targets = F.pad(targets, (0, w_pred - w_targ, 0, h_pred - h_targ), "constant", ArcTokenizer.PAD_TOKEN_ID)

    mask = targets != ArcTokenizer.PAD_TOKEN_ID
    
    correct_tokens = (preds[mask] == targets[mask]).float()
    token_acc = correct_tokens.mean().item() if mask.any() else 0.0
    
    correct_grids = ((preds == targets) | ~mask).all(dim=(1, 2))
    grid_acc = correct_grids.float().mean().item()

    return {"token_acc": token_acc, "grid_acc": grid_acc}


def calculate_pi_score(
    model: DynONNForArc, main_loss: torch.Tensor, logits: torch.Tensor, config: Config, expert_names: Set[str]
) -> dict[str, float]:
    surprise_norms = [
        torch.norm(p.grad.detach(), p=2) for n, p in model.named_parameters() if n in expert_names and p.grad is not None
    ]
    surprise_tensor = torch.stack(surprise_norms).mean() if surprise_norms else torch.tensor(0.0, device=main_loss.device)
    
    probs = F.softmax(logits, dim=-1)
    tau_tensor = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
    
    pi_score = torch.exp(-config.pi_alpha * ((main_loss.detach() / (tau_tensor + 1e-9)) + config.pi_gamma * surprise_tensor))
    
    return {"pi_score": pi_score.item(), "surprise": surprise_tensor.item(), "tau": tau_tensor.item()}


def train_one_epoch(
    model: DynONNForArc, train_loader: DataLoader, eval_loader: DataLoader, optimizer: AdamW, observer: Observer, config: Config, epoch: int, global_step: int, expert_names: Set[str]
) -> int:
    model.train()
    for batch in train_loader:
        start_time = time.time()
        
        inputs, targets = batch["input"].to(config.DEVICE), batch["output"].to(config.DEVICE)
        inputs, targets = apply_batch_color_remap(inputs, targets)
        aug_inputs, aug_targets = apply_batch_augmentations(inputs, targets)

        optimizer.zero_grad()

        logits_orig, caches_orig = model(input_grid=inputs, output_grid=targets)
        logits_aug, caches_aug = model(input_grid=aug_inputs, output_grid=aug_targets)

        loss_orig = F.cross_entropy(logits_orig.permute(0, 3, 1, 2), targets, ignore_index=ArcTokenizer.PAD_TOKEN_ID)
        
        h_aug_logits, w_aug_logits = logits_aug.shape[1], logits_aug.shape[2]
        h_aug_targs, w_aug_targs = aug_targets.shape[1], aug_targets.shape[2]
        aug_targets_padded = F.pad(aug_targets, (0, w_aug_logits - w_aug_targs, 0, h_aug_logits - h_aug_targs), "constant", ArcTokenizer.PAD_TOKEN_ID)
        loss_aug = F.cross_entropy(logits_aug.permute(0, 3, 1, 2), aug_targets_padded, ignore_index=ArcTokenizer.PAD_TOKEN_ID)

        main_loss = loss_orig + config.w_consistency_loss * loss_aug
        gating_loss, gate_metrics = calculate_gating_loss_and_metrics(caches_orig + caches_aug, config)
        total_loss = main_loss + gating_loss

        if torch.isnan(total_loss):
            observer.console.print(f"[yellow]Warning: NaN detected at step {global_step}. Skipping.[/yellow]")
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
        optimizer.step()

        global_step += 1
        if global_step % config.LOG_INTERVAL == 0:
            preds = torch.argmax(logits_orig.detach(), dim=-1)
            acc_metrics = calculate_accuracy(preds, targets)
            pi_metrics = calculate_pi_score(model, main_loss, logits_orig.detach(), config, expert_names)
            losses = {"main": loss_orig.item(), "consistency": loss_aug.item(), "gating": gating_loss.item()}
            ips = config.BATCH_SIZE / (time.time() - start_time)
            observer.log_step(epoch, global_step, losses, {**acc_metrics, **gate_metrics, **pi_metrics}, ips)

            if global_step > 0 and global_step % config.EVAL_INTERVAL == 0:
                run_evaluation(model, eval_loader, observer, config, global_step)
                
                checkpoint_dir = Path(config.CHECKPOINT_DIR)
                ckpt_path = checkpoint_dir / f"ckpt_step_{global_step}.pt"
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, ckpt_path)
                
                checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=os.path.getctime)
                if len(checkpoints) > config.MAX_CHECKPOINTS:
                    os.remove(checkpoints[0])
            
    return global_step


def run_evaluation(model: DynONNForArc, loader: DataLoader, observer: Observer, config: Config, global_step: int):
    model.eval()
    correct_count, total_count = 0, 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= config.EVAL_BATCHES: break
            
            inputs, targets = batch["input"].to(config.DEVICE), batch["output"].to(config.DEVICE)
            
            for j in range(inputs.shape[0]):
                input_grid, target_grid = inputs[j], targets[j]
                
                candidates = generate_candidates(
                    model, input_grid, target_grid.shape[0], target_grid.shape[1], num_candidates=8, strategy="greedy"
                )
                
                scores = [calculate_augmented_score(model, input_grid, cand) for cand in candidates]
                best_pred = candidates[torch.argmax(torch.tensor(scores, device=config.DEVICE))]
                
                if torch.equal(best_pred, target_grid):
                    correct_count += 1
                total_count += 1
                
                if i == 0 and j < 2:
                    observer.visualize_evaluation_sample(input_grid, target_grid, best_pred, j)

    metrics = {"aug_score_grid_acc": correct_count / total_count if total_count > 0 else 0.0, "correct_count": correct_count, "total_count": total_count}
    observer.log_eval_summary(metrics, global_step)


def handle_expert_regeneration(model: DynONNForArc, observer: Observer):
    observer.console.print("[bold cyan]Running expert regeneration check...[/bold cyan]")
    total_regenerated = 0
    for name, module in model.named_modules():
        if hasattr(module, "regenerate_dead_experts"):
            num_regenerated = module.regenerate_dead_experts()
            if num_regenerated > 0:
                observer.console.print(f"  - Module '{name}': Regenerated {num_regenerated} dead experts.")
                total_regenerated += num_regenerated
    if total_regenerated == 0:
        observer.console.print("  - No dead experts found.")


def main():
    config = Config()
    console = Console()
    observer = Observer(console, config)
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)
    
    train_loader, eval_loader, _, _ = get_arc_dataloaders(config)
    model = DynONNForArc(config).to(config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    expert_names = get_expert_param_names(model)

    start_epoch, global_step = 0, 0
    if ckpt_path := get_latest_checkpoint(checkpoint_dir):
        ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch, global_step = ckpt.get("epoch", 0), ckpt.get("global_step", 0)
        console.print(f"[green]Resumed from step {global_step}[/green]")

    for epoch in range(start_epoch, config.EPOCHS):
        global_step = train_one_epoch(model, train_loader, eval_loader, optimizer, observer, config, epoch, global_step, expert_names)
        handle_expert_regeneration(model, observer)


if __name__ == "__main__":
    main()
