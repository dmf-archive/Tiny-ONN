import os
import time
from pathlib import Path

import bayesian_torch.layers as bl
import torch
import torch.nn.functional as F
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
from rich.console import Console
from torch.optim import AdamW

from .bayesian_config import BayesianConfig
from .bayesian_model import BayesianTinyOnn
from .bayesian_observer import BayesianObserver
from .data import get_arc_dataloaders


def calculate_grid_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=3)
    correct = (preds == labels).all(dim=(1, 2))
    return correct.float().mean().item()


def get_sigma_stats(model: BayesianTinyOnn) -> dict[str, float]:
    sigmas = []
    for module in model.modules():
        if isinstance(module, bl.LinearReparameterization):
            sigmas.append(F.softplus(module.rho_weight).detach())
    if not sigmas:
        return {'sigma_mean': 0.0, 'sigma_std': 0.0}
    all_sigma = torch.cat([s.view(-1) for s in sigmas])
    return {'sigma_mean': all_sigma.mean().item(), 'sigma_std': all_sigma.std().item()}


def get_latest_checkpoint(path: Path) -> Path | None:
    checkpoints = list(path.glob("*.pt"))
    return max(checkpoints, key=os.path.getctime) if checkpoints else None


def run_evaluation(model: BayesianTinyOnn, eval_loader, observer: BayesianObserver, config: BayesianConfig):
    model.eval()
    total_grid_acc = 0.0
    with torch.no_grad():
        for i, (input_grids, output_grids) in enumerate(eval_loader):
            if i >= config.EVAL_BATCHES:
                break
            logits = model(input_grids)
            total_grid_acc += calculate_grid_accuracy(logits, output_grids)
            if i == 0:
                preds = torch.argmax(logits, dim=3)
                observer.visualize_batch(input_grids, output_grids, preds)
    avg_grid_acc = total_grid_acc / config.EVAL_BATCHES
    observer.log_eval_results({"grid_acc": avg_grid_acc})
    model.train()


def main():
    config = BayesianConfig()
    device = torch.device(config.DEVICE)
    console = Console()
    observer = BayesianObserver(console, config)

    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)

    train_loader, eval_loader, train_size, _ = get_arc_dataloaders(config)
    model = BayesianTinyOnn(config).to(device, dtype=config.DTYPE)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    start_epoch, global_step = 0, 0
    if ckpt_path := get_latest_checkpoint(checkpoint_dir):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        console.print(f"[green]Resumed from checkpoint: {ckpt_path} at step {global_step}[/green]")

    console.print("--- Starting Bayesian EAVI Training (Image-to-Image) ---")

    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        for batch_idx, (input_grids, output_grids) in enumerate(train_loader):
            batch_size = input_grids.shape[0]
            candidate_losses = torch.zeros(batch_size, config.pi_alpha, device=device)
            candidate_grads = [[] for _ in range(batch_size)]
            grad_norms = torch.zeros(batch_size, config.pi_alpha, device=device)

            for k in range(config.pi_alpha):
                logits = model(input_grids)
                task_loss_per_sample = F.cross_entropy(
                    logits.permute(0, 3, 1, 2), output_grids, reduction='none'
                ).mean(dim=[1, 2])
                kl_loss = get_kl_loss(model) / train_size
                total_loss_per_sample = task_loss_per_sample + kl_loss
                candidate_losses[:, k] = total_loss_per_sample.detach()

                for i in range(batch_size):
                    grads = torch.autograd.grad(total_loss_per_sample[i], model.parameters(), retain_graph=True)
                    flat_grads = torch.cat([g.view(-1) for g in grads if g is not None])
                    grad_norms[i, k] = torch.linalg.norm(flat_grads)
                    candidate_grads[i].append(grads)
            
            weights = F.softmax(-config.pi_gamma * grad_norms, dim=1)
            
            optimizer.zero_grad()
            avg_grads = [torch.zeros_like(p, requires_grad=False) for p in model.parameters()]
            for i in range(batch_size):
                for k in range(config.pi_alpha):
                    for i_p, grad in enumerate(candidate_grads[i][k]):
                        if grad is not None:
                            avg_grads[i_p] += (weights[i, k] / batch_size) * grad
            
            for i_p, p in enumerate(model.parameters()):
                p.grad = avg_grads[i_p]
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
            optimizer.step()

            global_step += 1

            if global_step % config.LOG_INTERVAL == 0:
                weighted_loss = (weights * candidate_losses).sum(dim=1).mean().item()
                mean_loss = candidate_losses.mean().item()
                mean_acc = calculate_grid_accuracy(logits.detach(), output_grids)
                surprise = grad_norms.mean().item()
                pi_score = torch.exp(-torch.tensor(mean_loss + config.pi_gamma * surprise)).item()
                
                stats = {
                    'step': global_step, 'weighted_loss': weighted_loss,
                    'mean_loss': mean_loss, 'mean_acc': mean_acc, 'pi_score': pi_score,
                    **get_sigma_stats(model)
                }
                observer.log_step(stats, epoch)

            if global_step > 0 and batch_idx % config.EVAL_INTERVAL == 0:
                run_evaluation(model, eval_loader, observer, config)
                ckpt_path = checkpoint_dir / f"ckpt_step_{global_step}.pt"
                torch.save({
                    'epoch': epoch, 'global_step': global_step, 
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict()
                }, ckpt_path)
                console.print(f"[green]Checkpoint saved to {ckpt_path}[/green]")
                
                checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=os.path.getctime)
                if len(checkpoints) > config.MAX_CHECKPOINTS:
                    os.remove(checkpoints[0])

if __name__ == "__main__":
    main()
