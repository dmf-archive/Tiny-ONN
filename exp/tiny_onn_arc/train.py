import os
import time
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from rich.console import Console

from .config import TinyOnnArcConfig
from .data import get_arc_dataset
from .model import TinyOnnForArcReconstruction, ExpertID

# --- HYPERPARAMETERS ---
TRAINING_MODE = 1
BATCH_SIZE = 8 if TRAINING_MODE == 0 else 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
CLIP_GRAD_NORM = 1.0
CHECKPOINT_DIR = "exp/tiny_onn_arc/checkpoints"
DEVICE = "cpu"
EPOCHS = 1000
LOG_INTERVAL = 1
EVAL_INTERVAL = 100
EVAL_BATCHES = 1
MAX_CHECKPOINTS = 3

ARC_COLORS = ["black", "blue", "red", "green", "yellow", "grey82", "magenta", "orange", "cyan", "white"]
SPLIT_POINT = 30 * 31

def get_latest_checkpoint(path: Path) -> Path | None:
    checkpoints = list(path.glob("*.pt"))
    return max(checkpoints, key=os.path.getctime) if checkpoints else None

def calculate_gating_loss_and_metrics(
    config: TinyOnnArcConfig, main_loss: torch.Tensor, aux_outputs: Dict[ExpertID, Dict]
) -> Tuple[torch.Tensor, Dict[str, float], float]:
    metrics = defaultdict(list)
    total_gating_loss = torch.tensor(0.0, device=DEVICE)
    expert_outputs = [cache["final_output"] for cache in aux_outputs.values()]
    if not expert_outputs:
        return total_gating_loss, {k: 0.0 for k in ["smha_gate_acc", "moe_gate_acc", "smha_avg_k", "moe_avg_k"]}, 0.0

    grads = torch.autograd.grad(main_loss, expert_outputs, retain_graph=True)
    all_surprises = []

    for i, (expert_id, cache) in enumerate(aux_outputs.items()):
        type, _, _ = expert_id
        B, T = cache["B"], cache["T"]
        logits = cache["gate_cache"]["logits"].view(B * T, -1)
        grad_norm = torch.linalg.norm(grads[i].view(B * T, -1), dim=-1).view(-1, 1)
        surprise = grad_norm.expand(-1, logits.shape[-1])
        all_surprises.append(surprise.flatten())

        targets = torch.argmin(surprise, dim=-1)
        metrics[f"{type}_gate_acc"].append((logits.argmax(-1) == targets).float().mean().item())
        
        w_ce, w_kl, w_aux = getattr(config, f"w_ce_{type}"), getattr(config, f"w_kl_{type}"), getattr(config, f"w_aux_{type}")
        g_loss = w_ce * F.cross_entropy(logits, targets)
        g_loss += w_kl * F.kl_div(F.log_softmax(logits, -1), F.log_softmax(-surprise, -1), log_target=True, reduction="batchmean")
        total_gating_loss += w_aux * g_loss
        
        mask = cache["gate_cache"]["activation_mask"].view(B * T, -1)
        metrics[f"{type}_avg_k"].append(mask.float().sum(1).mean().item())

    avg_surprise = torch.cat(all_surprises).mean().item()
    processed_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
    return total_gating_loss, processed_metrics, avg_surprise

def calculate_accuracy_metrics(generated_ids: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor) -> Dict[str, float]:
    gen_output = generated_ids[:, SPLIT_POINT:]
    label_output = labels[:, SPLIT_POINT:]
    grid_acc = (gen_output == label_output).all(dim=-1).float().mean().item()

    mask = labels != -100
    tok_acc = (logits.argmax(-1)[mask] == labels[mask]).float().mean().item()
    return {"tok_acc": tok_acc, "grid_acc": grid_acc}

def calculate_pi_score(config: TinyOnnArcConfig, main_loss: float, avg_surprise: float, logits: torch.Tensor) -> float:
    mask = logits.argmax(-1) != -100
    tau = torch.distributions.Categorical(logits=logits).entropy()[mask].mean().item()
    pi_score = math.exp(-config.pi_alpha * ((1 - config.pi_gamma) * (main_loss / (tau + 1e-9)) + config.pi_gamma * avg_surprise))
    return pi_score

def render_arc_grids(console: Console, input_ids: torch.Tensor, labels: torch.Tensor, generated_ids: torch.Tensor, batch_idx: int):
    def to_grid(seq: torch.Tensor) -> list[list[int]]:
        return seq.view(30, 31)[:, :-1].tolist()

    input_grid = to_grid(input_ids[batch_idx, :SPLIT_POINT].cpu())
    target_seq = labels[batch_idx, SPLIT_POINT:].clone().cpu()
    target_seq[target_seq == -100] = 0
    target_grid = to_grid(target_seq)
    pred_grid = to_grid(generated_ids[batch_idx, SPLIT_POINT:].cpu())

    def format_grid_rows(grid: list[list[int]]) -> list[str]:
        rows = []
        for r in grid[:15]:
            row_str = ""
            for p in r[:15]:
                color = ARC_COLORS[p] if 0 <= p < len(ARC_COLORS) else "white"
                row_str += f"[{color}]█[/]"
            rows.append(row_str)
        return rows

    input_rows = format_grid_rows(input_grid)
    target_rows = format_grid_rows(target_grid)
    pred_rows = format_grid_rows(pred_grid)

    console.print(f"\n--- Sample {batch_idx} ---")
    title_width = 15 * 2
    console.print(f"{'[bold]Input[/bold]':<{title_width}} {'[bold]Target[/bold]':<{title_width}} {'[bold]Prediction[/bold]':<{title_width}}")
    for i in range(15):
        console.print(f"{input_rows[i]:<{title_width}} {target_rows[i]:<{title_width}} {pred_rows[i]:<{title_width}}")

def run_evaluation(model: TinyOnnForArcReconstruction, eval_loader: DataLoader, console: Console):
    model.eval()
    total_tok_acc, total_grid_acc, count = 0, 0, 0
    with torch.no_grad():
        for i, (input_ids, labels) in enumerate(eval_loader):
            if i >= EVAL_BATCHES: break
            input_context = input_ids[:, :SPLIT_POINT]
            max_new = labels.shape[1] - SPLIT_POINT
            generated_ids = model.generate(input_context, max_new_tokens=max_new)
            
            gen_output = generated_ids[:, SPLIT_POINT:]
            label_output = labels[:, SPLIT_POINT:]
            total_grid_acc += (gen_output == label_output).all(dim=-1).float().sum().item()
            
            logits, _, _ = model(generated_ids)
            mask = labels != -100
            total_tok_acc += (logits.argmax(-1)[mask] == labels[mask]).float().sum().item()
            count += input_ids.shape[0]

            if i < 4:
                render_arc_grids(console, input_ids, labels, generated_ids, 0)
    
    avg_tok_acc = total_tok_acc / (count * (labels.shape[1] - SPLIT_POINT))
    avg_grid_acc = total_grid_acc / count
    console.print(f"\n[bold yellow]Eval Results | Token Acc: {avg_tok_acc:.4f} | Grid Acc: {avg_grid_acc:.4f}[/bold yellow]\n")
    model.train()

def main():
    config = TinyOnnArcConfig()
    device = torch.device(DEVICE)
    console = Console()
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)
    
    train_dataset, eval_dataset, train_collator, eval_collator = get_arc_dataset(device=device)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_collator)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=eval_collator)
    
    model = TinyOnnForArcReconstruction(config).to(device)
    
    if ckpt_path := get_latest_checkpoint(checkpoint_dir):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        console.print(f"Resumed model from checkpoint {ckpt_path}")
        
    model = torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    start_epoch, global_step = 0, 0
    if ckpt_path and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        console.print(f"Resumed optimizer and state from step {global_step}")

    last_log_time = time.time()
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        for input_ids, labels in train_loader:
            if input_ids is None: continue

            optimizer.zero_grad()

            if TRAINING_MODE == 0:
                logits, aux_outputs, _ = model(input_ids)
                generated_ids_for_acc = logits.argmax(-1)
            else:
                input_context = input_ids[:, :SPLIT_POINT]
                max_new = labels.shape[1] - SPLIT_POINT
                generated_ids = model.generate(input_context, max_new_tokens=max_new)
                logits, aux_outputs, _ = model(generated_ids.detach())
                generated_ids_for_acc = generated_ids

            main_loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1), ignore_index=-100)
            
            if torch.isnan(main_loss) or not aux_outputs:
                global_step += 1
                continue

            gating_loss, gate_metrics, avg_surprise = calculate_gating_loss_and_metrics(config, main_loss, aux_outputs)
            total_loss = main_loss + gating_loss
            
            if not torch.isnan(total_loss):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                optimizer.step()

            if global_step > 0 and global_step % LOG_INTERVAL == 0:
                acc_metrics = calculate_accuracy_metrics(generated_ids_for_acc, labels, logits)
                pi_score = calculate_pi_score(config, main_loss.item(), avg_surprise, logits)
                ips = LOG_INTERVAL / (time.time() - last_log_time)
                last_log_time = time.time()
                
                log_str = (
                    f"E:{epoch} S:{global_step} | L(M/G):{main_loss.item():.2f}/{gating_loss.item():.2f} | "
                    f"Acc(Tok/Grid):{acc_metrics['tok_acc']:.2f}/{acc_metrics['grid_acc']:.2f} | "
                    f"GateAcc(S/M):{gate_metrics.get('smha_gate_acc', 0):.2f}/{gate_metrics.get('moe_gate_acc', 0):.2f} | "
                    f"K(S/M):{gate_metrics.get('smha_avg_k', 0):.2f}/{gate_metrics.get('moe_avg_k', 0):.2f} | "
                    f"PI:{pi_score:.2f} | {ips:.2f}it/s"
                )
                console.print(log_str)

            if global_step > 0 and global_step % EVAL_INTERVAL == 0:
                run_evaluation(model, eval_loader, console)
                ckpt_path = checkpoint_dir / f"ckpt_step_{global_step}.pt"
                torch.save({
                    'epoch': epoch, 
                    'global_step': global_step, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, ckpt_path)

                checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=os.path.getctime)
                if len(checkpoints) > MAX_CHECKPOINTS:
                    os.remove(checkpoints[0])

            global_step += 1

if __name__ == "__main__":
    main()
