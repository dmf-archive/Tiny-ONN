import glob
import os
import random
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from rich.console import Console
from rich.table import Table
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from .config import TinyOnnArcConfig
from .data import collate_fn, get_arc_dataset
from .model import TinyOnnForArcReconstruction

ARC_COLORS = [
    "black", "bright_blue", "bright_red", "bright_green", "yellow", 
    "grey50", "bright_magenta", "orange4", "bright_cyan", "bright_white", "white", "grey85"
]

def visualize_predictions(model_input: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor, step: int):
    console = Console()
    newline_token, pad_token = 10, -100
    h, w = 30, 30

    def to_grid(seq: torch.Tensor):
        pixel_seq = seq[seq != newline_token]
        
        if pad_token is not None:
            pixel_seq = pixel_seq[pixel_seq != pad_token]
        
        num_pixels = pixel_seq.numel()
        if num_pixels > h * w:
            pixel_seq = pixel_seq[:h*w]
        elif num_pixels < h * w:
            pixel_seq = F.pad(pixel_seq, (0, h * w - num_pixels), "constant", 0)
        
        return pixel_seq.view(h, w)

    loss_mask = labels != pad_token
    input_part = model_input[~loss_mask]
    target_part = labels[loss_mask]
    pred_part = logits.argmax(-1)[loss_mask]

    input_grid = to_grid(input_part)
    target_grid = to_grid(target_part)
    pred_grid = to_grid(pred_part)

    table = Table(title=f"Step {step}: Prediction")
    table.add_column("Input", justify="center")
    table.add_column("Target Output", justify="center")
    table.add_column("Model Prediction", justify="center")

    def grid_to_rich_text(grid):
        lines = []
        for row in grid.cpu().numpy():
            line = "".join(f"[{ARC_COLORS[p]}]■[/{ARC_COLORS[p]}]" for p in row)
            lines.append(line)
        return "\n".join(lines)

    table.add_row(
        grid_to_rich_text(input_grid),
        grid_to_rich_text(target_grid),
        grid_to_rich_text(pred_grid),
    )
    
    console.print(table)


def get_latest_checkpoint(checkpoint_dir: str) -> str | None:
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "ckpt_step_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.split("_")[-1].replace(".pt", "")))

def calculate_accuracy_metrics(logits: torch.Tensor, labels: torch.Tensor):
    metrics = {}
    with torch.no_grad():
        preds = logits.argmax(-1)
        loss_mask = labels != -100

        masked_preds = preds[loss_mask]
        masked_labels = labels[loss_mask]
        
        if masked_labels.numel() > 0:
            token_acc = (masked_preds == masked_labels).float().mean()
        else:
            token_acc = torch.tensor(1.0, device=preds.device)
        
        # Grid accuracy needs careful reshaping and comparison
        B = labels.size(0)
        grid_acc = 0.0
        if B > 0 and masked_labels.numel() > 0:
            # This is a simplification; true grid accuracy is harder in seq2seq
            # We check if all predictable tokens for each sample are correct
            correct_samples = 0
            for i in range(B):
                sample_mask = loss_mask[i]
                if torch.all(preds[i][sample_mask] == labels[i][sample_mask]):
                    correct_samples += 1
            grid_acc = correct_samples / B

    metrics["token_acc"] = token_acc.item()
    metrics["grid_acc"] = grid_acc
    return metrics


def calculate_gating_loss_and_metrics(config, main_loss, aux_outputs, expert_activation_counts):
    metrics = {"smha_gate_acc": [], "moe_gate_acc": [], "smha_avg_k": [], "moe_avg_k": [], "gating_loss": torch.tensor(0.0, device=main_loss.device)}
    
    outputs_to_grad = [cache["final_output"] for cache in aux_outputs.values()]
    if not outputs_to_grad:
        return metrics, metrics["gating_loss"]

    grads_for_surprise = torch.autograd.grad(main_loss, outputs_to_grad, create_graph=False, retain_graph=True)
    
    grad_idx = 0
    for expert_id, layer_cache in aux_outputs.items():
        expert_type, layer_index, _ = expert_id
        layer_type_str = "smha" if expert_type == "smha" else "moe"
        B, T = layer_cache["B"], layer_cache["T"]
        
        grad_output_flat = grads_for_surprise[grad_idx].view(B * T, -1)
        
        with torch.no_grad():
            hs_norm = layer_cache["normed_hs"].view(B*T, -1)
            
            if expert_type == "smha":
                # Simplified surprise for attention using L2 distance
                q_experts = torch.einsum("btc,ech->bteh", layer_cache['normed_hs'], layer_cache['layer'].q_proj).view(B*T, -1, config.head_dim)
                grad_at_q = torch.einsum("btc,ehc->bteh", grad_output_flat.view(B, T, -1), layer_cache['layer'].o_proj).view(B*T, -1, config.head_dim)
                surprise_matrix = torch.linalg.norm(q_experts - grad_at_q, dim=-1)
            else: # moe
                w2_experts = layer_cache['layer'].w2
                grad_at_intermediate = torch.einsum('btc,eic->btei', grad_output_flat.view(B,T,-1), w2_experts)
                surprise_matrix = torch.linalg.norm(grad_at_intermediate, dim=-1).view(B*T, -1)

        logits_gating = layer_cache["gate_cache"]["logits"].view(B * T, -1)
        
        with torch.no_grad():
            targets = torch.argmin(surprise_matrix, dim=-1).detach()
            acc = (logits_gating.argmax(dim=-1) == targets).float().mean()
            metrics[f"{layer_type_str}_gate_acc"].append(acc.item())
        
        log_targ = F.log_softmax(-surprise_matrix.detach(), dim=-1)
        log_gate = F.log_softmax(logits_gating, dim=-1)
        
        w_config = getattr(config, f"w_ce_{layer_type_str}"), getattr(config, f"w_kl_{layer_type_str}"), getattr(config, f"w_aux_{layer_type_str}")
        gating_loss = w_config[0] * F.cross_entropy(logits_gating, targets) + w_config[1] * F.kl_div(log_gate, log_targ, reduction="batchmean", log_target=True)
        metrics["gating_loss"] += w_config[2] * gating_loss
        
        activation_mask_flat = layer_cache["gate_cache"]["activation_mask"].view(B * T, -1).to(torch.bool)
        active_experts_in_batch = torch.where(activation_mask_flat.sum(dim=0) > 0)[0]
        for expert_sub_idx in active_experts_in_batch:
            expert_activation_counts[(expert_type, layer_index, expert_sub_idx.item())] += 1
        
        metrics[f"{layer_type_str}_avg_k"].append(activation_mask_flat.float().sum(dim=1).mean().item())
        grad_idx += 1

    return metrics, metrics["gating_loss"]

def run_evaluation(model, loader, device, global_step: int):
    model.eval()
    total_token_acc, total_grid_acc, count = 0, 0, 0
    with torch.no_grad():
        for i, (input_ids, labels) in enumerate(loader):
            if input_ids is None: continue
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            logits, _ = model(input_ids=input_ids)

            accuracy_metrics = calculate_accuracy_metrics(logits, labels)
            total_token_acc += accuracy_metrics['token_acc'] * input_ids.size(0)
            total_grid_acc += accuracy_metrics['grid_acc'] * input_ids.size(0)
            count += input_ids.size(0)

            if i == 0:
                visualize_predictions(input_ids[0], labels[0], logits[0], global_step)

    avg_token_acc = total_token_acc / count if count > 0 else 0
    avg_grid_acc = total_grid_acc / count if count > 0 else 0
    return avg_token_acc, avg_grid_acc


def main():
    config = TinyOnnArcConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = Path("exp/tiny_onn_arc/checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    recent_checkpoints = deque(maxlen=3)

    train_dataset, eval_dataset = get_arc_dataset(config)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    
    model = TinyOnnForArcReconstruction(config).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    start_epoch, global_step = 0, 0
    latest_ckpt_path = get_latest_checkpoint(str(checkpoint_dir))
    if latest_ckpt_path:
        print(f"Resuming from {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        global_step = ckpt['global_step']
        for ckpt_p in sorted(glob.glob(os.path.join(checkpoint_dir, "ckpt_step_*.pt")), key=os.path.getmtime)[-3:]:
            recent_checkpoints.append(ckpt_p)

    for epoch in range(start_epoch, 1000):
        model.train()
        expert_activation_counts = defaultdict(int)

        for i, (input_ids, labels) in enumerate(train_loader):
            if input_ids is None: continue
            
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits, aux_outputs = model(input_ids=input_ids)
            
            main_loss = F.cross_entropy(rearrange(logits, 'b t d -> (b t) d'), rearrange(labels, 'b t -> (b t)'), ignore_index=-100)
            
            if torch.isnan(main_loss) or main_loss.isinf():
                continue

            gating_metrics, gating_loss = calculate_gating_loss_and_metrics(config, main_loss, aux_outputs, expert_activation_counts)
            
            total_loss = main_loss + gating_loss
            if not (torch.isnan(total_loss) or total_loss.isinf()):
                total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if global_step > 0 and global_step % 20 == 0:
                accuracy_metrics = calculate_accuracy_metrics(logits, labels)

                smha_gate_acc_avg = sum(gating_metrics['smha_gate_acc']) / len(gating_metrics['smha_gate_acc']) if gating_metrics['smha_gate_acc'] else 0
                moe_gate_acc_avg = sum(gating_metrics['moe_gate_acc']) / len(gating_metrics['moe_gate_acc']) if gating_metrics['moe_gate_acc'] else 0
                smha_avg_k = sum(gating_metrics['smha_avg_k']) / len(gating_metrics['smha_avg_k']) if gating_metrics['smha_avg_k'] else 0
                moe_avg_k = sum(gating_metrics['moe_avg_k']) / len(gating_metrics['moe_avg_k']) if gating_metrics['moe_avg_k'] else 0

                log_str = (f"E:{epoch} S:{global_step} | L(M/G):{main_loss.item():.2f}/{gating_loss.item():.2f} | Acc(Tok/Grid):{accuracy_metrics['token_acc']:.2f}/{accuracy_metrics['grid_acc']:.2f} | GateAcc(S/M):{smha_gate_acc_avg:.2f}/{moe_gate_acc_avg:.2f} | K(S/M):{smha_avg_k:.2f}/{moe_avg_k:.2f}")
                print(log_str)
                
            global_step += 1

            if global_step > 0 and global_step % 100 == 0:
                eval_subset_indices = random.sample(range(len(eval_dataset)), k=min(20, len(eval_dataset)))
                eval_subset = Subset(eval_dataset, eval_subset_indices)
                quick_eval_loader = DataLoader(eval_subset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)
                avg_token_acc, avg_grid_acc = run_evaluation(model, quick_eval_loader, device, global_step=global_step)
                print(f"\n--- Step {global_step} Quick Eval --- Acc(Tok/Grid): {avg_token_acc:.4f}/{avg_grid_acc:.4f}\n")

                if avg_grid_acc >= 0.9:
                    print("\n--- Quick Eval passed 90% threshold. Running full evaluation... ---")
                    full_eval_loader = DataLoader(eval_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)
                    full_avg_token_acc, full_avg_grid_acc = run_evaluation(model, full_eval_loader, device, global_step=global_step)
                    print(f"--- Full Evaluation Result --- Acc(Tok/Grid): {full_avg_token_acc:.4f}/{full_avg_grid_acc:.4f}\n")
                    if full_avg_grid_acc >= 0.9:
                        print("--- Full evaluation passed 90% threshold. Training will stop after this checkpoint. ---")
                        # We save one last time and then break the loop
                        ckpt_path = checkpoint_dir / f"ckpt_step_{global_step}_final.pt"
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'model_config': vars(config)
                        }, ckpt_path)
                        print(f"Final checkpoint saved to {ckpt_path}")
                        break
                
                ckpt_path = checkpoint_dir / f"ckpt_step_{global_step}.pt"
                torch.save({
                    'epoch': epoch, 
                    'global_step': global_step, 
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_config': vars(config)
                }, ckpt_path)
                if len(recent_checkpoints) == recent_checkpoints.maxlen:
                    oldest_ckpt = recent_checkpoints.popleft()
                    if os.path.exists(oldest_ckpt):
                        os.remove(oldest_ckpt)
                recent_checkpoints.append(str(ckpt_path))
                print(f"Checkpoint saved to {ckpt_path}")

        dead_experts = [eid for eid, count in expert_activation_counts.items() if count == 0]
        if dead_experts:
            with torch.no_grad():
                for expert_type, layer_idx, expert_sub_idx in dead_experts:
                    layer = model.model.layers[layer_idx]
                    
                    sub_layer = layer.smha_layer if expert_type == "smha" else layer.moe_layer
                    gating_network = sub_layer.gating_network

                    new_weights = torch.randn_like(gating_network.sim_matrix[:, expert_sub_idx])
                    gating_network.sim_matrix[:, expert_sub_idx] = F.normalize(new_weights, dim=0)
                    gating_network.gates.data[expert_sub_idx] = 0.0

if __name__ == "__main__":
    main()
