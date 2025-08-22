import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from rich.console import Console
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import Config
from .data import get_arc_dataloaders
from .model import ExpertID, TinyOnnForArcReconstruction
from .observer import Observer, calculate_pi_score


def get_latest_checkpoint(path: Path) -> Path | None:
    checkpoints = list(path.glob("*.pt"))
    return max(checkpoints, key=os.path.getctime) if checkpoints else None

def calculate_gating_loss_and_metrics(
    config: Config, main_loss: torch.Tensor, aux_outputs: dict[ExpertID, dict]
) -> tuple[torch.Tensor, dict[str, float], float]:
    metrics = defaultdict(list)
    total_gating_loss = torch.tensor(0.0, device=config.DEVICE)
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
            input_context = input_ids[:, :input_len]
            mask_context = attention_mask[:, :input_len]
            max_new = labels.shape[1] - input_len

            generated_ids = model.generate(input_context, mask_context, max_new_tokens=max_new)

            logits, _, _ = model(generated_ids, attention_mask=attention_mask)
            metrics = calculate_accuracy_metrics(generated_ids, labels, logits)

            total_tok_acc += metrics["tok_acc"] * input_ids.shape[0]
            total_grid_acc += metrics["grid_acc"] * input_ids.shape[0]
            count += input_ids.shape[0]

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
        for input_ids, labels, attention_mask in train_loader:
            optimizer.zero_grad()

            if config.TRAINING_MODE == 0:
                logits, aux_outputs, _ = model(input_ids, attention_mask=attention_mask)
                generated_ids_for_acc = logits.argmax(-1)
            else:
                input_len = (labels[0] == -100).sum()
                input_context = input_ids[:, :input_len]
                mask_context = attention_mask[:, :input_len]
                max_new = labels.shape[1] - input_len
                generated_ids = model.generate(input_context, mask_context, max_new_tokens=max_new)
                logits, aux_outputs, _ = model(generated_ids.detach(), attention_mask=F.pad(attention_mask, (0, generated_ids.shape[1] - attention_mask.shape[1]), "constant", 1))
                generated_ids_for_acc = generated_ids

            main_loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1), ignore_index=-100)

            if torch.isnan(main_loss) or not aux_outputs:
                global_step += 1
                continue

            gating_loss, gate_metrics, avg_surprise = calculate_gating_loss_and_metrics(config, main_loss, aux_outputs)
            total_loss = main_loss + gating_loss

            if not torch.isnan(total_loss):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
                optimizer.step()

            if global_step > 0 and global_step % config.LOG_INTERVAL == 0:
                acc_metrics = calculate_accuracy_metrics(generated_ids_for_acc, labels, logits)
                pi_score = calculate_pi_score(config, main_loss.item(), avg_surprise, logits)
                gate_metrics["pi_score"] = pi_score
                ips = config.LOG_INTERVAL / (time.time() - last_log_time)
                last_log_time = time.time()

                observer.log_step(epoch, global_step, {"main": main_loss.item(), "gating": gating_loss.item()}, {**acc_metrics, **gate_metrics}, ips)

            if global_step > 0 and global_step % config.EVAL_INTERVAL == 0:
                run_evaluation(model, eval_loader, observer, config)
                ckpt_path = checkpoint_dir / f"ckpt_step_{global_step}.pt"
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, ckpt_path)

                checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=os.path.getctime)
                if len(checkpoints) > config.MAX_CHECKPOINTS:
                    os.remove(checkpoints[0])

            global_step += 1

if __name__ == "__main__":
    main()
