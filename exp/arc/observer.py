import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from sklearn.decomposition import PCA

from .config import TrainConfig


class Observer:
    ARC_COLORS: list[str] = [
        "black", "blue", "red", "green", "yellow",
        "grey", "magenta", "orange", "cyan", "brown",
    ]

    def __init__(self, console: Console, config: TrainConfig):
        self.console = console
        self.config = config
        self.vis_dir = Path(__file__).parent / "pic"
        self.vis_dir.mkdir(exist_ok=True)
        self.max_vis_files = 5

    def calculate_metrics(
        self, main_loss: torch.Tensor, model_outputs: dict, signals: dict, input_ids: torch.Tensor, model: nn.Module
    ) -> dict[str, float]:
        logits, labels = model_outputs["logits"], model_outputs["labels"]
        routing_logits = model_outputs.get("routing_logits", [])
        masked_routing_logits = model_outputs.get("masked_routing_logits", [])

        logits_acc, labels_acc = logits[:, :-1, :], labels[:, 1:]
        mask = (labels_acc != -100) & (labels_acc != 0)
        active_logits = logits_acc[mask] if mask.any() else torch.empty(0, device=logits.device)
        acc = (
            (torch.argmax(active_logits, dim=-1) == labels_acc[mask]).float().mean().item()
            if mask.any()
            else 0.0
        )

        inefficient_routing_rate = 0.0
        if masked_routing_logits:
            all_module_silent_masks = [torch.all(mrl == 0, dim=-1) for mrl in masked_routing_logits]
            token_is_fully_silent = torch.stack(all_module_silent_masks).all(dim=0)
            token_is_fully_silent_aligned = token_is_fully_silent[:, :-1]

            input_ids_aligned = input_ids[:, :-1]
            is_identity_transform = (input_ids_aligned == labels_acc)

            is_computation_needed = ~is_identity_transform & mask

            inefficient_routing_event = token_is_fully_silent_aligned & is_computation_needed

            num_computation_needed = is_computation_needed.sum()
            if num_computation_needed > 0:
                inefficient_routing_rate = (inefficient_routing_event.sum() / num_computation_needed).item()

        mu_grad_norms = [s.item() for s in signals.get("mu_grad_norms", []) if s.numel() > 0]
        all_surp_norms = mu_grad_norms
        complexity_cost = sum(all_surp_norms) if all_surp_norms else 0.0
        pi_score = torch.exp(-1.0 * main_loss.detach() - 1.0 * complexity_cost).item()

        num_spl_modules = self.config.model.num_layers * 4
        act_rates = [0.0] * num_spl_modules
        if routing_logits:
            act_rates = [rl.gt(0).float().mean().item() for rl in routing_logits]

        num_layers = self.config.model.num_layers

        metrics = {
            "main_loss": main_loss.item(),
            "token_acc": acc,
            "pi_score": pi_score,
            "meta_loss": signals.get("meta_loss", torch.tensor(0.0)).item(),
            "sample_entropy": model_outputs.get("sample_entropy", torch.tensor(0.0)).mean().item(),
            "tau": (
                -torch.sum(F.softmax(active_logits, dim=-1) * F.log_softmax(active_logits, dim=-1), dim=-1)
                .mean()
                .item()
                if active_logits.numel() > 0
                else 0.0
            ),
            "seq_len": float(labels.shape[1]),
            "activation_rate_avg": sum(act_rates) / len(act_rates) if act_rates else 0.0,
            "activation_rate_l0": sum(act_rates[:4]) / 4 if act_rates else 0.0,
            "activation_rate_l_mid": sum(act_rates[num_layers * 2 : num_layers * 2 + 4]) / 4 if len(act_rates) > num_layers * 2 + 4 else 0.0,
            "activation_rate_ln": sum(act_rates[-4:]) / 4 if act_rates else 0.0,
            "act_rates": act_rates,
            "inefficient_routing_rate": inefficient_routing_rate,
        }

        routing_logits = model_outputs.get("routing_logits", [])
        if routing_logits:
            flat_logits = torch.cat([rl.detach().float().view(-1) for rl in routing_logits if rl.numel() > 0])
            if flat_logits.numel() > 0:
                metrics["gate_logit_avg"] = flat_logits.mean().item()
                metrics["gate_logit_max"] = flat_logits.max().item()
                metrics["gate_logit_sigma"] = flat_logits.std().item()

        goodness_logits_list = signals.get("goodness_logits", [])
        if goodness_logits_list:
            flat_goodness = torch.cat([g.detach().float().view(-1) for g in goodness_logits_list if g.numel() > 0])
            if flat_goodness.numel() > 0:
                total_neurons = flat_goodness.numel()
                metrics["goodness_rate"] = (torch.sum(flat_goodness > 0) / total_neurons).item()
                metrics["badness_rate"] = (torch.sum(flat_goodness < 0) / total_neurons).item()
                metrics["shutdown_rate"] = (torch.sum(flat_goodness == 0) / total_neurons).item()

        return metrics

    def log_step(
        self,
        epoch: int,
        step: int,
        task_idx: int | str,
        view_idx: int,
        metrics: dict[str, float],
        elapsed_time: float,
    ):
        steps_per_sec = 1 / elapsed_time if elapsed_time > 0 else float("inf")

        log_str = (
            f"E{epoch} S{step} T{task_idx} V{view_idx} | "
            f"L({metrics.get('main_loss', 0.0):.3f}/{metrics.get('meta_loss', 0.0):.4f}) | "
            f"Acc: {metrics.get('token_acc', 0.0):.3f} | "
            f"τ: {metrics.get('tau', 0.0):.3f} | "
            f"H(x): {metrics.get('sample_entropy', 0.0):.3f} | "
            f"Seq: {int(metrics.get('seq_len', 0))} | "
            f"PI: {metrics.get('pi_score', 0.0):.3f} | "
            f"Act%({metrics.get('activation_rate_l0', 0.0)*100:.1f}/{metrics.get('activation_rate_l_mid', 0.0)*100:.1f}/{metrics.get('activation_rate_ln', 0.0)*100:.1f}/{metrics.get('activation_rate_avg', 0.0)*100:.1f}) | "
            f"Gate({metrics.get('gate_logit_avg', 0.0):.3f}/{metrics.get('gate_logit_sigma', 0.0):.3f}/{metrics.get('gate_logit_max', 0.0):.3f}) | "
            f"GBS%({metrics.get('goodness_rate', 0.0)*100:.1f}/{metrics.get('badness_rate', 0.0)*100:.1f}/{metrics.get('shutdown_rate', 0.0)*100:.1f}) | "
            f"Fail: {metrics.get('inefficient_routing_rate', 0.0)*100:.1f}% | "
            f"Speed: {steps_per_sec:.2f} st/s"
        )
        self.console.print(log_str)

    def visualize_prototypes(self, signals: dict, global_step: int):
        proto_weights = signals.get("proto_weights", [])
        routing_logits = signals.get("routing_logits", [])
        goodness_logits = signals.get("goodness_logits", [])
 
        if not proto_weights or not routing_logits or not goodness_logits:
            return

        num_spl = 4
        num_layers = self.config.model.num_layers

        fig, axes = plt.subplots(num_layers, num_spl, figsize=(15, 5 * num_layers), squeeze=False)
        fig.suptitle(f"Prototype Distribution @ Step {global_step}", fontsize=16)
        color_map = {"good": "green", "bad": "red", "inactive": "grey"}
        names = ["attn_q", "attn_k", "attn_v", "attn_o"]

        for i in range(num_layers):
            for j in range(num_spl):
                ax = axes[i, j]
                idx = i * num_spl + j

                if idx >= len(proto_weights): continue

                protos = proto_weights[idx].cpu().to(torch.float32)
                rl = routing_logits[idx].cpu().to(torch.float32)
                logits_raw = goodness_logits[idx].cpu().to(torch.float32)
                logits = torch.mean(logits_raw, dim=(0, 1)) if logits_raw.ndim == 3 else logits_raw
 
                num_activations = (rl > 0).float().sum(dim=(0, 1))

                statuses = []
                for k in range(logits.shape[0]):
                    if num_activations[k] == 0:
                        statuses.append("inactive")
                    elif logits[k] > 0:
                        statuses.append("good")
                    else:
                        statuses.append("bad")

                if protos.shape[0] > 1:
                    pca = PCA(n_components=2)
                    protos_2d = pca.fit_transform(protos.numpy())
                    colors = [color_map.get(s, "black") for s in statuses]
                    ax.scatter(protos_2d[:, 0], protos_2d[:, 1], c=colors, alpha=0.7, s=10)
                else:
                    ax.text(0.5, 0.5, "Not enough data", ha='center', va='center')

                ax.set_title(f"L{i} - {names[j]}")
                ax.grid(True)

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        save_path = self.vis_dir / f"prototypes_step_{global_step}.png"
        plt.savefig(save_path)
        plt.close(fig)

        vis_files = sorted(self.vis_dir.glob("*.png"), key=os.path.getmtime)
        if len(vis_files) > self.max_vis_files:
            os.remove(vis_files[0])


    def maybe_log_and_visualize(
        self,
        epoch: int,
        step: int,
        task_idx: int,
        view_idx: int,
        metrics: dict,
        elapsed_time: float,
        signals: dict,
        evaluator,
        eval_loader,
        current_task_idx: int,
        save_checkpoint_fn,
        advance_curriculum_fn: callable,
        curriculum_stage: int,
    ):
        if step % self.config.log_interval == 0:
            self.log_step(epoch, step, task_idx, view_idx, metrics, elapsed_time)
            self.visualize_prototypes(signals, step)
            save_checkpoint_fn(task_idx, view_idx)

        if step > 0 and step % self.config.eval_interval == 0:
            evaluator.run(eval_loader, current_task_idx, step, curriculum_stage, advance_curriculum_fn)

    def _create_grid_text(self, grid: torch.Tensor) -> Text:
        text = Text()
        h, w = grid.shape
        for r in range(h):
            for p in range(w):
                pixel = int(grid[r, p].item())
                color = self.ARC_COLORS[pixel] if 0 <= pixel < len(self.ARC_COLORS) else "white"
                text.append("■ ", style=color)
            if r < h - 1:
                text.append("\n")
        return text

    def visualize_evaluation_sample(
        self,
        input_grid: torch.Tensor,
        target_grid: torch.Tensor,
        pred_grid: torch.Tensor | None,
        pred_tokens_decoded: str | None,
        probabilities: list[float] | None,
        step: int,
    ):
        self.console.print()
        self.console.print(f"--- Evaluation Sample @ Step {step} ---", style="bold yellow")

        if pred_grid is None:
            pred_grid = torch.zeros((1, 1), dtype=torch.long)

        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Input", justify="center")
        table.add_column("Target", justify="center")
        table.add_column("Prediction", justify="center")

        input_text = self._create_grid_text(input_grid)
        target_text = self._create_grid_text(target_grid)
        pred_text = self._create_grid_text(pred_grid)

        table.add_row(input_text, target_text, pred_text)
        self.console.print(table)

        if pred_tokens_decoded:
            num_tokens = len(pred_tokens_decoded.split())
            self.console.print(f"[bold]Generated Token Stream ({num_tokens} tokens):[/bold]", highlight=False)
            self.console.print(pred_tokens_decoded)
        else:
            self.console.print("[bold]Generated Token Stream:[/bold] [red]N/A[/red]")

        if probabilities:
            prob_str = " ".join([f"{p:.2f}" for p in probabilities])
            self.console.print(f"[bold]Probabilities:[/bold] {prob_str}")

        self.console.print()

    def log_eval_summary(self, metrics: dict[str, float], step: int):
        eval_loss = metrics.get("eval_loss", 0.0)
        eval_grid_acc = metrics.get("eval_grid_acc", 0.0)
        total_count = int(metrics.get("total_count", 0))
        summary_text = f"Avg Loss: {eval_loss:.4f} | Grid Accuracy: {eval_grid_acc:.4f} on {total_count} samples."
        self.console.print(
            Padding(
                f"[bold yellow]===== EVALUATION COMPLETE (Step: {step}) =====\n"
                f"{summary_text}\n"
                f"=============================================[/bold yellow]",
                (1, 2),
            )
        )
