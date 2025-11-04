
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from .config import TrainConfig


class Observer:
    def __init__(self, console: Console, config: TrainConfig):
        self.console = console
        self.config = config
        self.ARC_COLORS = [
            "black", "blue", "red", "green", "yellow",
            "grey", "magenta", "cyan", "white", "bright_white"
        ]

    def calculate_metrics(
        self, main_loss: torch.Tensor, model_outputs: dict, signals: dict, input_ids: torch.Tensor, model: nn.Module
    ) -> dict[str, float]:
        logits, labels = model_outputs["logits"], model_outputs["labels"]

        logits_acc, labels_acc = logits[:, :-1, :], labels[:, 1:]
        mask = labels_acc != -100
        active_logits = logits_acc[mask] if mask.any() else torch.empty(0, device=logits.device)
        acc = (
            (torch.argmax(active_logits, dim=-1) == labels_acc[mask]).float().mean().item()
            if mask.any()
            else 0.0
        )

        loss_value = main_loss.item()
        grad_norm_value = signals.get("grad_norm", 0.0)
        pi_value = math.exp(-(loss_value + grad_norm_value))

        metrics = {
            "main_loss": loss_value,
            "token_acc": acc,
            "grad_norm": grad_norm_value,
            "sample_entropy": model_outputs.get("sample_entropy", torch.tensor(0.0)).mean().item(),
            "tau": (
                -torch.sum(F.softmax(active_logits, dim=-1) * F.log_softmax(active_logits, dim=-1), dim=-1)
                .mean()
                .item()
                if active_logits.numel() > 0
                else 0.0
            ),
            "seq_len": float(labels.shape[1]),
            "pi": pi_value,
        }

        return metrics

    def log_step(
        self,
        epoch: int,
        step: int,
        task_idx: int | str,
        metrics: dict[str, float],
        elapsed_time: float,
    ):
        steps_per_sec = 1 / elapsed_time if elapsed_time > 0 else float("inf")

        log_str = (
            f"E{epoch} S{step} T{task_idx} | "
            f"Loss: {metrics.get('main_loss', 0.0):.3f} | "
            f"Acc: {metrics.get('token_acc', 0.0):.3f} | "
            f"GradNorm: {metrics.get('grad_norm', 0.0):.3e} | "
            f"τ: {metrics.get('tau', 0.0):.3f} | "
            f"H(x): {metrics.get('sample_entropy', 0.0):.3f} | "
            f"PI: {metrics.get('pi', 0.0):.3f} | "
            f"Seq: {int(metrics.get('seq_len', 0))} | "
            f"Speed: {steps_per_sec:.2f} st/s"
        )
        self.console.print(log_str)

    def maybe_log_and_visualize(
        self,
        epoch: int,
        step: int,
        task_idx: int,
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
            self.log_step(epoch, step, task_idx, metrics, elapsed_time)
            save_checkpoint_fn(task_idx)

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
