import torch
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from .config import TrainConfig


class Observer:
    ARC_COLORS: list[str] = [
        "black", "blue", "red", "green", "yellow",
        "grey", "magenta", "orange", "cyan", "brown"
    ]

    def __init__(self, console: Console, config: TrainConfig):
        self.console = console
        self.config = config

    def log_step(self, epoch: int, step: int, task_idx: int, metrics: dict[str, float], elapsed_time: float):
        steps_per_sec = 1 / elapsed_time if elapsed_time > 0 else float('inf')

        table = Table(title=f"Epoch {epoch} | Step {step} | Task {task_idx}", show_header=True, header_style="bold magenta", expand=True)

        table.add_column("Loss (M/P/G)", justify="center")
        table.add_column("Acc", justify="center")
        table.add_column("τ", justify="center")
        table.add_column("PI", justify="center")
        table.add_column("Act % (L0/Mid/LN/Avg)", justify="center")
        table.add_column("Gate (Top10/All/Max)", justify="center")
        table.add_column("Proto Norm", justify="center")
        table.add_column("Seq (L/E)", justify="center")
        table.add_column("Speed (st/s)", justify="center")

        act_l0 = metrics.get('activation_rate_l0', 0.0) * 100
        act_mid = metrics.get('activation_rate_l_mid', 0.0) * 100
        act_ln = metrics.get('activation_rate_ln', 0.0) * 100
        act_avg = metrics.get('activation_rate_avg', 0.0) * 100
        tau = metrics.get('tau', 0.0)
        table.add_row(
            f"{metrics.get('main_loss', 0.0):.3f}/{metrics.get('proto_loss', 0.0):.4f}/{metrics.get('gate_loss', 0.0):.4f}",
            f"{metrics.get('token_acc', 0.0):.3f}",
            f"{tau:.3f}",
            f"{metrics.get('pi_score', 0.0):.3f}",
            f"{act_l0:.1f}/{act_mid:.1f}/{act_ln:.1f}/{act_avg:.1f}%",
            f"{metrics.get('top10_gate_mean', 0.0):.3f}/{metrics.get('avg_gate_val', 0.0):.3f}/{metrics.get('max_gate_val', 0.0):.3f}",
            f"{metrics.get('avg_proto_norm', 0.0):.2f}",
            f"{int(metrics.get('seq_len', 0))} / {metrics.get('seq_entropy', 0.0):.2f}",
            f"{steps_per_sec:.2f}"
        )

        self.console.print(table)

    def _create_grid_text(self, grid: torch.Tensor) -> Text:
        text = Text()
        h, w = grid.shape
        for r in range(h):
            for p in range(w):
                pixel = grid[r, p].item()
                color = self.ARC_COLORS[pixel] if 0 <= pixel < len(self.ARC_COLORS) else "white"
                text.append("■ ", style=color)
            if r < h - 1:
                text.append("\n")
        return text

    def visualize_evaluation_sample(self, input_grid: torch.Tensor, target_grid: torch.Tensor, pred_grid: torch.Tensor | None, pred_tokens: list[int] | None, step: int):
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

        if pred_tokens:
            self.console.print(f"[bold]Generated Token Stream ({len(pred_tokens)} tokens):[/bold]", highlight=False)
            self.console.print(str(pred_tokens))
        else:
            self.console.print("[bold]Generated Token Stream:[/bold] [red]N/A[/red]")

        self.console.print()

    def log_eval_summary(self, metrics: dict[str, float], step: int):
        eval_loss = metrics.get('eval_loss', 0.0)
        eval_grid_acc = metrics.get('eval_grid_acc', 0.0)
        total_count = int(metrics.get('total_count', 0))
        summary_text = f"Avg Loss: {eval_loss:.4f} | Grid Accuracy: {eval_grid_acc:.4f} on {total_count} samples."
        self.console.print(Padding(
            f"[bold yellow]===== EVALUATION COMPLETE (Step: {step}) =====\n"
            f"{summary_text}\n"
            f"=============================================[/bold yellow]", (1, 2)))
