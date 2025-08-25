import torch
import torch.nn.functional as F
from rich.console import Console
from rich.columns import Columns
from rich.text import Text
from rich.padding import Padding

from .config import Config
from .tokenizer import ArcTokenizer


class Observer:
    ARC_COLORS = ArcTokenizer.ARC_COLORS

    def __init__(self, console: Console, config: Config):
        self.console = console
        self.config = config

    def log_step(self, epoch: int, step: int, losses: dict[str, float], metrics: dict[str, float], ips: float):
        log_str = (
            f"E:{epoch:03d} S:{step:06d} | "
            f"Loss(M/C/G):{losses.get('main', 0):.2f}/{losses.get('consistency', 0):.2f}/{losses.get('gating', 0):.2f} | "
            f"Acc(T/G):{metrics.get('token_acc', 0):.2f}/{metrics.get('grid_acc', 0):.2f} | "
            f"PI:{metrics.get('pi_score', 0):.2f} (S:{metrics.get('surprise', 0):.2f}/T:{metrics.get('tau', 0):.2f}) | "
            f"K(Att/MoE):{metrics.get('smha_avg_k', 0):.1f}/{metrics.get('moe_avg_k', 0):.1f} | "
            f"{ips:.1f}it/s"
        )
        self.console.print(log_str)

    def _create_grid_text(self, grid: torch.Tensor, title: str) -> Text:
        text = Text(f"{title}\n", style="bold")
        h, w = grid.shape
        for r in range(h):
            for p in range(w):
                pixel = grid[r, p].item()
                color = self.ARC_COLORS[pixel] if 0 <= pixel < len(self.ARC_COLORS) else "white"
                text.append("■ ", style=color)
            text.append("\n")
        return text

    def visualize_evaluation_sample(self, input_grid: torch.Tensor, target_grid: torch.Tensor, pred_grid: torch.Tensor, sample_idx: int):
        self.console.print()
        self.console.print(f"--- Evaluation Sample {sample_idx} ---", style="bold yellow")
        
        max_h = max(input_grid.shape[0], target_grid.shape[0], pred_grid.shape[0])
        max_w = max(input_grid.shape[1], target_grid.shape[1], pred_grid.shape[1])

        def pad(grid: torch.Tensor) -> torch.Tensor:
            pad_h = max_h - grid.shape[0]
            pad_w = max_w - grid.shape[1]
            return F.pad(grid, (0, pad_w, 0, pad_h), "constant", 0)

        input_padded = pad(input_grid)
        target_padded = pad(target_grid)
        pred_padded = pad(pred_grid)

        input_text = self._create_grid_text(input_padded, "Input")
        target_text = self._create_grid_text(target_padded, "Target")
        pred_text = self._create_grid_text(pred_padded, "Best Prediction (via AugScore)")

        self.console.print(Columns([input_text, target_text, pred_text], expand=True, equal=True))
        self.console.print()

    def log_eval_summary(self, metrics: dict[str, float], step: int):
        summary_text = (
            f"AugScore Grid Accuracy: {metrics.get('aug_score_grid_acc', 0):.4f} "
            f"({int(metrics.get('correct_count', 0))}/{int(metrics.get('total_count', 0))} correct)"
        )
        self.console.print(Padding(f"[bold yellow]===== EVALUATION COMPLETE (Step: {step}) =====\n"
                                   f"{summary_text}\n"
                                   f"=============================================[/bold yellow]", (1, 2)))
