import torch
from rich.console import Console

from .bayesian_config import BayesianConfig


class BayesianObserver:
    ARC_COLORS = ["black", "blue", "red", "green", "yellow", "grey82", "magenta", "orange", "cyan", "white"]

    def __init__(self, console: Console, config: BayesianConfig):
        self.console = console
        self.config = config

    def _format_grid_rows(self, grid: torch.Tensor) -> list[str]:
        rows = []
        grid_list = grid.tolist()
        h = min(len(grid_list), 15)
        w = min(len(grid_list[0]) if grid_list else 0, 15)

        for r in range(h):
            row_str = ""
            for p in range(w):
                pixel = grid_list[r][p]
                color = self.ARC_COLORS[pixel] if 0 <= pixel < len(self.ARC_COLORS) else "white"
                row_str += f"[{color}]█[/]"
            rows.append(row_str)
        return rows

    def log_step(self, stats: dict, epoch: int):
        log_str = (
            f"E:{epoch} S:{stats['step']} | "
            f"Loss(W/μ):{stats['weighted_loss']:.3f}/{stats['mean_loss']:.3f} | "
            f"Acc:{stats['mean_acc']:.3f} | PI:{stats['pi_score']:.3f} | "
            f"σ(μ):{stats['sigma_mean']:.4f}"
        )
        self.console.print(log_str)

    def log_eval_results(self, metrics: dict[str, float]):
        self.console.print(
            f"\n[bold yellow]Eval Results | Grid Acc: {metrics['grid_acc']:.4f}[/bold yellow]\n"
        )

    def visualize_batch(self, input_grids: torch.Tensor, target_grids: torch.Tensor, pred_grids: torch.Tensor):
        for i in range(min(input_grids.shape[0], 4)):
            input_rows = self._format_grid_rows(input_grids[i])
            target_rows = self._format_grid_rows(target_grids[i])
            pred_rows = self._format_grid_rows(pred_grids[i])

            self.console.print(f"\n--- Sample {i} ---")
            title_width = 15 * 2
            self.console.print(f"{'[bold]Input[/bold]':<{title_width}} {'[bold]Target[/bold]':<{title_width}} {'[bold]Prediction[/bold]':<{title_width}}")

            max_rows = max(len(input_rows), len(target_rows), len(pred_rows))
            for j in range(max_rows):
                ir = input_rows[j] if j < len(input_rows) else ""
                tr = target_rows[j] if j < len(target_rows) else ""
                pr = pred_rows[j] if j < len(pred_rows) else ""
                self.console.print(f"{ir:<{title_width}} {tr:<{title_width}} {pr:<{title_width}}")
