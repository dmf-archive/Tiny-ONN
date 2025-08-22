import math

import torch
from rich.console import Console

from .config import Config


class Observer:
    ARC_COLORS = ["black", "blue", "red", "green", "yellow", "grey82", "magenta", "orange", "cyan", "white"]

    def __init__(self, console: Console, config: Config):
        self.console = console
        self.config = config

    def _to_grid(self, seq: torch.Tensor, h: int, w: int) -> list[list[int]]:
        return seq.view(h, w + 1)[:, :-1].tolist()

    def _format_grid_rows(self, grid: list[list[int]]) -> list[str]:
        rows = []
        h = min(len(grid), 15)
        w = min(len(grid[0]) if grid else 0, 15)

        for r in range(h):
            row_str = ""
            for p in range(w):
                pixel = grid[r][p]
                color = self.ARC_COLORS[pixel] if 0 <= pixel < len(self.ARC_COLORS) else "white"
                row_str += f"[{color}]█[/]"
            rows.append(row_str)
        return rows

    def log_step(self, epoch: int, step: int, losses: dict[str, float], metrics: dict[str, float], ips: float):
        log_str = (
            f"E:{epoch} S:{step} | L(M/G):{losses['main']:.2f}/{losses['gating']:.2f} | "
            f"Acc(Tok/Grid):{metrics.get('tok_acc', 0):.2f}/{metrics.get('grid_acc', 0):.2f} | "
            f"GateAcc(S/M):{metrics.get('smha_gate_acc', 0):.2f}/{metrics.get('moe_gate_acc', 0):.2f} | "
            f"K(S/M):{metrics.get('smha_avg_k', 0):.2f}/{metrics.get('moe_avg_k', 0):.2f} | "
            f"PI:{metrics.get('pi_score', 0):.2f} | {ips:.2f}it/s"
        )
        self.console.print(log_str)

    def log_eval_results(self, metrics: dict[str, float]):
        self.console.print(
            f"\n[bold yellow]Eval Results | Token Acc: {metrics['tok_acc']:.4f} | Grid Acc: {metrics['grid_acc']:.4f}[/bold yellow]\n"
        )

    def visualize_batch(self, input_ids: torch.Tensor, labels: torch.Tensor, generated_ids: torch.Tensor):
        for i in range(min(input_ids.shape[0], 4)):
            input_len = (labels[i] == -100).sum()

            input_seq = input_ids[i, :input_len]
            target_seq = labels[i, input_len:]
            pred_seq = generated_ids[i, input_len:]

            h_in = (input_seq == 10).sum().item() + 1
            w_in = (input_len // h_in) - 1

            h_out = int((target_seq != -100).sum().item() // (w_in + 1)) if w_in > -1 else 0

            try:
                input_grid = self._to_grid(input_seq, h_in, w_in)
                target_grid = self._to_grid(target_seq[target_seq != -100], h_out, w_in)
                pred_grid = self._to_grid(pred_seq[:len(target_seq[target_seq != -100])], h_out, w_in)
            except RuntimeError:
                self.console.print(f"[red]Error visualizing sample {i}, skipping.[/red]")
                continue

            input_rows = self._format_grid_rows(input_grid)
            target_rows = self._format_grid_rows(target_grid)
            pred_rows = self._format_grid_rows(pred_grid)

            self.console.print(f"\n--- Sample {i} ---")
            title_width = 15 * 2
            self.console.print(f"{'[bold]Input[/bold]':<{title_width}} {'[bold]Target[/bold]':<{title_width}} {'[bold]Prediction[/bold]':<{title_width}}")

            max_rows = max(len(input_rows), len(target_rows), len(pred_rows))
            for j in range(max_rows):
                ir = input_rows[j] if j < len(input_rows) else ""
                tr = target_rows[j] if j < len(target_rows) else ""
                pr = pred_rows[j] if j < len(pred_rows) else ""
                self.console.print(f"{ir:<{title_width}} {tr:<{title_width}} {pr:<{title_width}}")

def calculate_pi_score(config: Config, main_loss: float, avg_surprise: float, logits: torch.Tensor) -> float:
    mask = logits.argmax(-1) != -100
    if not mask.any(): return 0.0
    tau = torch.distributions.Categorical(logits=logits).entropy()[mask].mean().item()
    pi_score = math.exp(-config.pi_alpha * ((1 - config.pi_gamma) * (main_loss / (tau + 1e-9)) + config.pi_gamma * avg_surprise))
    return pi_score
