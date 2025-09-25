import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from sklearn.decomposition import PCA

from .config import TrainConfig


class Observer:
    ARC_COLORS: list[str] = [
        "black", "blue", "red", "green", "yellow",
        "grey", "magenta", "orange", "cyan", "brown"
    ]

    def __init__(self, console: Console, config: TrainConfig):
        self.console = console
        self.config = config
        self.vis_dir = Path(__file__).parent / "pic"
        self.vis_dir.mkdir(exist_ok=True)
        self.max_vis_files = 5

    def log_step(self, epoch: int, step: int, task_idx: int, metrics: dict[str, float], elapsed_time: float):
        steps_per_sec = 1 / elapsed_time if elapsed_time > 0 else float('inf')

        table = Table(title=f"Epoch {epoch} | Step {step} | Task {task_idx}", show_header=True, header_style="bold magenta", expand=True)

        table.add_column("Loss (Main/Proto/Gate)", justify="center")
        table.add_column("Acc (Tok/EMA)", justify="center")
        table.add_column("τ", justify="center")
        table.add_column("PI", justify="center")
        table.add_column("Act % (L0/Mid/LN/Avg)", justify="center")
        table.add_column("Gate (Top10/All/Max)", justify="center")
        table.add_column("Seq (L/E)", justify="center")
        table.add_column("Speed (st/s)", justify="center")

        act_l0 = metrics.get('activation_rate_l0', 0.0) * 100
        act_mid = metrics.get('activation_rate_l_mid', 0.0) * 100
        act_ln = metrics.get('activation_rate_ln', 0.0) * 100
        act_avg = metrics.get('activation_rate_avg', 0.0) * 100
        tau = metrics.get('tau', 0.0)
        table.add_row(
            f"{metrics.get('main_loss', 0.0):.3f}/{metrics.get('proto_loss', 0.0):.3f}/{metrics.get('gate_loss', 0.0):.4f}",
            f"{metrics.get('token_acc', 0.0):.3f}/{metrics.get('ema_acc', 0.0):.3f}",
            f"{tau:.3f}",
            f"{metrics.get('pi_score', 0.0):.3f}",
            f"{act_l0:.1f}/{act_mid:.1f}/{act_ln:.1f}/{act_avg:.1f}%",
            f"{metrics.get('raw_top10_gate', 0.0):.3f}/{metrics.get('raw_avg_gate', 0.0):.3f}/{metrics.get('raw_max_gate', 0.0):.3f}",
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

    def visualize_saps_clusters(self, saps_data_per_block: list[dict[str, dict]], global_step: int):
        num_layers = len(saps_data_per_block)
        if num_layers == 0:
            return

        proto_names = saps_data_per_block[0].keys()
        fig, axes = plt.subplots(num_layers, len(proto_names), figsize=(20, 5 * num_layers), squeeze=False)

        fig.suptitle(f"SAPS Prototype Distribution @ Step {global_step}", fontsize=16)

        color_map = {"good": "green", "bad": "red", "neutral": "grey"}

        for i in range(num_layers):
            for j, name in enumerate(proto_names):
                ax = axes[i, j]
                block_data = saps_data_per_block[i]

                if name not in block_data:
                    ax.text(0.5, 0.5, "N/A", ha='center', va='center')
                    ax.set_title(f"Block {i} - {name}")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                protos = block_data[name]["protos"].detach().cpu().to(torch.float32).numpy()
                statuses = block_data[name]["statuses"]

                if protos.shape[0] == 0:
                    ax.text(0.5, 0.5, "No Actives", ha='center', va='center')
                else:
                    pca = PCA(n_components=2)
                    protos_2d = pca.fit_transform(protos)
                    colors = [color_map.get(s, "black") for s in statuses]
                    ax.scatter(protos_2d[:, 0], protos_2d[:, 1], c=colors, alpha=0.7, s=10)

                ax.set_title(f"Block {i} - {name}")
                ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = self.vis_dir / f"saps_step_{global_step}.png"
        plt.savefig(save_path)
        plt.close()

        # Rolling update for visualization files
        vis_files = sorted(self.vis_dir.glob("*.png"), key=os.path.getmtime)
        if len(vis_files) > self.max_vis_files:
            os.remove(vis_files[0])
