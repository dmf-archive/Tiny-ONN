import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from rich.console import Console
from sklearn.decomposition import PCA

from .config import Config
from .model import TinyOnnForArcReconstruction


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

    def visualize_expert_space(self, model: TinyOnnForArcReconstruction, step: int):
        Path("exp/tiny_onn_arc/expert_viz").mkdir(exist_ok=True)
        num_layers = model.config.num_hidden_layers
        fig, axes = plt.subplots(2, num_layers, figsize=(num_layers * 5, 10))
        fig.suptitle(f"Expert Space Visualization @ Step {step}", fontsize=16)

        for i in range(num_layers):
            # SMHA
            smha_gating = model.model.layers[i].smha_layer.gating_network
            smha_sim = smha_gating.sim_matrix.data.T.cpu().numpy()
            smha_gates = smha_gating.gates.data.cpu().numpy()
            
            pca_smha = PCA(n_components=2)
            smha_2d = pca_smha.fit_transform(smha_sim)
            
            axes[0, i].set_title(f"Layer {i} SMHA")
            sc_smha = axes[0, i].scatter(smha_2d[:, 0], smha_2d[:, 1], s=(-smha_gates + 1) * 20, c=smha_gates, cmap="viridis_r")
            fig.colorbar(sc_smha, ax=axes[0, i], label="Gate Value (lower is easier)")

            # MoE
            moe_gating = model.model.layers[i].moe_layer.gating_network
            moe_sim = moe_gating.sim_matrix.data.T.cpu().numpy()
            moe_gates = moe_gating.gates.data.cpu().numpy()
            
            pca_moe = PCA(n_components=2)
            moe_2d = pca_moe.fit_transform(moe_sim)
            
            axes[1, i].set_title(f"Layer {i} MoE")
            sc_moe = axes[1, i].scatter(moe_2d[:, 0], moe_2d[:, 1], s=(-moe_gates + 1) * 20, c=moe_gates, cmap="viridis_r")
            fig.colorbar(sc_moe, ax=axes[1, i], label="Gate Value (lower is easier)")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"exp/tiny_onn_arc/expert_viz/expert_space_step_{step}.png")
        plt.close(fig)

