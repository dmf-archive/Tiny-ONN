from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from numpy.random import default_rng
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")


class UnifiedObserver:
    def __init__(
        self, output_dir: Path, device: torch.device, max_history_steps: int = 1000
    ):
        self.log_dir = output_dir / "logs"
        self.plot_dir = output_dir / "img"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        sns.set_theme(style="whitegrid")
        self.metrics_history: list[dict[str, Any]] = []
        self.expert_history: list[dict[str, Any]] = []
        self.timing_history: list[dict[str, float]] = []
        self.device = device
        self.max_history_steps = max_history_steps
        self.rng = default_rng()

    def _downsample(self):
        if len(self.metrics_history) > self.max_history_steps:
            indices_to_keep = self.rng.choice(
                len(self.metrics_history), self.max_history_steps, replace=False
            )
            indices_to_keep_sorted = sorted(indices_to_keep)
            self.metrics_history = [self.metrics_history[i] for i in indices_to_keep_sorted]
            self.expert_history = [self.expert_history[i] for i in indices_to_keep_sorted]
            self.timing_history = [self.timing_history[i] for i in indices_to_keep_sorted]

    def log_metrics_and_expert_data(
        self,
        metrics: dict[str, Any],
        expert_data: dict[str, torch.Tensor],
        step: int,
    ):
        metrics["step"] = step
        self.metrics_history.append(metrics)

        processed_expert_data = {
            "global_step": step,
            "activated_scores": expert_data["activated_scores"].cpu().half(),
            "surprise_matrix": expert_data["surprise_matrix"].cpu().half(),
        }
        self.expert_history.append(processed_expert_data)

        for key, value in metrics.items():
            if isinstance(value, int | float):
                self.writer.add_scalar(f"Metrics/{key}", value, step)
        self._downsample()

    def log_timing(self, timing_data: dict[str, float], step: int):
        timing_data["step"] = step
        self.timing_history.append(timing_data)
        for key, value in timing_data.items():
            if key != "step":
                self.writer.add_scalar(f"Timings/{key}", value, step)

    def plot_all_dashboards(self, step: int, model_config: Any):
        self._plot_metrics_dashboard(step)
        self._plot_expert_dashboard(step, model_config)
        self._plot_timing_dashboard(step)

    def _save_plot(self, fig: plt.Figure, filename: str):
        fig.savefig(self.plot_dir / f"{filename}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    def _plot_metrics_dashboard(self, step: int):
        if not self.metrics_history:
            return
        df = pd.DataFrame(self.metrics_history)
        fig, axes = plt.subplots(4, 2, figsize=(20, 24), constrained_layout=True)
        fig.suptitle(f"Core Metrics at Step {step}", fontsize=20)

        plot_specs = [
            ("Losses", ["main_loss", "smk_loss", "balance_loss"], 0, 0),
            ("Accuracies", ["main_acc", "gating_acc"], 0, 1),
            ("Predictive Integrity (PI)", ["pi_score"], 1, 0),
            ("Surprise (Gradient Norm)", ["surprise"], 1, 1),
            ("Model Uncertainty (Tau)", ["tau"], 2, 0),
            ("Gating KL Divergence", ["gating_kld"], 2, 1),
            ("Avg Activated Experts / Token", ["avg_k"], 3, 0),
            ("Global Avg Activated Experts", ["global_avg_k"], 3, 1),
        ]

        for title, y_keys, r, c in plot_specs:
            ax = axes[r, c]
            for key in y_keys:
                if key in df.columns:
                    sns.lineplot(data=df, x="step", y=key, ax=ax, label=key)
            ax.set_title(title)
            ax.legend()
        self._save_plot(fig, "metrics_dashboard")

    def _plot_expert_dashboard(self, step: int, model_config: Any):
        if not self.expert_history:
            return

        num_layers = model_config.num_hidden_layers
        num_experts = model_config.num_experts_per_layer

        all_scores = torch.cat([e["activated_scores"].float() for e in self.expert_history])
        all_surprises = torch.cat([e["surprise_matrix"].float() for e in self.expert_history])
        all_steps = np.concatenate([np.full(e["activated_scores"].shape[0], e["global_step"]) for e in self.expert_history])

        fig, axs = plt.subplots(3, 2, figsize=(24, 30), constrained_layout=True)
        fig.suptitle(f"Expert Dashboard at Step {step}", fontsize=20)

        plot_configs = {
            0: ("Top-K Activation", all_scores, "viridis"),
            1: ("Surprise (Min-K)", all_surprises, "magma"),
        }

        for col, (title, data_tensor, cmap) in plot_configs.items():
            if "Surprise" in title:
                data_tensor = torch.where(torch.isfinite(data_tensor), data_tensor, torch.tensor(float('nan')))

            ax_scatter = axs[0, col]
            token_indices, layer_indices, expert_indices = torch.where(data_tensor > 0 if "Activation" in title else torch.isfinite(data_tensor))
            steps_for_scatter = all_steps[token_indices.numpy()]
            global_expert_ids = layer_indices.numpy() * num_experts + expert_indices.numpy()
            values = data_tensor[token_indices, layer_indices, expert_indices].numpy()

            scatter = ax_scatter.scatter(steps_for_scatter, global_expert_ids, c=values, cmap=cmap, s=5, alpha=0.5)
            ax_scatter.set_title(f"{title} - Scatter Heatmap")
            ax_scatter.set_xlabel("Global Step")
            ax_scatter.set_ylabel("Global Expert ID")
            ax_scatter.set_ylim(-1, num_layers * num_experts)
            plt.colorbar(scatter, ax=ax_scatter, label="Value")

            ax_heatmap_cum = axs[1, col]
            cumulative_data = torch.nansum(data_tensor, dim=0).numpy()
            sns.heatmap(cumulative_data, ax=ax_heatmap_cum, cmap=cmap, cbar=True)
            ax_heatmap_cum.set_title(f"{title} - Cumulative Heatmap")
            ax_heatmap_cum.set_ylabel("Layer ID")
            ax_heatmap_cum.set_xlabel("Expert ID in Layer")

            ax_heatmap_zscore = axs[2, col]
            z_scores = torch.zeros_like(data_tensor)
            for l in range(num_layers):
                layer_data = data_tensor[:, l, :]
                finite_mask = torch.isfinite(layer_data)
                if finite_mask.any():
                    flat_layer_data = layer_data[finite_mask]
                    if flat_layer_data.numel() > 1:
                         mean = torch.mean(flat_layer_data)
                         std = torch.std(flat_layer_data)
                         if std > 1e-6:
                            z_scores[:, l, :] = torch.where(finite_mask, (layer_data - mean) / std, 0)

            cumulative_zscore = torch.nansum(z_scores, dim=0).numpy()
            sns.heatmap(cumulative_zscore, ax=ax_heatmap_zscore, cmap="coolwarm", center=0, cbar=True)
            ax_heatmap_zscore.set_title(f"{title} - Z-score Normalized Heatmap")
            ax_heatmap_zscore.set_ylabel("Layer ID")
            ax_heatmap_zscore.set_xlabel("Expert ID in Layer")

        self._save_plot(fig, "expert_dashboard")


    def _plot_timing_dashboard(self, step: int):
        if not self.timing_history:
            return
        df = pd.DataFrame(self.timing_history)
        avg_timings = df.drop(columns="step").mean()
        fig, ax = plt.subplots(figsize=(12, 8))
        avg_timings.plot(kind="barh", ax=ax)
        ax.set_title(f"Average Step Time by Phase at Step {step}")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Training Phase")
        for index, value in enumerate(avg_timings):
            ax.text(value, index, f"{value:.4f}s", va="center")
        self._save_plot(fig, "timing_dashboard")

    def close(self):
        self.writer.close()
