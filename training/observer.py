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
        self.expert_history: list[dict[str, torch.Tensor]] = []
        self.timing_history: list[dict[str, float]] = []
        self.device = device
        self.max_history_steps = max_history_steps
        self.rng = default_rng()

    def _downsample(self):
        if len(self.metrics_history) > self.max_history_steps:
            steps_to_keep = {
                m["step"]
                for m in self.rng.choice(
                    self.metrics_history, self.max_history_steps, replace=False
                )
            }
            self.metrics_history = [
                m for m in self.metrics_history if m["step"] in steps_to_keep
            ]
            self.expert_history = [
                e for e in self.expert_history if e["global_step"] in steps_to_keep
            ]
            self.timing_history = [
                t for t in self.timing_history if t["step"] in steps_to_keep
            ]

    def log_metrics_and_expert_data(
        self, metrics: dict[str, Any], expert_data: dict[str, torch.Tensor], step: int
    ):
        metrics["step"] = step
        self.metrics_history.append(metrics)
        expert_data["global_step"] = torch.tensor(step, device=self.device)
        self.expert_history.append(expert_data)

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

    def plot_dashboards(self, step: int, model_config: Any):
        self._plot_metrics_dashboard(step)
        self._plot_expert_dashboard(step, model_config)
        self._plot_timing_dashboard(step)

    def _save_plot(self, fig: plt.Figure, filename: str):
        fig.savefig(self.plot_dir / f"{filename}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    def _plot_metrics_dashboard(self, step: int):
        df = pd.DataFrame(self.metrics_history)
        fig, axes = plt.subplots(4, 2, figsize=(20, 24), constrained_layout=True)
        fig.suptitle(f"Core Metrics at Step {step}", fontsize=20)

        plot_specs = [
            ("main_loss", "smk_loss", "Losses", 0, 0),
            ("main_acc", "gating_acc", "Accuracies", 0, 1),
            ("pi_score", None, "Predictive Integrity (PI)", 1, 0),
            ("surprise", None, "Surprise (Gradient Norm)", 1, 1),
            ("tau", None, "Model Uncertainty (Tau)", 2, 0),
            ("gating_kld", None, "Gating KLD (Optimal vs Selected)", 2, 1),
            ("avg_k", None, "Average Activated Experts per Token", 3, 0),
            ("global_avg_k", None, "Global Average Activated Experts", 3, 1),
        ]

        for y1, y2, title, r, c in plot_specs:
            ax = axes[r, c]
            sns.lineplot(
                data=df, x="step", y=y1, ax=ax, label=y1.replace("_", " ").title()
            )
            if y2 and y2 in df.columns:
                sns.lineplot(
                    data=df, x="step", y=y2, ax=ax, label=y2.replace("_", " ").title()
                )
            ax.set_title(title)
            ax.legend()

        self._save_plot(fig, "metrics_dashboard")

    def _plot_expert_dashboard(self, step: int, model_config: Any):
        fig, axs = plt.subplots(3, 2, figsize=(20, 27), constrained_layout=True)
        fig.suptitle(f"Expert Dashboard at Step {step}", fontsize=20)
        num_experts = getattr(model_config, "num_experts_per_layer", 1)
        num_layers = getattr(model_config, "num_hidden_layers", 1)
        total_experts = num_experts * num_layers

        steps_data = (
            torch.cat(
                [
                    torch.full_like(e["selected_experts"], e["global_step"].item())
                    for e in self.expert_history
                ]
            )
            .cpu()
            .numpy()
        )

        plot_data = {
            "selected": torch.cat([e["selected_experts"] for e in self.expert_history])
            .cpu()
            .numpy(),
            "optimal": torch.cat([e["optimal_experts"] for e in self.expert_history])
            .cpu()
            .numpy(),
        }

        for i, (key, title_suffix, color) in enumerate(
            [
                ("selected", "Selected (Top-K)", "C0"),
                ("optimal", "Optimal (Min-Surprise)", "C1"),
            ]
        ):
            ax_scatter = axs[0, i]
            ax_heatmap = axs[1, i]
            ax_zscore = axs[2, i]

            values = plot_data[key]
            ax_scatter.scatter(steps_data, values, s=1, alpha=0.3, c=color)
            ax_scatter.set_title(f"{title_suffix} Activation Scatter")
            ax_scatter.set_xlabel("Global Step")
            ax_scatter.set_ylabel("Global Expert ID")
            ax_scatter.set_ylim(-1, total_experts)
            ax_scatter.grid(True, which="both", linestyle="--", linewidth=0.5)

            layers, experts_in_layer = values // num_experts, values % num_experts
            heatmap, _, _ = np.histogram2d(
                layers,
                experts_in_layer,
                bins=[num_layers, num_experts],
                range=[[-0.5, num_layers - 0.5], [-0.5, num_experts - 0.5]],
            )
            sns.heatmap(heatmap.T, ax=ax_heatmap, cmap="viridis", cbar=True)
            ax_heatmap.set_title(f"Cumulative {title_suffix} Activation Heatmap")
            ax_heatmap.set_xlabel("Layer ID")
            ax_heatmap.set_ylabel("Expert ID in Layer")

            mean_act = heatmap.mean(axis=1, keepdims=True)
            std_act = heatmap.std(axis=1, keepdims=True)
            heatmap_z_score = (heatmap - mean_act) / (std_act + 1e-6)
            sns.heatmap(
                heatmap_z_score.T, ax=ax_zscore, cmap="coolwarm", center=0, cbar=True
            )
            ax_zscore.set_title(f"Layer-wise {title_suffix} Activation Z-Score")
            ax_zscore.set_xlabel("Layer ID")
            ax_zscore.set_ylabel("Layer ID")

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
