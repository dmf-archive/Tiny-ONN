from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import default_rng

matplotlib.use("Agg")


class LivePlotter:
    def __init__(self, plot_dir: str | Path):
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def _save_plot(self, fig, filename: str):
        fig.savefig(self.plot_dir / f"{filename}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    def plot_metrics_dashboard(self, history: list[dict[str, Any]], step: int):
        if not history:
            return

        df = pd.DataFrame(history)
        if "step" not in df.columns:
            df["step"] = range(len(df))

        df_train = df[df["type"] == "train"]
        df_eval = df[df["type"] == "eval"]

        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle(f"Core Metrics at Step {step}", fontsize=16)

        # Plot Losses
        sns.lineplot(data=df_train, x="step", y="main_loss", ax=axes[0, 0], label="Train Main Loss")
        sns.lineplot(data=df_train, x="step", y="router_loss", ax=axes[0, 0], label="Train Router Loss")
        if not df_eval.empty:
            sns.lineplot(data=df_eval, x="step", y="main_loss", ax=axes[0, 0], label="Eval Main Loss", linestyle='--')
        axes[0, 0].set_title("Losses")
        axes[0, 0].legend()

        # Plot Accuracies
        sns.lineplot(data=df_train, x="step", y="main_acc", ax=axes[0, 1], label="Train Main Acc")
        sns.lineplot(data=df_train, x="step", y="gating_acc", ax=axes[0, 1], label="Train Gating Acc")
        if not df_eval.empty:
            sns.lineplot(data=df_eval, x="step", y="main_acc", ax=axes[0, 1], label="Eval Main Acc", linestyle='--')
        axes[0, 1].set_title("Accuracies")
        axes[0, 1].legend()

        # Other metrics
        sns.lineplot(data=df_train, x="step", y="pi_score", ax=axes[1, 0], label="PI Score")
        axes[1, 0].set_title("PI Score")
        sns.lineplot(data=df_train, x="step", y="surprise", ax=axes[1, 1], label="Surprise")
        axes[1, 1].set_title("Surprise")
        sns.lineplot(data=df_train, x="step", y="tau", ax=axes[2, 0], label="Tau")
        axes[2, 0].set_title("Tau")
        if "avg_k" in df_train.columns:
            sns.lineplot(data=df_train, x="step", y="avg_k", ax=axes[2, 1], label="Avg K")
            axes[2, 1].set_title("Average K per Token")
        else:
            fig.delaxes(axes[2, 1])

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        self._save_plot(fig, "core_metrics_latest")

    def plot_expert_dashboard(
        self, expert_data: dict[str, np.ndarray], step: int, model_config: Any
    ):
        if not expert_data:
            return

        fig, axs = plt.subplots(2, 2, figsize=(20, 18))
        fig.suptitle(f"Expert Activation Dashboard at Step {step}", fontsize=16)

        num_experts = getattr(model_config, "num_experts_per_layer", 1)
        num_layers = getattr(model_config, "num_hidden_layers", 1)
        total_experts = num_experts * num_layers

        max_points = 50000

        # Top-K Activation Scatter
        if "selected_experts_steps" in expert_data and "selected_experts_values" in expert_data:
            steps = expert_data["selected_experts_steps"]
            values = expert_data["selected_experts_values"]
            if len(steps) > max_points:
                rng = default_rng()
                indices = rng.choice(len(steps), max_points, replace=False)
                steps = steps[indices]
                values = values[indices]
            axs[0, 0].scatter(steps, values, s=0.5, alpha=0.2)
        axs[0, 0].set_title("Top-K Activation Scatter")
        axs[0, 0].set_xlabel("Global Step")
        axs[0, 0].set_ylabel("Global Expert ID")
        axs[0, 0].set_ylim(-1, total_experts)

        # Surprise Min-K Activation Scatter
        if "optimal_experts_steps" in expert_data and "optimal_experts_values" in expert_data:
            steps = expert_data["optimal_experts_steps"]
            values = expert_data["optimal_experts_values"]
            if len(steps) > max_points:
                rng = default_rng()
                indices = rng.choice(len(steps), max_points, replace=False)
                steps = steps[indices]
                values = values[indices]
            axs[0, 1].scatter(steps, values, s=0.5, alpha=0.2, c='orange')
        axs[0, 1].set_title("Surprise Min-K Activation Scatter")
        axs[0, 1].set_xlabel("Global Step")
        axs[0, 1].set_ylabel("Global Expert ID")
        axs[0, 1].set_ylim(-1, total_experts)

        # Cumulative Top-K Activation Heatmap
        if "selected_experts_values" in expert_data:
            top_k_data = expert_data["selected_experts_values"]
            layers = top_k_data // num_experts
            experts_in_layer = top_k_data % num_experts
            heatmap, _, _ = np.histogram2d(
                layers, experts_in_layer, bins=[num_layers, num_experts]
            )
            sns.heatmap(heatmap, ax=axs[1, 0], cmap="viridis")
        axs[1, 0].set_title("Cumulative Top-K Activation Heatmap")
        axs[1, 0].set_xlabel("Expert ID in Layer")
        axs[1, 0].set_ylabel("Layer ID")

        # Cumulative Surprise Min-K Heatmap
        if "optimal_experts_values" in expert_data:
            min_k_data = expert_data["optimal_experts_values"]
            layers = min_k_data // num_experts
            experts_in_layer = min_k_data % num_experts
            heatmap, _, _ = np.histogram2d(
                layers, experts_in_layer, bins=[num_layers, num_experts]
            )
            sns.heatmap(heatmap, ax=axs[1, 1], cmap="viridis")
        axs[1, 1].set_title("Cumulative Surprise Min-K Heatmap")
        axs[1, 1].set_xlabel("Expert ID in Layer")
        axs[1, 1].set_ylabel("Layer ID")

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        self._save_plot(fig, "expert_dashboard_latest")
