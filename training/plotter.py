import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class LivePlotter:
    def __init__(self, plot_dir: str):
        self.plot_dir = Path(plot_dir)
        os.makedirs(self.plot_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def _save_plot(self, fig, filename: str):
        fig.savefig(self.plot_dir / f"{filename}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    def plot_metrics_dashboard(self, history: list[dict[str, Any]], step: int):
        if not history:
            return

        df = pd.DataFrame(history)
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f"Training Metrics at Step {step}", fontsize=16)

        sns.lineplot(data=df, x="step", y="main_loss", ax=axes[0, 0], label="Main Loss")
        axes[0, 0].set_title("Main Task Loss")

        sns.lineplot(
            data=df, x="step", y="router_loss", ax=axes[0, 1], label="Router Loss"
        )
        axes[0, 1].set_title("Router Meta-Loss")

        sns.lineplot(
            data=df, x="step", y="gating_acc", ax=axes[1, 0], label="Gating Accuracy"
        )
        axes[1, 0].set_title("Gating Accuracy")

        sns.lineplot(
            data=df, x="step", y="surprise", ax=axes[1, 1], label="Surprise (Grad Norm)"
        )
        axes[1, 1].set_title("Average Surprise")

        sns.lineplot(data=df, x="step", y="pi_score", ax=axes[2, 0], label="PI Score")
        axes[2, 0].set_title("Predictive Integrity (PI) Score")

        sns.lineplot(
            data=df, x="step", y="tau", ax=axes[2, 1], label="Tau (Logits Entropy)"
        )
        axes[2, 1].set_title("Model Uncertainty (Tau)")

        for ax_row in axes:
            for ax in ax_row:
                ax.legend()
                ax.tick_params(axis="x", rotation=45)

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        self._save_plot(fig, f"metrics_dashboard_step_{step}")

    def plot_expert_dashboard(self, expert_data: dict, step: int):
        # This is a placeholder for the more complex expert analysis plots
        # A real implementation would require more detailed data about expert activations
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "Expert Dashboard Placeholder",
            ha="center",
            va="center",
            fontsize=16,
        )
        ax.set_title(f"Expert Analysis at Step {step}")
        self._save_plot(fig, f"expert_dashboard_step_{step}")
