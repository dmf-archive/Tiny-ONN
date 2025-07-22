from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedModel


class TensorBoardLogger:
    def __init__(self, log_dir: Path):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, metrics: dict[str, float], step: int):
        for key, value in metrics.items():
            self.writer.add_scalar(f"metrics/{key}", value, step)

    def close(self):
        self.writer.close()


class MatplotlibVisualizer:
    def __init__(self, img_dir: Path, model_name: str):
        self.img_dir = img_dir
        self.model_name = model_name
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.colors = {
            "train_main": "#1f77b4",
            "train_router": "#ff7f0e",
            "eval_main": "#2ca02c",
            "eval_router": "#d62728",
        }
        plt.style.use("seaborn-v0_8-whitegrid")

    def update_plots(
        self,
        metrics_cache: dict[str, list],
        expert_data_cache: dict[str, np.ndarray],
        global_step: int,
        model: PreTrainedModel,
    ):
        self._plot_core_metrics(metrics_cache, global_step)
        if expert_data_cache:
            self._plot_expert_dashboard(expert_data_cache, global_step, model)

    def _plot_core_metrics(self, cache: dict[str, list], step: int):
        fig, axs = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle(f"Core Metrics for {self.model_name} at Step {step}", fontsize=16)

        # Loss
        axs[0, 0].plot(
            cache["main_loss"],
            label="Train Main Loss",
            color=self.colors["train_main"],
            alpha=0.7,
        )
        axs[0, 0].plot(
            cache["router_loss"],
            label="Train Router Loss",
            color=self.colors["train_router"],
            alpha=0.7,
        )
        axs[0, 0].set_title("Loss")
        axs[0, 0].legend()

        # Accuracy
        axs[0, 1].plot(
            cache["main_acc"],
            label="Train Main Acc",
            color=self.colors["train_main"],
            alpha=0.7,
        )
        axs[0, 1].plot(
            cache["gating_acc"],
            label="Train Gating Acc",
            color=self.colors["train_router"],
            alpha=0.7,
        )
        axs[0, 1].set_title("Accuracy")
        axs[0, 1].legend()

        # PI Score
        axs[1, 0].plot(cache["pi_score"], label="PI Score", color="#8c564b", alpha=0.7)
        axs[1, 0].set_title("PI Score")

        # Surprise
        axs[1, 1].plot(cache["surprise"], label="Surprise", color="#e377c2", alpha=0.7)
        axs[1, 1].set_title("Surprise")

        # Tau
        axs[2, 0].plot(cache["tau"], label="Tau", color="#7f7f7f", alpha=0.7)
        axs[2, 0].set_title("Tau")

        fig.delaxes(axs[2, 1])
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(self.img_dir / "core_metrics_latest.png")
        plt.close(fig)

    def _plot_expert_dashboard(
        self, cache: dict[str, np.ndarray], step: int, model: PreTrainedModel
    ):
        fig, axs = plt.subplots(2, 2, figsize=(20, 18))
        fig.suptitle(f"Expert Activation Dashboard at Step {step}", fontsize=16)

        if hasattr(model.config, "num_experts") and hasattr(
            model.config, "num_hidden_layers"
        ):
            num_experts = getattr(model.config, "num_experts", 1)
        else:
            num_experts = getattr(model.config, "num_experts", 1)

        # Top-K Activation Scatter
        top_k_data = cache["selected_experts"]
        axs[0, 0].scatter(top_k_data[:, 0], top_k_data[:, 1], s=1, alpha=0.5)
        axs[0, 0].set_title("Top-K Activation Scatter")
        axs[0, 0].set_xlabel("Global Step")
        axs[0, 0].set_ylabel("Expert ID")
        axs[0, 0].set_ylim(-1, num_experts)

        # Surprise Min-K Activation Scatter
        min_k_data = cache["optimal_experts"]
        axs[0, 1].scatter(min_k_data[:, 0], min_k_data[:, 1], s=1, alpha=0.5)
        axs[0, 1].set_title("Surprise Min-K Activation Scatter")
        axs[0, 1].set_xlabel("Global Step")
        axs[0, 1].set_ylabel("Expert ID")
        axs[0, 1].set_ylim(-1, num_experts)

        # Top-K Activation Heatmap
        top_k_heatmap, _, _ = np.histogram2d(
            top_k_data[:, 0], top_k_data[:, 1], bins=[10, num_experts]
        )
        sns.heatmap(top_k_heatmap.T, ax=axs[1, 0], cmap="viridis")
        axs[1, 0].set_title("Top-K Activation Heatmap")
        axs[1, 0].set_xlabel("Step Bins")
        axs[1, 0].set_ylabel("Expert ID")

        # Surprise Min-K Heatmap
        min_k_heatmap, _, _ = np.histogram2d(
            min_k_data[:, 0], min_k_data[:, 1], bins=[10, num_experts]
        )
        sns.heatmap(min_k_heatmap.T, ax=axs[1, 1], cmap="viridis")
        axs[1, 1].set_title("Surprise Min-K Heatmap")
        axs[1, 1].set_xlabel("Step Bins")
        axs[1, 1].set_ylabel("Expert ID")

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(self.img_dir / "expert_dashboard_latest.png")
        plt.close(fig)
