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

from tiny_onn.modular import TinyOnnForCausalLM

matplotlib.use("Agg")


class UnifiedObserver:
    def __init__(self, output_dir: Path, device: torch.device):
        self.log_dir = output_dir / "logs"
        self.plot_dir = output_dir / "img"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        sns.set_theme(style="whitegrid")

        self.metrics_history: list[dict[str, Any]] = []
        self.expert_data: dict[str, torch.Tensor] = {
            "selected_experts_steps": torch.tensor([], device=device, dtype=torch.long),
            "selected_experts_values": torch.tensor([], device=device, dtype=torch.long),
            "optimal_experts_steps": torch.tensor([], device=device, dtype=torch.long),
            "optimal_experts_values": torch.tensor([], device=device, dtype=torch.long),
        }
        self.device = device
        self.max_history_size = 50000

    def log_metrics(self, metrics: dict[str, Any], step: int):
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[::2]
        self.metrics_history.append(metrics)

        for key, value in metrics.items():
            if isinstance(value, int | float):
                self.writer.add_scalar(f"metrics/{key}", value, step)

    def update_expert_data(
        self,
        optimal_indices: torch.Tensor,
        all_router_logits: list[torch.Tensor],
        model: TinyOnnForCausalLM,
        global_step: int,
        routing_mask_list: list[torch.Tensor],
    ):
        num_experts_per_layer = model.config.num_experts_per_layer

        concatenated_routing_mask = torch.cat([m.view(-1, m.shape[-1]) for m in routing_mask_list])

        token_to_layer_map = torch.zeros(concatenated_routing_mask.shape[0], dtype=torch.long, device=self.device)
        current_offset = 0
        for i, r in enumerate(all_router_logits):
            num_tokens = r.view(-1, r.shape[-1]).shape[0]
            token_to_layer_map[current_offset:current_offset + num_tokens] = i
            current_offset += num_tokens

        routing_mask = concatenated_routing_mask
        token_indices, expert_indices = torch.where(routing_mask)

        if token_indices.numel() == 0:
            global_selected_experts = torch.tensor([], device=self.device, dtype=torch.long)
        else:
            activated_layers = token_to_layer_map[token_indices]
            global_selected_experts = expert_indices + activated_layers * num_experts_per_layer

        for key in ["selected_experts_values", "selected_experts_steps", "optimal_experts_values", "optimal_experts_steps"]:
            if self.expert_data[key].numel() > self.max_history_size:
                self.expert_data[key] = self.expert_data[key][::2]

        self.expert_data["selected_experts_values"] = torch.cat([self.expert_data["selected_experts_values"], global_selected_experts])
        steps = torch.full_like(global_selected_experts, global_step, device=self.device)
        self.expert_data["selected_experts_steps"] = torch.cat([self.expert_data["selected_experts_steps"], steps])

        optimal_layers = token_to_layer_map
        global_optimal_experts = optimal_indices + optimal_layers * num_experts_per_layer

        self.expert_data["optimal_experts_values"] = torch.cat([self.expert_data["optimal_experts_values"], global_optimal_experts])
        steps_opt = torch.full_like(global_optimal_experts, global_step, device=self.device)
        self.expert_data["optimal_experts_steps"] = torch.cat([self.expert_data["optimal_experts_steps"], steps_opt])


    def plot_dashboards(self, step: int, model_config: Any):
        self._plot_metrics_dashboard(step)
        self._plot_expert_dashboard(step, model_config)

    def _save_plot(self, fig, filename: str):
        fig.savefig(self.plot_dir / f"{filename}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    def _plot_metrics_dashboard(self, step: int):
        if not self.metrics_history:
            return

        df = pd.DataFrame(self.metrics_history)
        df_train = df[df["type"] == "train"]
        df_eval = df[df["type"] == "eval"]

        fig, axes = plt.subplots(4, 2, figsize=(20, 24))
        fig.suptitle(f"Core Metrics at Step {step}", fontsize=16)

        sns.lineplot(data=df_train, x="step", y="main_loss", ax=axes[0, 0], label="Train Main Loss")
        sns.lineplot(data=df_train, x="step", y="smk_loss", ax=axes[0, 0], label="Train SMK Loss")
        if not df_eval.empty:
            sns.lineplot(data=df_eval, x="step", y="main_loss", ax=axes[0, 0], label="Eval Main Loss", linestyle='--')
        axes[0, 0].set_title("Losses")
        axes[0, 0].legend()

        sns.lineplot(data=df_train, x="step", y="main_acc", ax=axes[0, 1], label="Train Main Acc")
        sns.lineplot(data=df_train, x="step", y="gating_acc", ax=axes[0, 1], label="Train Gating Acc")
        if not df_eval.empty:
            sns.lineplot(data=df_eval, x="step", y="main_acc", ax=axes[0, 1], label="Eval Main Acc", linestyle='--')
        axes[0, 1].set_title("Accuracies")
        axes[0, 1].legend()

        sns.lineplot(data=df_train, x="step", y="pi_score", ax=axes[1, 0], label="PI Score")
        axes[1, 0].set_title("PI Score")
        sns.lineplot(data=df_train, x="step", y="surprise", ax=axes[1, 1], label="Surprise")
        axes[1, 1].set_title("Surprise")
        sns.lineplot(data=df_train, x="step", y="tau", ax=axes[2, 0], label="Tau")
        axes[2, 0].set_title("Tau")

        fig.delaxes(axes[2, 1])

        sns.lineplot(data=df_train, x="step", y="avg_k", ax=axes[3, 0], label="Avg K per Token")
        axes[3, 0].set_title("Average K per Token")
        axes[3, 0].legend()

        if "global_avg_k" in df_train.columns:
            sns.lineplot(data=df_train, x="step", y="global_avg_k", ax=axes[3, 1], label="Global Avg K")
            axes[3, 1].set_title("Global Average K")
            axes[3, 1].legend()

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        self._save_plot(fig, "core_metrics_latest")

    def _plot_expert_dashboard(self, step: int, model_config: Any):
        expert_data_np = {k: v.cpu().numpy() for k, v in self.expert_data.items()}

        if not any(val.size > 0 for val in expert_data_np.values()):
            return

        fig, axs = plt.subplots(3, 2, figsize=(20, 27))
        fig.suptitle(f"Expert Activation Dashboard at Step {step}", fontsize=16)

        num_experts = getattr(model_config, "num_experts_per_layer", 1)
        num_layers = getattr(model_config, "num_hidden_layers", 1)
        total_experts = num_experts * num_layers
        max_points = 50000

        for ax, key_suffix, title, color in [
            (axs[0, 0], "selected", "Top-K", None),
            (axs[0, 1], "optimal", "Surprise Min-K", "orange"),
        ]:
            steps_key = f"{key_suffix}_experts_steps"
            values_key = f"{key_suffix}_experts_values"
            if steps_key in expert_data_np and values_key in expert_data_np:
                steps = expert_data_np[steps_key]
                values = expert_data_np[values_key]
                rng = default_rng()
                indices = rng.choice(len(steps), min(len(steps), max_points), replace=False)
                steps, values = steps[indices], values[indices]
                ax.scatter(steps, values, s=0.5, alpha=0.2, c=color)
            ax.set_title(f"{title} Activation Scatter")
            ax.set_xlabel("Global Step")
            ax.set_ylabel("Global Expert ID")
            ax.set_ylim(-1, total_experts)

        for ax_heatmap, ax_zscore, key_suffix, title in [
            (axs[1, 0], axs[2, 0], "selected", "Top-K"),
            (axs[1, 1], axs[2, 1], "optimal", "Surprise Min-K"),
        ]:
            values_key = f"{key_suffix}_experts_values"
            if values_key in expert_data_np and expert_data_np[values_key].size > 0:
                data = expert_data_np[values_key]
                layers = data // num_experts
                experts_in_layer = data % num_experts
                heatmap, _, _ = np.histogram2d(layers, experts_in_layer, bins=[num_layers, num_experts])
                sns.heatmap(heatmap.T, ax=ax_heatmap, cmap="viridis")

                heatmap_norm = (heatmap - heatmap.mean(axis=0, keepdims=True)) / (heatmap.std(axis=0, keepdims=True) + 1e-6)
                sns.heatmap(heatmap_norm.T, ax=ax_zscore, cmap="viridis")

            ax_heatmap.set_title(f"Cumulative {title} Activation Heatmap")
            ax_heatmap.set_xlabel("Expert ID in Layer")
            ax_heatmap.set_ylabel("Layer ID")
            ax_zscore.set_title(f"Layer-Normalized {title} (Z-Score)")
            ax_zscore.set_xlabel("Expert ID in Layer")
            ax_zscore.set_ylabel("Layer ID")

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        self._save_plot(fig, "expert_dashboard_latest")

    def close(self):
        self.writer.close()
