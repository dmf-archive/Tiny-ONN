import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from sklearn.decomposition import PCA

from .config import TrainConfig


class Observer:
    ARC_COLORS: list[str] = [
        "black", "blue", "red", "green", "yellow",
        "grey", "magenta", "orange", "cyan", "brown",
    ]

    def __init__(self, console: Console, config: TrainConfig):
        self.console = console
        self.config = config
        self.vis_dir = Path(__file__).parent / "pic"
        self.vis_dir.mkdir(exist_ok=True)
        self.max_vis_files = 5

    def calculate_metrics(
        self, main_loss: torch.Tensor, model_outputs: dict, signals: dict, input_ids: torch.Tensor, model: nn.Module
    ) -> dict[str, float]:
        logits, labels = model_outputs["logits"], model_outputs["labels"]
        raw_weights = model_outputs.get("raw_weights", [])

        logits_acc, labels_acc = logits[:, :-1, :], labels[:, 1:]
        mask = labels_acc != -100
        active_logits = logits_acc[mask] if mask.any() else torch.empty(0, device=logits.device)
        acc = (
            (torch.argmax(active_logits, dim=-1) == labels_acc[mask]).float().mean().item()
            if mask.any()
            else 0.0
        )

        identity_transform_rate = 0.0
        if raw_weights:
            token_routing_failed_full = torch.stack([torch.all(rw == 0, dim=-1) for rw in raw_weights]).all(dim=0)
            token_routing_failed = token_routing_failed_full[:, :-1]
            input_ids_aligned = input_ids[:, :-1]
            is_identity = (input_ids_aligned == labels_acc)
            true_identity_mask = is_identity & token_routing_failed & mask
            if mask.sum() > 0:
                identity_transform_rate = (true_identity_mask.sum() / mask.sum()).item()

        mu_surp_norms = [torch.norm(s.detach(), p=2).item() for s in signals.get("mu_surprises", []) if s.numel() > 0]
        proto_surp_norms = [torch.norm(s.detach(), p=2).item() for s in signals.get("proto_surprises", []) if s.numel() > 0]

        all_surp_norms = mu_surp_norms + proto_surp_norms
        complexity_cost = sum(all_surp_norms) if all_surp_norms else 0.0
        pi_score = torch.exp(-1.0 * main_loss.detach() - 1.0 * complexity_cost).item()

        num_spl_modules = self.config.model.num_layers * 6
        act_rates = [0.0] * num_spl_modules
        if raw_weights:
            act_rates = [rw.gt(0).float().mean().item() for rw in raw_weights]

        num_layers = self.config.model.num_layers

        routing_failures = (
            [torch.all(rw == 0, dim=-1).float().mean().item() for rw in raw_weights] if raw_weights else [0.0]
        )
        routing_failure_rate = sum(routing_failures) / len(routing_failures) if routing_failures else 0.0

        metrics = {
            "main_loss": main_loss.item(),
            "token_acc": acc,
            "pi_score": pi_score,
            "route_jsd_loss": signals.get("route_jsd_loss", torch.tensor(0.0)).item(),
            "sample_entropy": model_outputs.get("sample_entropy", torch.tensor(0.0)).mean().item(),
            "tau": (
                -torch.sum(F.softmax(active_logits, dim=-1) * F.log_softmax(active_logits, dim=-1), dim=-1)
                .mean()
                .item()
                if active_logits.numel() > 0
                else 0.0
            ),
            "seq_len": float(labels.shape[1]),
            "activation_rate_avg": sum(act_rates) / len(act_rates) if act_rates else 0.0,
            "activation_rate_l0": sum(act_rates[:6]) / 6 if act_rates else 0.0,
            "activation_rate_l_mid": sum(act_rates[num_layers * 3 : num_layers * 3 + 6]) / 6 if len(act_rates) > num_layers * 3 + 6 else 0.0,
            "activation_rate_ln": sum(act_rates[-6:]) / 6 if act_rates else 0.0,
            "act_rates": act_rates,
            "routing_failure_rate": routing_failure_rate,
            "identity_transform_rate": identity_transform_rate,
        }

        routing_logits = model_outputs.get("routing_logits", [])
        if routing_logits:
            flat_logits = torch.cat([rl.detach().float().view(-1) for rl in routing_logits if rl.numel() > 0])
            if flat_logits.numel() > 0:
                metrics["gate_logit_avg"] = flat_logits.mean().item()
                metrics["gate_logit_max"] = flat_logits.max().item()
                metrics["gate_logit_top10_avg"] = flat_logits[flat_logits >= torch.quantile(flat_logits, 0.9)].mean().item()

        return metrics

    def log_step(
        self,
        epoch: int,
        step: int,
        task_idx: int | str,
        view_idx: int,
        metrics: dict[str, float],
        elapsed_time: float,
        consistency: dict[str, float] | None,
    ):
        steps_per_sec = 1 / elapsed_time if elapsed_time > 0 else float("inf")
        title = f"Epoch {epoch} | Step {step} | Task {task_idx}"
        if view_idx is not None:
            title += f" | View {view_idx}"

        table = Table(title=title, show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Loss (Main/JSD)", justify="center")
        table.add_column("Acc (Token)", justify="center")
        table.add_column("τ", justify="center")
        table.add_column("H(x)", justify="center")
        table.add_column("Seq Len", justify="center")
        table.add_column("PI", justify="center")
        table.add_column("Act % (Avg)", justify="center")
        table.add_column("Gate Logit (Top10/Avg/Max)", justify="center")
        table.add_column("Routing Fail % (True ID %)", justify="center")
        table.add_column("Consist (Cos/Euc)", justify="center")
        table.add_column("Speed (st/s)", justify="center")

        act_avg = metrics.get("activation_rate_avg", 0.0) * 100
        if consistency:
            consistency_str = f"{consistency.get('cos_sim', 0.0):.3f}/{consistency.get('euc_dist', 0.0):.3f}"
        else:
            consistency_str = "N/A"
        routing_fail_str = f"{metrics.get('routing_failure_rate', 0.0) * 100:.1f}%"
        identity_str = f"{metrics.get('identity_transform_rate', 0.0) * 100:.1f}%"

        table.add_row(
            f"{metrics.get('main_loss', 0.0):.3f}/{metrics.get('route_jsd_loss', 0.0):.4f}",
            f"{metrics.get('token_acc', 0.0):.3f}",
            f"{metrics.get('tau', 0.0):.3f}",
            f"{metrics.get('sample_entropy', 0.0):.3f}",
            f"{int(metrics.get('seq_len', 0))}",
            f"{metrics.get('pi_score', 0.0):.3f}",
            f"{act_avg:.1f}%",
            f"{metrics.get('gate_logit_top10_avg', 0.0):.3f}/{metrics.get('gate_logit_avg', 0.0):.3f}/{metrics.get('gate_logit_max', 0.0):.3f}",
            f"{routing_fail_str} ({identity_str})",
            consistency_str,
            f"{steps_per_sec:.2f}",
        )
        self.console.print(table)

    def visualize_prototypes(self, signals: dict, global_step: int):
        proto_weights = signals.get("proto_weights", [])
        raw_weights = signals.get("raw_weights", [])
        goodness_scores = signals.get("goodness_scores", [])

        if not proto_weights or not raw_weights or not goodness_scores:
            return


        num_spl = 6
        num_layers = self.config.model.num_layers

        fig, axes = plt.subplots(num_layers, num_spl, figsize=(20, 5 * num_layers), squeeze=False)
        fig.suptitle(f"Prototype Goodness Distribution @ Step {global_step}", fontsize=16)
        color_map = {"good": "green", "bad": "red", "inactive": "grey"}
        names = ["attn_q", "attn_k", "attn_v", "attn_o", "ffn_sbl1", "ffn_sbl2"]

        for i in range(num_layers):
            for j in range(num_spl):
                ax = axes[i, j]
                idx = i * num_spl + j

                if idx >= len(proto_weights): continue

                protos = proto_weights[idx].cpu().to(torch.float32)
                rw = raw_weights[idx].cpu().to(torch.float32)
                goodness = goodness_scores[idx].cpu().to(torch.float32)

                num_activations = (rw > 0).float().sum(dim=(0, 1))
                avg_goodness = (goodness * (rw > 0).float()).sum(dim=(0, 1)) / (num_activations + 1e-6)

                activated_goodness = avg_goodness[num_activations > 0]
                threshold = torch.quantile(activated_goodness, 0.5) if activated_goodness.numel() > 0 else 0.0

                statuses = []
                for k in range(avg_goodness.shape[0]):
                    if num_activations[k] == 0:
                        statuses.append("inactive")
                    elif avg_goodness[k] >= threshold:
                        statuses.append("good")
                    else:
                        statuses.append("bad")

                if protos.shape[0] > 1:
                    pca = PCA(n_components=2)
                    protos_2d = pca.fit_transform(protos.numpy())
                    colors = [color_map.get(s, "black") for s in statuses]
                    ax.scatter(protos_2d[:, 0], protos_2d[:, 1], c=colors, alpha=0.7, s=10)
                else:
                    ax.text(0.5, 0.5, "Not enough data", ha='center', va='center')

                ax.set_title(f"L{i} - {names[j]}")
                ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = self.vis_dir / f"prototypes_step_{global_step}.png"
        plt.savefig(save_path)
        plt.close(fig)

        vis_files = sorted(self.vis_dir.glob("*.png"), key=os.path.getmtime)
        if len(vis_files) > self.max_vis_files:
            os.remove(vis_files[0])


    def maybe_log_and_visualize(
        self,
        epoch: int,
        step: int,
        task_idx: int,
        view_idx: int,
        metrics: dict,
        elapsed_time: float,
        signals: dict,
        evaluator,
        eval_loader,
        current_task_idx: int,
        save_checkpoint_fn,
        consistency: dict[str, float] | None,
        reinit_fn,
    ):
        if step % self.config.log_interval == 0:
            self.log_step(epoch, step, task_idx, view_idx, metrics, elapsed_time, consistency)
            self.visualize_prototypes(signals, step)
            save_checkpoint_fn(task_idx, view_idx)
            reinit_fn()

        if step > 0 and step % self.config.eval_interval == 0:
            evaluator.run(eval_loader, current_task_idx, step)

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

    def visualize_evaluation_sample(
        self,
        input_grid: torch.Tensor,
        target_grid: torch.Tensor,
        pred_grid: torch.Tensor | None,
        pred_tokens: list[int] | None,
        step: int,
    ):
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
        eval_loss = metrics.get("eval_loss", 0.0)
        eval_grid_acc = metrics.get("eval_grid_acc", 0.0)
        total_count = int(metrics.get("total_count", 0))
        summary_text = f"Avg Loss: {eval_loss:.4f} | Grid Accuracy: {eval_grid_acc:.4f} on {total_count} samples."
        self.console.print(
            Padding(
                f"[bold yellow]===== EVALUATION COMPLETE (Step: {step}) =====\n"
                f"{summary_text}\n"
                f"=============================================[/bold yellow]",
                (1, 2),
            )
        )
