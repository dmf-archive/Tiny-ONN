import math
from typing import Any
from collections import Counter
import numpy as np

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class ARCObserver:
    ARC_COLORS = ["black", "blue", "red", "green", "yellow", "grey", "magenta", "orange", "cyan", "brown"]

    def __init__(self):
        self.console = Console()
        self.routing_history = [] # List of (task_id, routing_weights)
        self.task_names = []

    def update_routing_history(self, task_ids: list[str], routing_info: list[dict[str, torch.Tensor]]):
        if not routing_info:
            return
        
        last_step = routing_info[-1]
        if "mlp_weights" in last_step:
            weights = last_step["mlp_weights"].detach().cpu()
            avg_w = weights.mean(dim=1).numpy()
            for tid, w in zip(task_ids, avg_w):
                self.routing_history.append((tid, w))
                if tid not in self.task_names:
                    self.task_names.append(tid)

    def compute_rmi(self, threshold=0.5) -> float:
        if not self.routing_history: return 0.0
        total = len(self.routing_history)
        task_counts = Counter([h[0] for h in self.routing_history])
        states = []
        for name, w in self.routing_history:
            mask = (w > threshold).astype(int)
            state_int = sum(m * (2**i) for i, m in enumerate(mask))
            states.append((name, state_int))
        state_counts = Counter([s[1] for s in states])
        h_r = -sum((c/total) * math.log2(c/total + 1e-10) for c in state_counts.values())
        h_r_t = 0.0
        for t, t_c in task_counts.items():
            p_t = t_c / total
            sub_h = 0.0
            t_states = [s[1] for s in states if s[0] == t]
            t_state_counts = Counter(t_states)
            for s_c in t_state_counts.values():
                p_s_t = s_c / t_c
                sub_h -= p_s_t * math.log2(p_s_t + 1e-10)
            h_r_t += p_t * sub_h
        return h_r - h_r_t

    def compute_itjd(self) -> float:
        if not self.routing_history: return 0.0
        task_avg = {}
        for name in self.task_names:
            ws = [h[1] for h in self.routing_history if h[0] == name]
            if ws:
                task_avg[name] = np.mean(ws, axis=0)
        
        dists = []
        names = list(task_avg.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                a, b = task_avg[names[i]], task_avg[names[j]]
                inter = np.minimum(a, b).sum()
                union = np.maximum(a, b).sum()
                dists.append(1.0 - (inter / (union + 1e-8)))
        return float(np.mean(dists)) if dists else 0.0

    def _render_grid(self, grid: torch.Tensor) -> Text:
        text = Text()
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                color_idx = int(grid[r, c].item())
                text.append("■ ", style=self.ARC_COLORS[color_idx] if 0 <= color_idx < 10 else "white")
            if r < h - 1: text.append("\n")
        return text

    def visualize_prediction(self, input_grid: torch.Tensor, target_grid: torch.Tensor, pred_grid: torch.Tensor, title: str = "ARC Result"):
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("Input"); table.add_column("Target"); table.add_column("Prediction")
        table.add_row(self._render_grid(input_grid), self._render_grid(target_grid), self._render_grid(pred_grid))
        status = "[bold green]PASS[/]" if torch.equal(target_grid, pred_grid) else "[bold red]FAIL[/]"
        self.console.print(Panel(table, title=f"{title} - {status}", expand=False))

    def calculate_metrics(
        self,
        loss: float,
        logits: torch.Tensor,
        labels: torch.Tensor,
        optimizer_diagnostics: dict[str, Any]
    ) -> dict[str, float]:
        # Token Accuracy
        logits_acc = logits[:, :-1, :].contiguous()
        labels_acc = labels[:, 1:].contiguous()
        mask = labels_acc != -100

        if mask.any():
            active_logits = logits_acc[mask]
            acc = (torch.argmax(active_logits, dim=-1) == labels_acc[mask]).float().mean().item()
            # Entropy (tau)
            tau = -torch.sum(F.softmax(active_logits, dim=-1) * F.log_softmax(active_logits, dim=-1), dim=-1).mean().item()
        else:
            acc = 0.0
            tau = 0.0

        grad_norm = optimizer_diagnostics.get("grad_norm", 0.0)
        # PI: Predictive Integrity proxy
        pi = math.exp(-(loss + grad_norm))

        return {
            "loss": loss,
            "token_acc": acc,
            "tau": tau,
            "pi": pi,
            "grad_norm": grad_norm
        }

    def log_metrics(self, metrics: dict[str, Any], step: int, epoch: int | None = None):
        prefix = f"[bold magenta]E{epoch} [/]" if epoch is not None else ""
        parts = [f"[bold cyan]S{step}[/]"]

        # Priority metrics
        priority = ["loss", "token_acc", "pi", "grad_norm"]
        for k in priority:
            if k in metrics:
                v = metrics[k]
                parts.append(f"{k}: [green]{v:.4f}[/]")

        # Other metrics
        for k, v in metrics.items():
            if k in priority or k == "epoch" or k == "step": continue
            if isinstance(v, (int, float)):
                parts.append(f"{k}: [blue]{v:.4f}[/]")
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                parts.append(f"{k}: [blue]{v.item():.4f}[/]")

        self.console.print(prefix + " | ".join(parts))
