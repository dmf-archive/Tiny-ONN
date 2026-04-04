from __future__ import annotations

import argparse
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Dataset

from src.models.dynsiha.recursive.configuration_recursive_dynsiha import (
    RecursiveDynSIHAConfig,
)
from src.models.dynsiha.recursive.modeling_recursive_dynsiha import (
    RecursiveDynSIHAForCausalLM,
)
from src.optimizers.ars2_neo import ARS2Neo


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    n_heads: int
    d_mlp: int
    n_layers_dense: int
    n_layers_rds: int
    n_steps: int
    act_weight: float
    seq_len: int


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    lr: float
    max_steps: int
    log_interval: int
    early_stop_acc: float


@dataclass(frozen=True)
class DataConfig:
    p: int
    train_frac: float
    seed: int


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig


CONFIG = Config(
    model=ModelConfig(
        d_model=128,
        n_heads=4,
        d_mlp=512,
        n_layers_dense=3,
        n_layers_rds=2,
        n_steps=4,
        act_weight=0.01,
        seq_len=3,
    ),
    training=TrainingConfig(
        batch_size=256,
        lr=1e-3,
        max_steps=2000,
        log_interval=100,
        early_stop_acc=0.98,
    ),
    data=DataConfig(
        p=97,
        train_frac=0.3,
        seed=1337,
    ),
)

ROUTING_CONFIG = {
    "fars_weight": 0.1,
    "temperature_initial": 1.0,
    "temperature_final": 0.1,
    "temperature_warmup": 500,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ModAddDataset(Dataset):
    def __init__(self, pairs: torch.Tensor, labels: torch.Tensor):
        self.pairs = pairs
        self.labels = labels

    def __len__(self) -> int:
        return self.pairs.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pairs[idx], self.labels[idx]


def build_mod_addition(p: int, train_frac: float, seed: int) -> tuple[ModAddDataset, ModAddDataset]:
    g = torch.Generator().manual_seed(seed)
    a = torch.arange(p, dtype=torch.long)
    b = torch.arange(p, dtype=torch.long)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    base_pairs = torch.stack([aa.flatten(), bb.flatten()], dim=1)
    total = base_pairs.size(0)
    perm = torch.randperm(total, generator=g)
    shuffled = base_pairs[perm]
    add_token = p
    sub_token = p + 1
    add_count = int(total * 0.55)
    add_pairs = shuffled[:add_count]
    sub_pairs = shuffled[add_count:]
    ops = torch.empty(total, dtype=torch.long)
    labels = torch.empty(total, dtype=torch.long)
    ops[:add_count] = add_token
    ops[add_count:] = sub_token
    labels[:add_count] = (add_pairs[:, 0] + add_pairs[:, 1]) % p
    labels[add_count:] = (sub_pairs[:, 0] - sub_pairs[:, 1]) % p
    pairs = torch.cat([shuffled, ops.unsqueeze(1)], dim=1)
    split = int(total * train_frac)
    train_ds = ModAddDataset(pairs[:split], labels[:split])
    test_ds = ModAddDataset(pairs[split:], labels[split:])
    return train_ds, test_ds


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.RMSNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        self.ln2 = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out = self.attn(h, h, h, need_weights=False)[0]
        x = x + attn_out
        h = self.ln2(x)
        x = x + self.ff(h)
        return x


class ModAddTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab: int, steps: int, n_layers: int, use_sia: bool):
        super().__init__()
        self.steps = steps
        self.use_sia = use_sia
        self.emb = nn.Embedding(vocab, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(cfg.seq_len, cfg.d_model) / math.sqrt(cfg.d_model))
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_mlp) for _ in range(n_layers)]
        )
        self.ln = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab)
        self.halt = nn.Linear(cfg.d_model, 1)

    def forward_steps(self, ids: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        x = self.emb(ids) + self.pos[: ids.size(1)]
        logits_list: list[torch.Tensor] = []
        halt_list: list[torch.Tensor] = []
        for s in range(self.steps):
            if self.use_sia and s > 0:
                x = x.detach()
            for block in self.blocks:
                x = block(x)
            h = self.ln(x)
            pooled = h[:, -1]
            logits_list.append(self.head(pooled))
            halt_list.append(self.halt(pooled).squeeze(-1))
        return logits_list, halt_list

    @property
    def steps(self) -> int:
        return self._steps

    @steps.setter
    def steps(self, value: int) -> None:
        self._steps = value

    def forward_bundle(self, ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        logits_list, halt_list = self.forward_steps(ids)
        loss_stack = None
        if labels is not None:
            loss_stack = torch.stack(
                [F.cross_entropy(logits, labels, reduction="none") for logits in logits_list]
            )
        halt_logits = None
        if halt_list:
            halt_logits = torch.stack(halt_list).permute(1, 0)
        return {
            "logits": logits_list[-1],
            "logits_list": logits_list,
            "loss_stack": loss_stack,
            "halt_logits": halt_logits,
        }


class RecursiveDynSIHATrunk(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab: int, steps: int, plsd_as_hypernet: bool = True):
        super().__init__()
        config = RecursiveDynSIHAConfig(
            vocab_size=vocab,
            hidden_size=cfg.d_model,
            max_refinement_steps=steps,
            max_inference_steps=steps,
            num_heads=cfg.n_heads,
            num_experts=32,
            top_k=4,
            max_position_embeddings=64,
            use_cache=True,
            use_cache_in_train=True,
            use_sia=True,
            use_act_inference=False,
            plsd_as_hypernet=plsd_as_hypernet,
        )
        self.model = RecursiveDynSIHAForCausalLM(config)
        self._steps = steps
        self.plsd_as_hypernet = plsd_as_hypernet

    @property
    def steps(self) -> int:  # type: ignore[override]
        return self._steps

    def forward_bundle(self, ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, Any]:
        input_ids = ids.clone()
        label_seq = None
        if labels is not None:
            label_seq = torch.full_like(input_ids, -100)
            label_seq[:, 2] = labels
        out = self.model(input_ids=input_ids, labels=label_seq, use_cache=True, return_dict=True)
        loss_stack = out.all_step_losses
        halt_logits = out.halt_logits
        halt_step = None
        if halt_logits is not None:
            # halt_logits 现在是 [T, B] -> 转置为 [B, T]
            halt_step = halt_logits.transpose(0, 1)
        eff_k = float(out.eff_k.item()) if out.eff_k is not None else None
        routing_weights = out.routing_weights
        if out.all_step_logits is not None:
            step_logits = out.all_step_logits[:, :, -1, :]
            logits_list = [step_logits[s] for s in range(step_logits.size(0))]
            logits_last = logits_list[-1]
        else:
            logits_last = out.logits[:, -1]
            logits_list = [logits_last for _ in range(self._steps)]
        return {
            "logits": logits_last,
            "logits_list": logits_list,
            "loss_stack": loss_stack,
            "halt_logits": halt_step,
            "eff_k": eff_k,
            "routing_weights": routing_weights,
        }


def flatten_grads(grads: Iterable[torch.Tensor | None], params: list[nn.Parameter]) -> torch.Tensor:
    parts = []
    for g, p in zip(grads, params):
        if g is not None:
            parts.append(g.reshape(-1))
        else:
            parts.append(torch.zeros_like(p).reshape(-1))
    if len(parts) == 0:
        return torch.zeros(1, device="cpu")
    return torch.cat(parts)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a.norm() + 1e-12
    b_norm = b.norm() + 1e-12
    return float((a @ b) / (a_norm * b_norm))


def compute_itjd_rmi(
    routing_weights: torch.Tensor,
    step_idx: torch.LongTensor,
    num_steps: int,
) -> tuple[float, float]:
    E = routing_weights.size(-1)
    step_weights = torch.zeros(num_steps, E, device=routing_weights.device, dtype=routing_weights.dtype)
    for t in range(num_steps):
        w_t = routing_weights[t] if routing_weights.dim() == 3 else routing_weights
        step_weights[t] = w_t.mean(dim=0)
    occupied = step_weights.sum(1) > 0
    if occupied.sum() < 2:
        return 0.0, 0.0
    w = step_weights[occupied]
    inter = torch.minimum(w.unsqueeze(0), w.unsqueeze(1)).sum(-1)
    union = torch.maximum(w.unsqueeze(0), w.unsqueeze(1)).sum(-1)
    jaccard = inter / (union + 1e-9)
    itjd = (1 - jaccard).triu(1).sum() / (jaccard.triu(1).numel() + 1e-9)
    top1 = routing_weights.argmax(-1) if routing_weights.dim() == 3 else routing_weights.argmax(-1)
    joint = torch.zeros(num_steps, E, device=routing_weights.device, dtype=routing_weights.dtype)
    for t in range(num_steps):
        if routing_weights.dim() == 3:
            cnt = torch.bincount(top1[t], minlength=E).float()
        else:
            cnt = torch.bincount(top1, minlength=E).float()
        joint[t] = cnt / (cnt.sum() + 1e-9)
    p_t = joint.sum(1)
    p_e = joint.sum(0)
    mi = 0.0
    for t in range(num_steps):
        for e in range(E):
            p_te = joint[t, e].item()
            p_t_val = p_t[t].item()
            p_e_val = p_e[e].item()
            if p_te > 0 and p_t_val > 0 and p_e_val > 0:
                mi += p_te * math.log(p_te / (p_t_val * p_e_val))
    return float(itjd), float(mi)


def compute_loss_components(
    model: nn.Module,
    ids: torch.Tensor,
    labels: torch.Tensor,
    act_weight: float,
    current_step: int = 1000,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor, torch.Tensor]:
    bundle = model.forward_bundle(ids, labels)
    logits_list = bundle["logits_list"]
    loss_stack = bundle["loss_stack"]
    halt_logits = bundle["halt_logits"]
    eff_k = bundle.get("eff_k")
    routing_weights = bundle.get("routing_weights")
    if loss_stack is None:
        loss_stack = torch.stack(
            [F.cross_entropy(logits, labels, reduction="none") for logits in logits_list]
        )
    if halt_logits is None:
        halt_logits = torch.zeros(ids.size(0), int(model.steps), device=ids.device, dtype=ids.dtype)
    steps = loss_stack.size(0)
    if steps == 1:
        total_loss = loss_stack[0].mean()
        eff_k = None
        stats: dict[str, Any] = {
            "plsd_losses": [float(loss_stack[0].mean().item())],
            "pred_losses": [float(torch.exp(halt_logits).mean().item())],
            "eff_K": eff_k,
        }
        return total_loss, stats, loss_stack, halt_logits
    if halt_logits.size(1) != steps:
        halt_logits = halt_logits[:, :steps]
    best_loss, best_step = loss_stack.min(0)

    warmup_steps = 300
    if current_step < warmup_steps:
        alpha = current_step / warmup_steps
        full_loss = loss_stack.mean()
        step_mask = torch.arange(steps, device=ids.device)[None, :] <= best_step[:, None]
        oracle_loss = (loss_stack.permute(1, 0) * step_mask).sum() / (step_mask.sum() + 1e-9)
        main_loss = (1 - alpha) * full_loss + alpha * oracle_loss
    else:
        step_mask = torch.arange(steps, device=ids.device)[None, :] <= best_step[:, None]
        main_loss = (loss_stack.permute(1, 0) * step_mask).sum() / (step_mask.sum() + 1e-9)
    target_loss = torch.log(loss_stack.detach().permute(1, 0) + 1.0)
    act_loss = F.mse_loss(halt_logits, target_loss)

    total_loss = main_loss + act_weight * act_loss
    oracle_dist = torch.bincount(best_step, minlength=steps).float().cpu()
    oracle_dist = (oracle_dist / oracle_dist.sum()).tolist()
    itjd, rmi = 0.0, 0.0
    if routing_weights is not None:
        itjd, rmi = compute_itjd_rmi(routing_weights, best_step, steps)
    eff_layer = float(best_step.float().mean().item())
    eff_act = None
    if eff_k is not None:
        num_experts = 32
        top_k = 4
        eff_act = eff_k / (num_experts * top_k)
    routing_entropy = None
    task_routing = None
    if routing_weights is not None:
        routing_entropy = float(torch.distributions.Categorical(probs=routing_weights.mean(0)).entropy().mean().item())
        task_routing = analyze_task_routing(routing_weights, ids, labels, CONFIG.data.p)
    stats: dict[str, Any] = {
        "eff_layer": eff_layer,
        "eff_act": eff_act,
        "eff_K": eff_k,
        "oracle_dist": oracle_dist,
        "pred_losses": torch.exp(halt_logits).mean(0).detach().cpu().tolist(),
        "plsd_losses": loss_stack.mean(1).detach().cpu().tolist(),
        "itjd": itjd,
        "rmi": rmi,
        "routing_entropy": routing_entropy,
        "task_routing": task_routing,
    }
    return total_loss, stats, loss_stack, halt_logits


def analyze_task_routing(
    routing_weights: torch.Tensor,
    ids: torch.Tensor,
    labels: torch.Tensor,
    p: int,
) -> dict[str, list[float]]:
    add_token = p
    sub_token = p + 1
    ops = ids[:, -1]
    add_mask = ops == add_token
    sub_mask = ops == sub_token
    # routing_weights 形状：[num_steps, batch, num_experts]
    # 我们需要对 step 和 batch 维度都求平均，得到 [num_experts]
    result: dict[str, list[float]] = {"add": [], "sub": []}
    if add_mask.any():
        w_add = routing_weights[:, add_mask, :].mean(dim=(0, 1))
        result["add"] = w_add.cpu().tolist()
    if sub_mask.any():
        w_sub = routing_weights[:, sub_mask, :].mean(dim=(0, 1))
        result["sub"] = w_sub.cpu().tolist()
    return result


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for ids, labels in data_loader:
            ids = ids.to(device)
            labels = labels.to(device)
            bundle = model.forward_bundle(ids, labels)
            logits = bundle["logits"]
            loss_stack = bundle["loss_stack"]
            if loss_stack is not None:
                total_loss += float(loss_stack.mean().item()) * ids.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += int((preds == labels).sum().item())
            total_samples += ids.size(0)
    model.train()
    return total_loss / total_samples, total_correct / total_samples


def compute_phi_s(
    model: nn.Module,
    loss_stack: torch.Tensor,
    total_loss: torch.Tensor,
) -> list[float]:
    params = [p for p in model.parameters() if p.requires_grad]
    total_grads = torch.autograd.grad(total_loss, params, retain_graph=True, allow_unused=True)
    total_vec = flatten_grads(total_grads, params)
    phi_list: list[float] = []
    for s in range(loss_stack.size(0)):
        step_loss = loss_stack[s].mean()
        step_grads = torch.autograd.grad(step_loss, params, retain_graph=True, allow_unused=True)
        step_vec = flatten_grads(step_grads, params)
        phi_list.append(cosine_similarity(step_vec, total_vec))
    return phi_list


def infer_logits(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    bundle = model.forward_bundle(ids, None)
    logits = bundle["logits"]
    logits_list = bundle["logits_list"]
    halt_logits = bundle["halt_logits"]
    if model.steps == 1:  # type: ignore[operator]
        return logits
    if halt_logits is None:
        return logits
    logits = torch.stack(logits_list).permute(1, 0, 2)
    # 回归模式：选择预测 Loss 最小的步骤
    # halt_logits: [B, T]
    selected = halt_logits.argmin(dim=1)
    gathered = logits.gather(1, selected[:, None, None].expand(-1, 1, logits.size(-1)))
    return gathered.squeeze(1)


@dataclass
class MetricsBuffer:
    loss_window: list[float]
    acc_window: list[float]
    eff_layer_window: list[float]
    eff_act_window: list[float]
    itjd_window: list[float]
    rmi_window: list[float]
    routing_entropy_window: list[float]
    oracle_dist_sum: list[float]
    oracle_count: int
    task_routing_add_sum: list[float]
    task_routing_sub_sum: list[float]
    task_routing_count: int

    @classmethod
    def create(cls, window_size: int, num_steps: int, num_experts: int = 32):
        return cls(
            loss_window=[],
            acc_window=[],
            eff_layer_window=[],
            eff_act_window=[],
            itjd_window=[],
            rmi_window=[],
            routing_entropy_window=[],
            oracle_dist_sum=[0.0] * num_steps,
            oracle_count=0,
            task_routing_add_sum=[0.0] * num_experts,
            task_routing_sub_sum=[0.0] * num_experts,
            task_routing_count=0,
        )

    def update(self, loss: float, acc: float, stats: dict[str, Any]) -> None:
        self.loss_window.append(loss)
        self.acc_window.append(acc)
        if "eff_layer" in stats and stats["eff_layer"] is not None:
            self.eff_layer_window.append(float(stats["eff_layer"]))
        if "eff_act" in stats and stats["eff_act"] is not None:
            self.eff_act_window.append(float(stats["eff_act"]))
        if "itjd" in stats:
            self.itjd_window.append(stats["itjd"])
        if "rmi" in stats:
            self.rmi_window.append(stats["rmi"])
        if "routing_entropy" in stats and stats["routing_entropy"] is not None:
            self.routing_entropy_window.append(stats["routing_entropy"])
        if "oracle_dist" in stats:
            for i, v in enumerate(stats["oracle_dist"]):
                if i < len(self.oracle_dist_sum):
                    self.oracle_dist_sum[i] += v
            self.oracle_count += 1
        if "task_routing" in stats and stats["task_routing"] is not None:
            tr = stats["task_routing"]
            if "add" in tr and tr["add"]:
                for i, v in enumerate(tr["add"]):
                    if i < len(self.task_routing_add_sum):
                        self.task_routing_add_sum[i] += v
            if "sub" in tr and tr["sub"]:
                for i, v in enumerate(tr["sub"]):
                    if i < len(self.task_routing_sub_sum):
                        self.task_routing_sub_sum[i] += v
            self.task_routing_count += 1

    def get_averages(self) -> dict:
        n = len(self.loss_window)
        if n == 0:
            return {}
        result = {
            "loss": sum(self.loss_window[-n:]) / n,
            "acc": sum(self.acc_window[-n:]) / n,
        }
        if self.eff_layer_window:
            result["eff_layer"] = sum(self.eff_layer_window[-n:]) / n
        if self.eff_act_window:
            result["eff_act"] = sum(self.eff_act_window[-n:]) / n
        if self.itjd_window:
            result["itjd"] = sum(self.itjd_window[-n:]) / n
        if self.rmi_window:
            result["rmi"] = sum(self.rmi_window[-n:]) / n
        if self.routing_entropy_window:
            result["routing_entropy"] = sum(self.routing_entropy_window[-n:]) / n
        if self.oracle_count > 0:
            result["oracle_dist"] = [v / self.oracle_count for v in self.oracle_dist_sum]
        return result

    def reset_window(self) -> None:
        self.loss_window.clear()
        self.acc_window.clear()
        self.eff_layer_window.clear()
        self.eff_act_window.clear()
        self.itjd_window.clear()
        self.rmi_window.clear()
        self.routing_entropy_window.clear()
        self.oracle_dist_sum[:] = [0.0] * len(self.oracle_dist_sum)
        self.oracle_count = 0
        self.task_routing_add_sum[:] = [0.0] * len(self.task_routing_add_sum)
        self.task_routing_sub_sum[:] = [0.0] * len(self.task_routing_sub_sum)
        self.task_routing_count = 0


def run_experiment(
    name: str,
    model: nn.Module,
    optimizer: ARS2Neo,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> None:
    model.train()
    metrics = MetricsBuffer.create(cfg.training.log_interval, int(model.steps))  # type: ignore[arg-type]
    train_iter = iter(train_loader)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"[cyan]{name}", total=cfg.training.max_steps)

        for step in range(cfg.training.max_steps):
            try:
                ids, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                ids, labels = next(train_iter)
            ids = ids.to(device)
            labels = labels.to(device)

            def closure(_ids=ids, _labels=labels, _step=step) -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                total_loss, _, _, _ = compute_loss_components(model, _ids, _labels, cfg.model.act_weight, current_step=_step)
                return total_loss

            loss = optimizer.step(closure)

            if step % cfg.training.log_interval == 0 or step == cfg.training.max_steps - 1:
                model.eval()
                with torch.enable_grad():
                    total_loss, stats_dict, loss_stack, _ = compute_loss_components(
                        model, ids, labels, cfg.model.act_weight, current_step=step
                    )
                    _ = compute_phi_s(model, loss_stack, total_loss)
                model.zero_grad(set_to_none=True)
                with torch.no_grad():
                    logits = infer_logits(model, ids)
                    preds = logits.argmax(dim=-1)
                    train_acc = float((preds == labels).float().mean().item())

                eval_loss, eval_acc = evaluate_model(model, test_loader, device)
                metrics.update(float(loss.item()), train_acc, stats_dict)

                avg = metrics.get_averages()
                loss_avg = avg.get("loss", 0)
                train_acc_avg = avg.get("acc", 0)

                log_lines = [f"[{name}] Step {step:04d} | Train Loss: {loss_avg:.4f} | Train Acc: {train_acc_avg:.2%} | Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.2%}"]

                if loss_avg < 0.01:
                    progress.console.print(" |".join(log_lines))
                    progress.update(task, completed=step + 1)
                    break
                if "eff_layer" in avg:
                    log_lines.append(f" eff_layer: {avg['eff_layer']:.2f}")
                if "eff_act" in avg:
                    log_lines.append(f" eff_act: {avg['eff_act']:.3f}")
                if "itjd" in avg:
                    log_lines.append(f" ITJD: {avg['itjd']:.3f}")
                if "rmi" in avg:
                    log_lines.append(f" RMI: {avg['rmi']:.3f}")
                if "routing_entropy" in avg:
                    log_lines.append(f" RouteEnt: {avg['routing_entropy']:.3f}")
                if "oracle_dist" in avg:
                    oracle_str = ", ".join(f"{v:.2f}" for v in avg["oracle_dist"])
                    log_lines.append(f" Oracle: [{oracle_str}]")

                if "pred_losses" in stats_dict:
                    pred_str = ", ".join(f"{v:.2f}" for v in stats_dict["pred_losses"])
                    log_lines.append(f" PredL: [{pred_str}]")

                if metrics.task_routing_count > 0:
                    add_str = ", ".join(f"{v/metrics.task_routing_count:.3f}" for v in metrics.task_routing_add_sum[:8])
                    sub_str = ", ".join(f"{v/metrics.task_routing_count:.3f}" for v in metrics.task_routing_sub_sum[:8])
                    log_lines.append(f" Route[Add]: [{add_str}]")
                    log_lines.append(f" Route[Sub]: [{sub_str}]")

                progress.console.print(" |".join(log_lines))
                metrics.reset_window()

                model.train()
                if eval_acc >= cfg.training.early_stop_acc:
                    progress.update(task, completed=step + 1)
                    break

            progress.update(task, advance=1)


def build_optimizer(model: nn.Module, lr: float) -> ARS2Neo:
    params_hi = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    params_lo = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return ARS2Neo(
        [
            {"params": params_hi, "is_rmsuon_group": True},
            {"params": params_lo, "is_rmsuon_group": False},
        ],
        lr=lr,
        k=0,
        rho=0.0,
        adaptive_sync=False,
    )


def run(skip_dense: bool = True) -> None:
    set_seed(CONFIG.data.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_ds = build_mod_addition(CONFIG.data.p, CONFIG.data.train_frac, CONFIG.data.seed)
    train_loader = DataLoader(train_ds, batch_size=CONFIG.training.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG.training.batch_size, shuffle=True)

    configs: list[tuple[str, int, int, bool]] = []

    # 只对比 Dense 和 BlockSIA-HyperPLSD
    configs.append(("Dense", 1, CONFIG.model.n_layers_dense, False))
    configs.append(("RDS-BlockSIA-HyperPLSD", CONFIG.model.n_steps, CONFIG.model.n_layers_rds, True))
    # 注释掉其他 RDS 变体
    # configs.append(("RDS-BlockSIA", CONFIG.model.n_steps, CONFIG.model.n_layers_rds, False))
    # configs.append(("RDS-NoSIA", CONFIG.model.n_steps, CONFIG.model.n_layers_rds, False))

    for name, steps, layers, plsd_cfg in configs:
        vocab_size = CONFIG.data.p + 2
        if name == "Dense":
            model: nn.Module = ModAddTransformer(CONFIG.model, vocab_size, steps, layers, use_sia=False).to(device)
        else:
            use_sia = "NoSIA" not in name
            rds_model = RecursiveDynSIHATrunk(
                CONFIG.model, vocab_size, steps,
                plsd_as_hypernet=plsd_cfg
            ).to(device)
            if not use_sia:
                rds_model.model.config.use_sia = False  # type: ignore[union-attr]
                rds_model.model.block.use_sia = False  # type: ignore[union-attr]
            model = rds_model

        optimizer = build_optimizer(model, CONFIG.training.lr)
        print(f"\n>>> Starting {name} Experiment (PLSD_Hypernet={plsd_cfg})")
        run_experiment(name, model, optimizer, train_loader, test_loader, CONFIG, device)
        print(f"--- {name} FINISHED ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dense", action="store_true", default=True)
    args = parser.parse_args()
    run(args.skip_dense)
