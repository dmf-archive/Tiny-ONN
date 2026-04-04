from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


DistillMode = Literal[
    "ce_only",
    "teacher_only",
    "fixed_kd",
    "confidence_kd",
    "entropy_kd",
    "pi_weighted_kd",
]


@dataclass(frozen=True)
class SmallTransformerConfig:
    d_model: int
    n_heads: int
    n_layers: int
    d_mlp: int
    seq_len: int


@dataclass(frozen=True)
class SpikeConfig:
    p: int = 31
    train_frac: float = 0.5
    seed: int = 7
    batch_size: int = 128
    epochs_a: int = 80
    epochs_b: int = 40
    lr: float = 1e-3
    fixed_lambda: float = 0.5
    pi_alpha: float = 1.0
    pi_gamma: float = 0.5
    pi_lambda_floor: float = 0.1
    model: SmallTransformerConfig = SmallTransformerConfig(
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_mlp=256,
        seq_len=3,
    )


class SmallTransformer(nn.Module):
    def __init__(self, cfg: SmallTransformerConfig, vocab_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(cfg.seq_len, cfg.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_mlp,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids) + self.pos[: ids.size(1)]
        h = self.encoder(x)
        y = self.norm(h[:, -1])
        return self.head(y)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_task_dataset(p: int, op: Literal["add", "sub"]) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.arange(p, dtype=torch.long)
    b = torch.arange(p, dtype=torch.long)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    ids = torch.stack([aa.flatten(), bb.flatten()], dim=1)
    if op == "add":
        labels = (ids[:, 0] + ids[:, 1]) % p
        op_token = torch.full((ids.size(0), 1), p, dtype=torch.long)
    else:
        labels = (ids[:, 0] - ids[:, 1]) % p
        op_token = torch.full((ids.size(0), 1), p + 1, dtype=torch.long)
    return torch.cat([ids, op_token], dim=1), labels


def split_train_test(
    ids: torch.Tensor,
    labels: torch.Tensor,
    train_frac: float,
    seed: int,
) -> tuple[TensorDataset, TensorDataset]:
    g = torch.Generator().manual_seed(seed)
    n = ids.size(0)
    perm = torch.randperm(n, generator=g)
    cut = int(n * train_frac)
    train_idx = perm[:cut]
    test_idx = perm[cut:]
    train_ds = TensorDataset(ids[train_idx], labels[train_idx])
    test_ds = TensorDataset(ids[test_idx], labels[test_idx])
    return train_ds, test_ds


def make_loader(dataset: TensorDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)


def soft_target_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def compute_lambda(
    mode: DistillMode,
    teacher_probs: torch.Tensor,
    labels: torch.Tensor,
    fixed_lambda: float,
    pi_alpha: float,
    pi_gamma: float,
    pi_lambda_floor: float,
) -> torch.Tensor:
    batch = labels.size(0)
    device = labels.device
    if mode == "ce_only":
        return torch.zeros(batch, device=device)
    if mode == "teacher_only":
        return torch.ones(batch, device=device)
    if mode == "fixed_kd":
        return torch.full((batch,), float(fixed_lambda), device=device)
    if mode == "confidence_kd":
        return teacher_probs.max(dim=-1).values.clamp(0.0, 1.0)
    entropy = -(teacher_probs * torch.log(teacher_probs.clamp_min(1e-9))).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(teacher_probs.size(-1)), device=device))
    complexity = (entropy / max_entropy).clamp(0.0, 1.0)
    if mode == "entropy_kd":
        return (1.0 - complexity).clamp(0.0, 1.0)
    teacher_confidence = teacher_probs.max(dim=-1).values
    inaccuracy = (1.0 - teacher_confidence).clamp(0.0, 1.0)
    score = 1.0 - ((inaccuracy + pi_gamma * complexity) / (1.0 + pi_gamma))
    pi = score.clamp(0.0, 1.0).pow(pi_alpha)
    return (pi_lambda_floor + (1.0 - pi_lambda_floor) * pi).clamp(0.0, 1.0)


def summarize_lambda(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "lambda_mean": 0.0,
            "lambda_std": 0.0,
            "lambda_min": 0.0,
            "lambda_max": 0.0,
            "lambda_p10": 0.0,
            "lambda_p50": 0.0,
            "lambda_p90": 0.0,
        }
    tensor = torch.tensor(values, dtype=torch.float32)
    quantiles = torch.quantile(tensor, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32))
    return {
        "lambda_mean": float(tensor.mean().item()),
        "lambda_std": float(tensor.std(unbiased=False).item()),
        "lambda_min": float(tensor.min().item()),
        "lambda_max": float(tensor.max().item()),
        "lambda_p10": float(quantiles[0].item()),
        "lambda_p50": float(quantiles[1].item()),
        "lambda_p90": float(quantiles[2].item()),
    }


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ids, labels in loader:
            ids = ids.to(device)
            labels = labels.to(device)
            logits = model(ids)
            pred = logits.argmax(dim=-1)
            correct += int((pred == labels).sum().item())
            total += int(labels.size(0))
    return 0.0 if total == 0 else correct / total


def train_task_a(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        for ids, labels in loader:
            ids = ids.to(device)
            labels = labels.to(device)
            logits = model(ids)
            loss = F.cross_entropy(logits, labels)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()


def train_task_b(
    model: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    cfg: SpikeConfig,
    mode: DistillMode,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    teacher.eval()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    lambda_values: list[float] = []
    vocab = cfg.p + 2
    for _ in range(cfg.epochs_b):
        for ids, labels in loader:
            ids = ids.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                teacher_probs = F.softmax(teacher(ids), dim=-1)
            logits = model(ids)
            lamb = compute_lambda(
                mode=mode,
                teacher_probs=teacher_probs,
                labels=labels,
                fixed_lambda=cfg.fixed_lambda,
                pi_alpha=cfg.pi_alpha,
                pi_gamma=cfg.pi_gamma,
                pi_lambda_floor=cfg.pi_lambda_floor,
            )
            one_hot = F.one_hot(labels, num_classes=vocab).to(logits.dtype)
            mix = ((1.0 - lamb).unsqueeze(1) * one_hot) + (lamb.unsqueeze(1) * teacher_probs)
            loss = soft_target_ce(logits, mix)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            lambda_values.extend(lamb.detach().cpu().tolist())
    return summarize_lambda(lambda_values)


def run_single_mode(cfg: SpikeConfig, mode: DistillMode, device: torch.device) -> dict[str, float | str]:
    set_seed(cfg.seed)
    vocab = cfg.p + 2
    add_ids, add_labels = build_task_dataset(cfg.p, "add")
    sub_ids, sub_labels = build_task_dataset(cfg.p, "sub")
    add_train, add_test = split_train_test(add_ids, add_labels, cfg.train_frac, cfg.seed)
    sub_train, sub_test = split_train_test(sub_ids, sub_labels, cfg.train_frac, cfg.seed + 1)
    add_train_loader = make_loader(add_train, cfg.batch_size, True, cfg.seed)
    add_test_loader = make_loader(add_test, cfg.batch_size, False, cfg.seed)
    sub_train_loader = make_loader(sub_train, cfg.batch_size, True, cfg.seed + 1)
    sub_test_loader = make_loader(sub_test, cfg.batch_size, False, cfg.seed + 1)

    student = SmallTransformer(cfg.model, vocab).to(device)
    train_task_a(student, add_train_loader, device, cfg.epochs_a, cfg.lr)
    task_a_acc_before_b = evaluate_accuracy(student, add_test_loader, device)
    teacher = deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    lambda_stats = train_task_b(student, teacher, sub_train_loader, cfg, mode, device)
    task_a_acc_after_b = evaluate_accuracy(student, add_test_loader, device)
    task_b_acc = evaluate_accuracy(student, sub_test_loader, device)
    forgetting = task_a_acc_before_b - task_a_acc_after_b
    avg_acc = (task_a_acc_after_b + task_b_acc) / 2.0
    result: dict[str, float | str] = {
        "mode": mode,
        "task_a_acc_before_b": task_a_acc_before_b,
        "task_a_acc_after_b": task_a_acc_after_b,
        "task_b_acc": task_b_acc,
        "forgetting": forgetting,
        "avg_acc": avg_acc,
    }
    result.update(lambda_stats)
    return result


def run_spike_experiment(cfg: SpikeConfig, device: str | torch.device = "cpu") -> list[dict[str, float | str]]:
    device_obj = torch.device(device)
    modes: list[DistillMode] = [
        "ce_only",
        "teacher_only",
        "fixed_kd",
        "confidence_kd",
        "entropy_kd",
        "pi_weighted_kd",
    ]
    return [run_single_mode(cfg, mode, device_obj) for mode in modes]


def to_markdown(results: list[dict[str, float | str]]) -> str:
    headers = [
        "mode",
        "task_a_acc_before_b",
        "task_a_acc_after_b",
        "task_b_acc",
        "forgetting",
        "avg_acc",
        "lambda_mean",
        "lambda_std",
        "lambda_p50",
        "lambda_p90",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in results:
        values: list[str] = []
        for key in headers:
            value = row[key]
            if isinstance(value, str):
                values.append(value)
            else:
                values.append(f"{value:.6f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_results(
    cfg: SpikeConfig,
    results: list[dict[str, float | str]],
    output_json: Path,
    output_md: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": asdict(cfg), "results": results}
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(to_markdown(results), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=31)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs-a", type=int, default=25)
    parser.add_argument("--epochs-b", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fixed-lambda", type=float, default=0.5)
    parser.add_argument("--pi-alpha", type=float, default=1.0)
    parser.add_argument("--pi-gamma", type=float, default=0.5)
    parser.add_argument("--pi-lambda-floor", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-mlp", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-json",
        type=str,
        default="docs/spikes/experiment-pi-weighted-self-distillation-spike-results.json",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="docs/spikes/experiment-pi-weighted-self-distillation-spike-results.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SpikeConfig(
        p=args.p,
        train_frac=args.train_frac,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs_a=args.epochs_a,
        epochs_b=args.epochs_b,
        lr=args.lr,
        fixed_lambda=args.fixed_lambda,
        pi_alpha=args.pi_alpha,
        pi_gamma=args.pi_gamma,
        pi_lambda_floor=args.pi_lambda_floor,
        model=SmallTransformerConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_mlp=args.d_mlp,
            seq_len=args.seq_len,
        ),
    )
    results = run_spike_experiment(cfg, device=args.device)
    save_results(cfg, results, Path(args.output_json), Path(args.output_md))
    print(to_markdown(results))


if __name__ == "__main__":
    main()
