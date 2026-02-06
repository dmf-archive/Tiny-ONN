from __future__ import annotations

import math
import random
from dataclasses import dataclass
import argparse
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.dynsiha.recursive.configuration_recursive_dynsiha import RecursiveDynSIHAConfig
from src.models.dynsiha.recursive.modeling_recursive_dynsiha import RecursiveDynSIHAForCausalLM
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
        seq_len=2,
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
    pairs = torch.stack([aa.flatten(), bb.flatten()], dim=1)
    labels = (pairs[:, 0] + pairs[:, 1]) % p
    perm = torch.randperm(pairs.size(0), generator=g)
    pairs = pairs[perm]
    labels = labels[perm]
    split = int(pairs.size(0) * train_frac)
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
    def __init__(self, cfg: ModelConfig, vocab: int, steps: int, use_sia: bool):
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
            use_sia=use_sia,
            use_act_inference=False,
        )
        self.model = RecursiveDynSIHAForCausalLM(config)
        self._steps = steps

    @property
    def steps(self) -> int:
        return self._steps

    def forward_bundle(self, ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        input_ids = torch.zeros(ids.size(0), 3, device=ids.device, dtype=ids.dtype)
        input_ids[:, :2] = ids
        label_seq = None
        if labels is not None:
            label_seq = torch.full_like(input_ids, -100)
            label_seq[:, 2] = labels
        out = self.model(input_ids=input_ids, labels=label_seq, use_cache=True, return_dict=True)
        loss_stack = out.all_step_losses
        halt_logits = out.halt_logits
        halt_step = None
        if halt_logits is not None:
            halt_step = halt_logits.mean(-1).transpose(0, 1)
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
    """
    routing_weights: [B, E] 每样本每专家权重
    step_idx: [B] 每样本的 step id (0..num_steps-1)
    """
    B, E = routing_weights.shape
    # ITJD: 两两 step 之间的 Jaccard 距离
    step_mask = F.one_hot(step_idx, num_classes=num_steps).bool()  # [B, T]
    step_weights = torch.zeros(num_steps, E, device=routing_weights.device)
    for t in range(num_steps):
        mask = step_mask[:, t]
        if mask.any():
            step_weights[t] = routing_weights[mask].mean(0)
    # 只计算有样本的 step
    occupied = step_weights.sum(1) > 0
    if occupied.sum() < 2:
        return 0.0, 0.0
    w = step_weights[occupied]
    # Jaccard 距离矩阵
    inter = torch.minimum(w.unsqueeze(0), w.unsqueeze(1)).sum(-1)
    union = torch.maximum(w.unsqueeze(0), w.unsqueeze(1)).sum(-1)
    jaccard = inter / (union + 1e-9)
    itjd = (1 - jaccard).triu(1).sum() / (jaccard.triu(1).numel() + 1e-9)
    # RMI: I(step; routing)
    # 离散化路由分布：取 top-1 专家索引
    top1 = routing_weights.argmax(-1)
    # 联合分布
    joint = torch.zeros(num_steps, E, device=routing_weights.device)
    for t in range(num_steps):
        mask = step_idx == t
        if mask.any():
            cnt = torch.bincount(top1[mask], minlength=E).float()
            joint[t] = cnt / (cnt.sum() + 1e-9)
    p_t = joint.sum(1)
    p_e = joint.sum(0)
    # MI = sum p(t,e) log(p(t,e)/(p(t)p(e)))
    mi = 0.0
    for t in range(num_steps):
        for e in range(E):
            p_te = joint[t, e].item()
            if p_te > 0:
                mi += p_te * math.log(p_te / (p_t[t].item() * p_e[e].item() + 1e-9) + 1e-9)
    return float(itjd), float(mi)


def compute_itjd_rmi(
    routing_weights: torch.Tensor,
    step_idx: torch.LongTensor,
    num_steps: int,
) -> tuple[float, float]:
    """
    routing_weights: [B, E] 每样本每专家权重
    step_idx: [B] 每样本的 step id (0..num_steps-1)
    """
    B = routing_weights.size(0)
    E = routing_weights.size(-1)
    # ITJD: 两两 step 之间的 Jaccard 距离
    step_mask = F.one_hot(step_idx, num_classes=num_steps).bool()  # [B, T]
    # routing_weights: [T, B, E] -> 按 step 聚合
    step_weights = torch.zeros(num_steps, E, device=routing_weights.device)
    for t in range(num_steps):
        w_t = routing_weights[t]  # [B, E]
        step_weights[t] = w_t.mean(dim=0)
    # 只计算有样本的 step
    occupied = step_weights.sum(1) > 0
    if occupied.sum() < 2:
        return 0.0, 0.0
    w = step_weights[occupied]
    # Jaccard 距离矩阵
    inter = torch.minimum(w.unsqueeze(0), w.unsqueeze(1)).sum(-1)
    union = torch.maximum(w.unsqueeze(0), w.unsqueeze(1)).sum(-1)
    jaccard = inter / (union + 1e-9)
    itjd = (1 - jaccard).triu(1).sum() / (jaccard.triu(1).numel() + 1e-9)
    # RMI: I(step; routing)
    # 离散化路由分布：取 top-1 专家索引
    # routing_weights: [T, B, E] -> 按 step 聚合
    top1 = routing_weights.argmax(-1)  # [T, B]
    joint = torch.zeros(num_steps, E, device=routing_weights.device)
    for t in range(num_steps):
        cnt = torch.bincount(top1[t], minlength=E).float()
        joint[t] = cnt / (cnt.sum() + 1e-9)
    p_t = joint.sum(1)
    p_e = joint.sum(0)
    # MI = sum p(t,e) log(p(t,e)/(p(t)p(e)))
    mi = 0.0
    for t in range(num_steps):
        for e in range(E):
            p_te = joint[t, e].item()
            if p_te > 0:
                mi += p_te * math.log(p_te / (p_t[t].item() * p_e[e].item() + 1e-9) + 1e-9)
    return float(itjd), float(mi)


def compute_loss_components(
    model: nn.Module,
    ids: torch.Tensor,
    labels: torch.Tensor,
    act_weight: float,
) -> tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
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
        halt_logits = torch.zeros(ids.size(0), model.steps, device=ids.device)
    steps = loss_stack.size(0)
    if steps == 1:
        total_loss = loss_stack[0].mean()
        eff_k = None
        stats = {
            "plsd_losses": [float(loss_stack[0].mean().item())],
            "halt_probs": [float(torch.sigmoid(halt_logits).mean().item())],
            "eff_K": eff_k,
        }
        return total_loss, stats, loss_stack, halt_logits
    if halt_logits.size(1) != steps:
        halt_logits = halt_logits[:, :steps]
    best_loss, best_step = loss_stack.min(0)
    step_mask = torch.arange(steps, device=ids.device)[None, :] <= best_step[:, None]
    main_loss = (loss_stack.permute(1, 0) * step_mask).sum() / step_mask.sum()
    target = (torch.arange(steps, device=ids.device)[None, :] >= best_step[:, None]).float()
    act_loss = F.binary_cross_entropy_with_logits(halt_logits, target)
    total_loss = main_loss + act_weight * act_loss
    oracle_dist = torch.bincount(best_step, minlength=steps).float().cpu()
    oracle_dist = (oracle_dist / oracle_dist.sum()).tolist()
    itjd, rmi = 0.0, 0.0
    if routing_weights is not None:
        itjd, rmi = compute_itjd_rmi(routing_weights, best_step, steps)
    stats = {
        "eff_L": float(best_step.float().mean().item()),
        "eff_K": eff_k,
        "oracle_dist": oracle_dist,
        "halt_probs": torch.sigmoid(halt_logits).mean(0).detach().cpu().tolist(),
        "plsd_losses": loss_stack.mean(1).detach().cpu().tolist(),
        "itjd": itjd,
        "rmi": rmi,
    }
    return total_loss, stats, loss_stack, halt_logits


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
    if model.steps == 1:
        return logits
    if halt_logits is None:
        return logits
    logits = torch.stack(logits_list).permute(1, 0, 2)
    halt_probs = torch.sigmoid(halt_logits)
    mask = halt_probs > 0.5
    first = mask.float().argmax(1)
    has_halt = mask.any(1)
    selected = torch.where(has_halt, first, torch.full_like(first, model.steps - 1))
    gathered = logits.gather(1, selected[:, None, None].expand(-1, 1, logits.size(-1)))
    return gathered.squeeze(1)


class StatsCollector:
    def __init__(self, name: str, is_rds: bool):
        self.name = name
        self.is_rds = is_rds
        self.loss: list[float] = []
        self.acc: list[float] = []
        self.eff_L: list[float] = []
        self.eff_K: list[float] = []
        self.oracle_dist: list[list[float]] = []
        self.halt_probs: list[list[float]] = []
        self.plsd_losses: list[list[float]] = []
        self.phi_s: list[list[float]] = []
        self.itjd: list[float] = []
        self.rmi: list[float] = []

    def collect(self, loss: float, acc: float, stats: dict, phi_s: list[float]) -> None:
        self.loss.append(loss)
        self.acc.append(acc)
        if self.is_rds:
            if "eff_L" in stats:
                self.eff_L.append(float(stats["eff_L"]))
            if "eff_K" in stats:
                if stats["eff_K"] is not None:
                    self.eff_K.append(float(stats["eff_K"]))
            if "oracle_dist" in stats:
                self.oracle_dist.append([float(x) for x in stats["oracle_dist"]])
            if "halt_probs" in stats:
                self.halt_probs.append([float(x) for x in stats["halt_probs"]])
            if "plsd_losses" in stats:
                self.plsd_losses.append([float(x) for x in stats["plsd_losses"]])
            if len(phi_s) > 0:
                self.phi_s.append([float(x) for x in phi_s])
            if "itjd" in stats:
                self.itjd.append(float(stats["itjd"]))
            if "rmi" in stats:
                self.rmi.append(float(stats["rmi"]))

    def summarize(self, step: int, log_interval: int) -> str:
        window = min(len(self.loss), log_interval)
        loss_avg = sum(self.loss[-window:]) / window
        acc_avg = sum(self.acc[-window:]) / window
        base = f"[{self.name}] Step {step:04d} | Loss: {loss_avg:.4f} | Acc: {acc_avg:.2%}"
        if not self.is_rds or len(self.eff_L) == 0:
            return base
        eff_l = sum(self.eff_L[-window:]) / window
        eff_k = sum(self.eff_K[-window:]) / window if self.eff_K else 0.0
        oracle = [sum(x) / window for x in zip(*self.oracle_dist[-window:])]
        halt = [sum(x) / window for x in zip(*self.halt_probs[-window:])]
        plsd = [sum(x) / window for x in zip(*self.plsd_losses[-window:])]
        phi = [sum(x) / window for x in zip(*self.phi_s[-window:])] if self.phi_s else []
        itjd = sum(self.itjd[-window:]) / window if self.itjd else 0.0
        rmi = sum(self.rmi[-window:]) / window if self.rmi else 0.0
        return (
            f"{base} | eff_L: {eff_l:.2f} | eff_K: {eff_k:.2f}\n"
            f"  Oracle: {format_list(oracle)} | Halt: {format_list(halt)}\n"
            f"  PLSD: {format_list(plsd)} | Phi: {format_list(phi)} | ITJD: {itjd:.3f} | RMI: {rmi:.3f}"
        )


def format_list(values: list[float]) -> str:
    if len(values) == 0:
        return "[]"
    return "[" + ", ".join(f"{v:.2f}" for v in values) + "]"


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
    stats = StatsCollector(name, model.steps > 1)
    train_iter = iter(train_loader)
    test_iter = iter(test_loader)
    for step in range(cfg.training.max_steps):
        try:
            ids, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            ids, labels = next(train_iter)
        ids = ids.to(device)
        labels = labels.to(device)

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            total_loss, _, _, _ = compute_loss_components(model, ids, labels, cfg.model.act_weight)
            return total_loss

        loss = optimizer.step(closure)

        if step % cfg.training.log_interval == 0 or step == cfg.training.max_steps - 1:
            model.eval()
            with torch.enable_grad():
                total_loss, stats_dict, loss_stack, _ = compute_loss_components(
                    model, ids, labels, cfg.model.act_weight
                )
                phi_s = compute_phi_s(model, loss_stack, total_loss)
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                logits = infer_logits(model, ids)
                preds = logits.argmax(dim=-1)
                acc = float((preds == labels).float().mean().item())
            stats.collect(float(loss.item()), acc, stats_dict, phi_s)
            print(stats.summarize(step, cfg.training.log_interval))
            model.train()
            if acc >= cfg.training.early_stop_acc:
                break


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


def run(skip_dense: bool) -> None:
    set_seed(CONFIG.data.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_ds = build_mod_addition(CONFIG.data.p, CONFIG.data.train_frac, CONFIG.data.seed)
    train_loader = DataLoader(train_ds, batch_size=CONFIG.training.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG.training.batch_size, shuffle=True)
    configs = [("RDS", CONFIG.model.n_steps, CONFIG.model.n_layers_rds, False)]
    if not skip_dense:
        configs.insert(0, ("Dense", 1, CONFIG.model.n_layers_dense, False))
    for name, steps, layers, use_sia in configs:
        if name == "Dense":
            model = ModAddTransformer(CONFIG.model, CONFIG.data.p, steps, layers, use_sia).to(device)
        else:
            model = RecursiveDynSIHATrunk(CONFIG.model, CONFIG.data.p, steps, use_sia).to(device)
        optimizer = build_optimizer(model, CONFIG.training.lr)
        print(f"\n>>> Starting {name} Experiment")
        run_experiment(name, model, optimizer, train_loader, test_loader, CONFIG, device)
        print(f"--- {name} FINISHED ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dense", action="store_true", default=True)
    args = parser.parse_args()
    run(args.skip_dense)
