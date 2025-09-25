
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from rich.console import Console
from torch.utils.data import DataLoader

from .config import TrainConfig
from .consistency import ConsistencyTools
from .data import ArcCollator, GridDeserializer, GridSerializer, InMemoryArcDataset
from .evaluation import EvaluationStep
from .model import ArcTransformer, SparseProtoLinear
from .observer import Observer
from .tokenizer import ArcColorTokenizer

torch.set_default_dtype(torch.float32)

@torch.jit.script
def calculate_saps_loss(
    proto_weights: list[torch.Tensor],
    sbl_inputs: list[torch.Tensor],
    raw_weights: list[torch.Tensor],
    mu_surprises: list[torch.Tensor],
    importances: list[torch.Tensor]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    total_proto_loss = torch.tensor(0.0, device=proto_weights[0].device)
    all_saps_masks = []
    for p, x, rw, mu_s, imp in zip(proto_weights, sbl_inputs, raw_weights, mu_surprises, importances):
        N, _ = p.shape
        saps_mask = torch.zeros(N, device=p.device, dtype=torch.int8)
        if x.numel() == 0 or rw.numel() == 0 or mu_s.numel() == 0:
            all_saps_masks.append(saps_mask)
            continue
        B, S, D = x.shape
        x_flat, rw_flat = x.view(B * S, D), rw.view(B * S, N)
        activated_mask = rw_flat > 0
        if not activated_mask.any():
            all_saps_masks.append(saps_mask)
            continue
        activation_rate = activated_mask.float().mean()
        dynamic_factor = activation_rate
        mu_s_float = mu_s.float()
        surprise_q_low = torch.quantile(mu_s_float, dynamic_factor)
        surprise_q_high = torch.quantile(mu_s_float, 1.0 - dynamic_factor)
        is_good, is_bad = mu_s <= surprise_q_low, mu_s >= surprise_q_high
        
        # Only assign status to experts that were actually activated in this batch
        activated_experts_mask = activated_mask.any(dim=0)
        saps_mask[is_good & activated_experts_mask] = 1
        saps_mask[is_bad & activated_experts_mask] = 2
        all_saps_masks.append(saps_mask)
        loss_mask = is_good | is_bad
        if not loss_mask.any():
            continue
        signs = torch.ones(N, device=p.device)
        signs[is_good], signs[is_bad] = -1.0, 1.0
        expert_token_counts = activated_mask.sum(dim=0).clamp(min=1)
        expert_token_sums = torch.matmul(activated_mask.to(x_flat.dtype).t(), x_flat)
        anchors = F.normalize(expert_token_sums / expert_token_counts.unsqueeze(1), p=2.0, dim=-1)
        p_norm = F.normalize(p, p=2.0, dim=-1)
        similarities = (p_norm * anchors).sum(dim=-1)
        imp_normalized = imp / (imp.max() + 1e-6)
        adaptive_strength = imp_normalized[loss_mask] * mu_s[loss_mask]
        proto_loss_per_expert = signs[loss_mask] * (adaptive_strength * (1 - similarities[loss_mask]))
        total_proto_loss += proto_loss_per_expert.mean()
    final_loss = total_proto_loss / len(proto_weights) if len(proto_weights) > 0 else torch.tensor(0.0)
    return final_loss, all_saps_masks

@torch.jit.script
def calculate_gate_loss(
    predicted_costs: list[torch.Tensor],
    mu_surprises: list[torch.Tensor],
    proto_surprises: list[torch.Tensor],
    importances: list[torch.Tensor]
) -> torch.Tensor:
    total_gate_loss = torch.tensor(0.0, device=predicted_costs[0].device)
    num_losses = 0
    for pc, mu_s, proto_s, imp in zip(predicted_costs, mu_surprises, proto_surprises, importances):
        if pc.numel() == 0 or mu_s.numel() == 0 or proto_s.numel() == 0 or imp.numel() == 0:
            continue
        target_surprise = imp * (mu_s + proto_s)
        total_gate_loss += F.mse_loss(pc.mean(dim=(0, 1)), target_surprise.to(pc.dtype))
        num_losses += 1
    return total_gate_loss / num_losses if num_losses > 0 else torch.tensor(0.0)

class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        self.console = Console()
        self.observer = Observer(self.console, config)
        self.tokenizer = ArcColorTokenizer()
        self.serializer = GridSerializer(self.tokenizer)
        self.deserializer = GridDeserializer(self.tokenizer)
        train_dataset = InMemoryArcDataset(data_path=config.data.data_path, split="training")
        eval_dataset = InMemoryArcDataset(data_path=config.data.data_path, split="evaluation")
        train_collator = ArcCollator(self.tokenizer, max_len=config.model.max_position_embeddings)
        self.train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size, collate_fn=train_collator, num_workers=config.data.num_workers, shuffle=False)
        self.eval_loader = DataLoader(eval_dataset, batch_size=1, collate_fn=train_collator, num_workers=config.data.num_workers, shuffle=False)
        self.config.model.vocab_size = self.tokenizer.vocab_size
        self.model = ArcTransformer(self.config.model, device=self.device).to(self.device)
        self._setup_optimizers()
        self.consistency_tools = ConsistencyTools()
        self.evaluator = EvaluationStep(self.model, self.serializer, self.deserializer, self.observer, self.device, train_dataset, self.config)
        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.global_step, self.epoch, self.start_task_idx, self.start_view_idx, self.ema_acc = 0, 0, 0, 0, 0.0

    def _setup_optimizers(self):
        mu_params = [p for name, p in self.model.named_parameters() if 'mu_weight' in name or 'mu_bias' in name]
        proto_params = [p for name, p in self.model.named_parameters() if 'proto_weight' in name]
        gate_params = [p for name, p in self.model.named_parameters() if 'gate_param' in name]
        other_params = [p for name, p in self.model.named_parameters() if not ('mu_' in name or 'proto_' in name or 'gate_' in name)]
        self.optimizer_mu = torch.optim.AdamW(mu_params + other_params, lr=self.config.lr_mu)
        self.optimizer_proto = torch.optim.AdamW(proto_params, lr=self.config.lr_proto)
        self.optimizer_gate = torch.optim.AdamW(gate_params, lr=self.config.lr_gate)

    def _prepare_batch(self, mini_task: dict, view_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        input_grid = torch.tensor(mini_task['input'], device=self.device)
        output_grid = torch.tensor(mini_task['output'], device=self.device)
        transform = self.consistency_tools.get_transforms()[view_idx]
        augmented = {'input': transform(input_grid).cpu().tolist(), 'output': transform(output_grid).cpu().tolist()}
        ids, labels = self.serializer.serialize_mini_task(augmented)
        if len(ids) > self.config.model.max_position_embeddings: return None
        return torch.tensor([ids], dtype=torch.long, device=self.device), torch.tensor([labels], dtype=torch.long, device=self.device)

    def _get_training_signals(self, main_loss: torch.Tensor, model_outputs: dict) -> dict:
        mu_weights = [p for name, p in self.model.named_parameters() if 'mu_weight' in name and p.requires_grad]
        proto_weights = [p for name, p in self.model.named_parameters() if 'proto_weight' in name and p.requires_grad]
        grads = torch.autograd.grad(main_loss, mu_weights + model_outputs["computation_outputs"], retain_graph=True, create_graph=True, allow_unused=True)
        mu_grads, output_grads = grads[:len(mu_weights)], grads[len(mu_weights):]
        mu_surprises = [g.norm(p=2, dim=-1) if g is not None else torch.zeros(p.shape[0], device=p.device) for g, p in zip(mu_grads, mu_weights)]
        importances = [g.norm(p=2, dim=(0, 1)).detach() if g is not None else torch.zeros(p.shape[0], device=p.device) for g, p in zip(output_grads, proto_weights)]
        proto_loss, saps_masks = calculate_saps_loss(proto_weights, model_outputs["sbl_inputs"], model_outputs["raw_weights"], mu_surprises, importances)
        proto_grads = torch.autograd.grad(proto_loss, proto_weights, retain_graph=True, allow_unused=True)
        proto_surprises = [g.norm(p=2, dim=-1).detach() if g is not None else torch.zeros(p.shape[0], device=p.device) for g, p in zip(proto_grads, proto_weights)]
        return {"mu_weights": mu_weights, "proto_weights": proto_weights, "mu_grads": mu_grads, "proto_grads": proto_grads, "mu_surprises": mu_surprises, "proto_surprises": proto_surprises, "importances": importances, "saps_masks": saps_masks, "proto_loss": proto_loss}

    def _apply_proto_protection(self, signals: dict):
        proto_grads = signals["proto_grads"]
        gate_params = [p for name, p in self.model.named_parameters() if 'gate_param' in name and p.requires_grad]
        all_gate_norms = torch.cat([p.norm(p=2, dim=-1).detach() for p in gate_params])
        quality_scores = -all_gate_norms
        temperature = max(1.0 - self.ema_acc, 0.01)
        p_protect = F.softmax(quality_scores / temperature, dim=0)
        protection_mask = 1.0 - p_protect
        start_idx = 0
        for grad in proto_grads:
            if grad is not None:
                num_experts = grad.shape[0]
                end_idx = start_idx + num_experts
                mask_slice = protection_mask[start_idx:end_idx].unsqueeze(1)
                grad.mul_(mask_slice)
                start_idx = end_idx

    def _apply_mu_suppression(self, signals: dict, model_outputs: dict):
        mu_grads = signals["mu_grads"]
        raw_weights = model_outputs["raw_weights"]
        
        if len(mu_grads) != len(raw_weights):
            return

        for grad, rw in zip(mu_grads, raw_weights):
            if grad is not None and rw.numel() > 0:
                activation_rate = (rw > 0).float().mean()
                k_val = 1.0 - activation_rate
                
                threshold = torch.quantile(grad.abs().to(torch.float32), k_val)
                mask = grad.abs() >= threshold
                grad.mul_(mask)

    def _update_parameters(self, signals: dict, model_outputs: dict):
        gate_params = [p for name, p in self.model.named_parameters() if 'gate_param' in name and p.requires_grad]
        gate_loss = calculate_gate_loss(model_outputs["predicted_costs"], [s.detach() for s in signals["mu_surprises"]], [s.detach() for s in signals["proto_surprises"]], signals["importances"])
        gate_grads = torch.autograd.grad(gate_loss, gate_params, allow_unused=True)
        self._apply_proto_protection(signals)
        self._apply_mu_suppression(signals, model_outputs)
        for p, g in zip(signals["mu_weights"], signals["mu_grads"]):
            if g is not None: p.grad = g.detach()
        for p, g in zip(signals["proto_weights"], signals["proto_grads"]):
            if g is not None: p.grad = g.detach()
        for p, g in zip(gate_params, gate_grads):
            if g is not None: p.grad = g
        self.optimizer_mu.step()
        self.optimizer_proto.step()
        self.optimizer_gate.step()
        return {"gate_loss": gate_loss}

    @torch.no_grad()
    def _calculate_metrics(self, logits: torch.Tensor, labels: torch.Tensor, losses: dict, signals: dict, model_outputs: dict) -> dict:
        logits_acc, labels_acc = logits[:, :-1, :], labels[:, 1:]
        mask = labels_acc != -100
        acc = 0.0
        if mask.any():
            active_logits, active_labels = logits_acc[mask], labels_acc[mask]
            acc = (torch.argmax(active_logits, dim=-1) == active_labels).float().mean().item()
        self.ema_acc = self.config.ema_alpha_acc * self.ema_acc + (1 - self.config.ema_alpha_acc) * acc
        pc_flat = torch.cat([pc.detach().float().view(-1) for pc in model_outputs["predicted_costs"] if pc.numel() > 0])
        agg_imp = torch.cat(signals["importances"])
        agg_surp = torch.cat(signals["mu_surprises"])
        pi_score = torch.exp(-1.0 * losses["main_loss"].item() - 0.1 * (agg_imp * agg_surp).mean()).item()
        num_spl, num_layers = 4, self.config.model.num_layers
        act_rates = [torch.cat([rw.view(-1) for rw in model_outputs["raw_weights"][i*num_spl:(i+1)*num_spl] if rw.numel() > 0]).gt(0).float().mean().item() for i in range(num_layers)]
        return {"main_loss": losses["main_loss"].item(), "proto_loss": losses["proto_loss"].item(), "gate_loss": losses["gate_loss"].item(),
                "token_acc": acc, "ema_acc": self.ema_acc, "pi_score": pi_score,
                "tau": -torch.sum(F.softmax(active_logits, dim=-1) * F.log_softmax(active_logits, dim=-1), dim=-1).mean().item() if mask.any() else 0.0,
                "seq_len": float(labels.shape[1]), "activation_rate_avg": sum(act_rates) / len(act_rates) if act_rates else 0.0,
                "activation_rate_l0": act_rates[0] if act_rates else 0.0, "activation_rate_l_mid": act_rates[num_layers // 2] if len(act_rates) > num_layers//2 else 0.0,
                "activation_rate_ln": act_rates[-1] if act_rates else 0.0,
                "raw_top10_gate": pc_flat[pc_flat >= torch.quantile(pc_flat, 0.9)].mean().item() if pc_flat.numel() > 0 else 0.0,
                "raw_avg_gate": pc_flat.mean().item() if pc_flat.numel() > 0 else 0.0, "raw_max_gate": pc_flat.max().item() if pc_flat.numel() > 0 else 0.0}

    def _run_step(self, mini_task: dict, view_idx: int, epoch: int, task_idx: int):
        start_time = time.time()
        batch = self._prepare_batch(mini_task, view_idx)
        if batch is None: return None
        input_ids, labels = batch
        self.model.train()
        self.optimizer_mu.zero_grad()
        self.optimizer_proto.zero_grad()
        self.optimizer_gate.zero_grad()
        logits, _, masked, comp, _, sbl_ins, raw, pred_costs, _ = self.model(input_ids)
        model_outputs = {"masked_outputs": masked, "computation_outputs": comp, "sbl_inputs": sbl_ins, "raw_weights": raw, "predicted_costs": pred_costs}
        main_loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, self.config.model.vocab_size), labels[:, 1:].contiguous().view(-1), ignore_index=-100)
        if not torch.isfinite(main_loss): return None
        signals = self._get_training_signals(main_loss, model_outputs)
        update_info = self._update_parameters(signals, model_outputs)
        losses = {"main_loss": main_loss, "proto_loss": signals["proto_loss"], "gate_loss": update_info["gate_loss"]}
        metrics = self._calculate_metrics(logits, labels, losses, signals, model_outputs)
        if self.global_step % self.config.log_interval == 0:
            self.observer.log_step(epoch, self.global_step, task_idx, metrics, time.time() - start_time)
            self._visualize_and_checkpoint(task_idx, view_idx, signals["saps_masks"])

        if self.global_step > 0 and self.global_step % self.config.eval_interval == 0:
            self.evaluator.run(self.eval_loader, task_idx, self.global_step)

        self.global_step += 1
        torch.cuda.empty_cache()
        return metrics

    def _visualize_and_checkpoint(self, task_idx: int, view_idx: int, saps_masks: list):
        spl_layers = [m for m in self.model.modules() if isinstance(m, SparseProtoLinear)]
        spl_names = ["attn_qkv", "attn_o", "ffn_sbl1", "ffn_sbl2"]
        num_blocks = len(spl_layers) // len(spl_names)
        status_map = {0: "neutral", 1: "good", 2: "bad"}
        saps_data = []
        for i in range(num_blocks):
            block_data = {}
            for j, name in enumerate(spl_names):
                idx = i * len(spl_names) + j
                if idx < len(spl_layers) and idx < len(saps_masks):
                    protos = spl_layers[idx].proto_weight.detach().cpu()
                    statuses = [status_map.get(s.item(), "neutral") for s in saps_masks[idx]]
                    block_data[name] = {"protos": protos, "statuses": statuses}
            saps_data.append(block_data)
        self.observer.visualize_saps_clusters(saps_data, self.global_step)
        self._save_checkpoint(task_idx, view_idx)

    def _train_epoch(self, epoch: int):
        self.model.train()
        dataset = self.train_loader.dataset
        for task_idx in range(self.start_task_idx, len(dataset)):
            mini_task = dataset[task_idx]
            start_view = self.start_view_idx if task_idx == self.start_task_idx else 0
            for view_idx in range(start_view, 8):
                step = 0
                MAX_STEPS = 500
                while step < MAX_STEPS:
                    metrics = self._run_step(mini_task, view_idx, epoch, task_idx)
                    if metrics is None:
                        self.console.print(f"[yellow]Skipping task {task_idx} view {view_idx}.[/yellow]")
                        break
                    if metrics.get("token_acc", 0.0) >= 1.0 and metrics.get("tau", 1.0) <= 0.01:
                        self.console.print(f"Task {task_idx} view {view_idx} converged in {step + 1} steps.")
                        break
                    step += 1
                if step == MAX_STEPS: self.console.print(f"[red]Task {task_idx} view {view_idx} hit MAX_STEPS.[/red]")
            if task_idx == self.start_task_idx: self.start_view_idx = 0
        self.start_task_idx = 0

    def train(self):
        self._load_checkpoint()
        self.console.print("[bold green]Starting Training...[/bold green]")
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            self._train_epoch(epoch)
        self.console.print("[bold green]Training Finished.[/bold green]")

    def _save_checkpoint(self, task_idx: int, view_idx: int):
        state = {'epoch': self.epoch, 'step': self.global_step, 'task_idx': task_idx, 'view_idx': view_idx,
                 'model_state_dict': self.model.state_dict(), 'optimizer_mu_state_dict': self.optimizer_mu.state_dict(),
                 'optimizer_proto_state_dict': self.optimizer_proto.state_dict(), 'optimizer_gate_state_dict': self.optimizer_gate.state_dict(),
                 'ema_acc': self.ema_acc}
        path = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        torch.save(state, path)
        ckpts = sorted(self.checkpoint_dir.glob("*.pt"), key=os.path.getmtime)
        if len(ckpts) > self.config.max_checkpoints: os.remove(ckpts[0])

    def _load_checkpoint(self):
        ckpts = sorted(self.checkpoint_dir.glob("*.pt"), key=os.path.getmtime, reverse=True)
        if not ckpts: self.console.print("[bold yellow]No checkpoint found.[/bold yellow]"); return
        for path in ckpts:
            try:
                ckpt = torch.load(path, map_location=self.device)
                self.model.load_state_dict(ckpt['model_state_dict'])
                self.optimizer_mu.load_state_dict(ckpt['optimizer_mu_state_dict'])
                self.optimizer_proto.load_state_dict(ckpt['optimizer_proto_state_dict'])
                self.optimizer_gate.load_state_dict(ckpt['optimizer_gate_state_dict'])
                self.global_step = ckpt.get('step', 0)
                self.epoch = ckpt.get('epoch', 0)
                self.start_task_idx = ckpt.get('task_idx', 0)
                self.start_view_idx = ckpt.get('view_idx', 0)
                self.ema_acc = ckpt.get('ema_acc', 0.0)
                self.console.print(f"[bold green]Loaded checkpoint from {path} at step {self.global_step}.[/bold green]")
                return
            except (RuntimeError, KeyError, EOFError) as e:
                self.console.print(f"[bold red]Corrupted checkpoint {path}: {e}. Trying next.[/bold red]")
                os.remove(path)
        self.console.print("[bold yellow]No valid checkpoint found.[/bold yellow]")

def main():
    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
