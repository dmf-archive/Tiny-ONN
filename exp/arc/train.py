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
from .data import ArcCollator, ArcDataset, GridDeserializer, GridSerializer
from .evaluation import EvaluationStep
from .model import ArcTransformer
from .observer import Observer
from .tokenizer import ArcColorTokenizer


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

        train_dataset = ArcDataset(data_path=config.data.data_path, split="training")
        eval_dataset = ArcDataset(data_path=config.data.data_path, split="evaluation")

        train_collator = ArcCollator(self.tokenizer, max_len=config.model.max_position_embeddings)
        eval_collator = ArcCollator(self.tokenizer, max_len=config.model.max_position_embeddings)

        self.train_loader = DataLoader(
            train_dataset, batch_size=config.data.batch_size, collate_fn=train_collator,
            num_workers=config.data.num_workers, shuffle=True
        )
        self.eval_loader = DataLoader(
            eval_dataset, batch_size=1, collate_fn=eval_collator, num_workers=config.data.num_workers, shuffle=False
        )

        self.config.model.vocab_size = self.tokenizer.vocab_size
        self.model = ArcTransformer(self.config.model, device=self.device).to(self.device)

        # SPLv2: Unified optimizer for global loss optimization
        param_groups = [
            {'params': [p for name, p in self.model.named_parameters() if 'mu_' in name or 'embedding' in name or 'lm_head' in name or 'ln' in name], 'lr': config.base_learning_rate},
            {'params': [p for name, p in self.model.named_parameters() if 'proto_weight' in name], 'lr': config.proto_learning_rate},
            {'params': [p for name, p in self.model.named_parameters() if 'gate_param' in name], 'lr': config.gate_learning_rate}
        ]
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

        self.consistency_tools = ConsistencyTools()
        self.evaluator = EvaluationStep(self.model, self.serializer, self.deserializer, self.observer, self.device)

        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.global_step = 0

    def _save_checkpoint(self):
        state = {
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        filepath = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        torch.save(state, filepath)
        checkpoints = sorted(self.checkpoint_dir.glob("*.pt"), key=os.path.getmtime)
        if len(checkpoints) > self.config.max_checkpoints: os.remove(checkpoints[0])

    def _load_checkpoint(self):
        checkpoints = sorted(self.checkpoint_dir.glob("*.pt"), key=os.path.getmtime, reverse=True)
        if not checkpoints:
            self.console.print("[bold yellow]No checkpoint found, starting from scratch.[/bold yellow]")
            return
        latest_checkpoint = checkpoints[0]
        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['step']
        self.console.print(f"[bold blue]Loaded checkpoint from {latest_checkpoint} at step {self.global_step}[/bold blue]")

    def _calculate_token_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]

        masked_labels = labels.contiguous().view(-1)
        active_mask = masked_labels != -100
        if not active_mask.any():
            return torch.tensor(0.0)

        active_logits = logits.contiguous().view(-1, logits.size(-1))[active_mask]
        active_labels = masked_labels[active_mask]

        preds = torch.argmax(active_logits, dim=-1)
        return (preds == active_labels).float().mean()

    def _calculate_diversity_loss(self, proto_weights: list[torch.Tensor]) -> torch.Tensor:
        if not proto_weights: return torch.tensor(0.0, device=self.device)
        losses = [1.0 - F.cosine_similarity(p.unsqueeze(0), p.unsqueeze(1), dim=-1).abs().mean() for p in proto_weights if p.dim() > 1]
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)

    def _calculate_kl_proto_loss(self, proto_weights: list[torch.Tensor], tau: torch.Tensor) -> torch.Tensor:
        if not proto_weights: return torch.tensor(0.0, device=self.device)

        kl_losses = []
        tau_sq = tau.square().clamp(min=1e-9)

        for p in proto_weights:
            if p.dim() > 1:
                p_norm_sq = p.norm(p=2, dim=1).square().clamp(min=1e-9)
                # Formula from SPL.md: 0.5 * [ (1 / (||p||² * τ²)) - 1 + ln(τ² * ||p||²) ]
                kl_div = 0.5 * ( (1 / (p_norm_sq * tau_sq)) - 1 + torch.log(tau_sq * p_norm_sq) )
                kl_losses.append(kl_div.mean())

        return torch.stack(kl_losses).mean() if kl_losses else torch.tensor(0.0, device=self.device)

    def _calculate_sml_loss(self, main_loss: torch.Tensor, computation_outputs: list[torch.Tensor]) -> torch.Tensor:
        if not computation_outputs or not torch.isfinite(main_loss):
            return torch.tensor(0.0, device=self.device)

        surprise_grads = torch.autograd.grad(main_loss, computation_outputs, retain_graph=True, allow_unused=True)

        sml_components = []
        for grad in surprise_grads:
            if grad is not None:
                surprise_per_neuron = grad.norm(p=2, dim=-1)
                clamped_surprise = surprise_per_neuron.clamp(min=1e-9)
                sml_loss_per_neuron = -clamped_surprise.log() * clamped_surprise
                sml_components.append(sml_loss_per_neuron.mean())

        return torch.stack(sml_components).mean() if sml_components else torch.tensor(0.0, device=self.device)

    def _run_step(self, task_data: dict[str, Any], view_idx: int, epoch: int):
        start_time = time.time()
        transforms = self.consistency_tools.get_transforms()
        input_grid = torch.tensor(task_data['test'][0]['input'])
        output_grid = torch.tensor(task_data['test'][0]['output'])
        input_grid_aug, output_grid_aug = transforms[view_idx](input_grid), transforms[view_idx](output_grid)

        view_task_data = {'train': task_data['train'], 'test': [{'input': input_grid_aug.tolist(), 'output': output_grid_aug.tolist()}]}
        input_ids_list, labels_list = self.serializer.serialize_task_with_context(view_task_data)

        if len(input_ids_list) > self.config.model.max_position_embeddings: return

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)
        labels = torch.tensor([labels_list], dtype=torch.long, device=self.device)

        logits, _, computation_outputs, raw_weights, proto_weights, layer_taus, _ = self.model(input_ids)

        main_loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, self.config.model.vocab_size), labels[:, 1:].contiguous().view(-1), ignore_index=-100)

        sml_loss = self._calculate_sml_loss(main_loss, computation_outputs)
        diversity_loss = self._calculate_diversity_loss(proto_weights)

        avg_tau = layer_taus.mean().detach()
        kl_proto_loss = self._calculate_kl_proto_loss(proto_weights, avg_tau)

        total_loss = (main_loss +
                      self.config.gate_loss_weight * sml_loss +
                      self.config.diversity_loss_weight * diversity_loss +
                      self.config.kl_loss_weight * kl_proto_loss)

        if torch.isfinite(total_loss):
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        with torch.no_grad():
            token_acc = self._calculate_token_accuracy(logits, labels).item()
            pi_score = torch.exp(-main_loss).item()
            all_gates = [p for name, p in self.model.named_parameters() if 'gate_param' in name]
            avg_gate = torch.mean(torch.stack([g.mean() for g in all_gates])).item() if all_gates else 0.0
            activation_rate = torch.cat([rw.view(-1) for rw in raw_weights]).gt(0).float().mean().item() if raw_weights else 0.0

            metrics = {
                "main_loss": main_loss.item(), "sml_loss": sml_loss.item(),
                "div_loss": diversity_loss.item(), "kl_loss": kl_proto_loss.item(),
                "token_acc": token_acc, "pi_score": pi_score, "avg_tau": avg_tau.item(),
                "avg_gate": avg_gate, "activation_rate": activation_rate
            }

        elapsed_time = time.time() - start_time
        if self.global_step % self.config.log_interval == 0:
            self.observer.log_step(epoch, self.global_step, metrics, elapsed_time)

        if self.global_step > 0 and self.global_step % self.config.eval_interval == 0:
            self._save_checkpoint()
            self.evaluator.run(self.eval_loader, self.global_step, quick_eval=True)

        self.global_step += 1
        torch.cuda.empty_cache()

    def _train_epoch(self, epoch: int):
        self.model.train()
        for _i, batch in enumerate(self.train_loader):
            if not batch: continue
            task_data = batch['task_data'][0]
            for view_idx in range(8):
                self._run_step(task_data, view_idx, epoch)

        self.console.print(f"[bold yellow]End of Epoch {epoch}.[/bold yellow]")

    def train(self):
        self._load_checkpoint()
        self.console.print("[bold green]Starting Training...[/bold green]")
        self.optimizer.zero_grad()
        for epoch in range(self.config.num_epochs):
            self._train_epoch(epoch)
        self.console.print("[bold green]Training Finished.[/bold green]")

def main():
    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
