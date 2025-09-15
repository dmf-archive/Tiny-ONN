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

        mu_params = [p for name, p in self.model.named_parameters() if 'mu_weight' in name or 'mu_bias' in name]
        proto_params = [p for name, p in self.model.named_parameters() if 'proto_weight' in name]
        gate_params = [p for name, p in self.model.named_parameters() if 'gate_param' in name]
        base_params = [p for name, p in self.model.named_parameters() if 'mu_' not in name and 'proto_' not in name and 'gate_' not in name]

        self.optimizer_main = torch.optim.AdamW(
            [{'params': base_params}, {'params': mu_params}],
            lr=self.config.base_learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.optimizer_proto = torch.optim.AdamW(proto_params, lr=self.config.proto_learning_rate)
        self.optimizer_gate = torch.optim.AdamW(gate_params, lr=self.config.gate_learning_rate)

        self.consistency_tools = ConsistencyTools()
        self.evaluator = EvaluationStep(self.model, self.serializer, self.deserializer, self.observer, self.device)

        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.global_step = 0

    def _save_checkpoint(self):
        state = {
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_main_state_dict': self.optimizer_main.state_dict(),
            'optimizer_proto_state_dict': self.optimizer_proto.state_dict(),
            'optimizer_gate_state_dict': self.optimizer_gate.state_dict(),
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
        self.optimizer_main.load_state_dict(checkpoint['optimizer_main_state_dict'])
        self.optimizer_proto.load_state_dict(checkpoint['optimizer_proto_state_dict'])
        self.optimizer_gate.load_state_dict(checkpoint['optimizer_gate_state_dict'])
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

    def _inner_loop_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        last_meta_loss = torch.tensor(float('inf'))
        patience_counter = 0
        final_mu_surprise_loss = torch.tensor(0.0, device=self.device)
        final_diversity_loss = torch.tensor(0.0, device=self.device)

        for _ in range(self.config.inner_loop_max_steps):
            self.optimizer_proto.zero_grad(set_to_none=True)
            self.optimizer_gate.zero_grad(set_to_none=True)

            logits, _, _, _, mu_weights, proto_weights, _, _ = self.model(input_ids)
            main_loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, self.config.model.vocab_size), labels[:, 1:].contiguous().view(-1), ignore_index=-100)

            if not torch.isfinite(main_loss):
                break

            mu_grads = torch.autograd.grad(main_loss, mu_weights, retain_graph=True, allow_unused=True)
            
            mu_surprises = [g.norm(p=2) for g in mu_grads if g is not None]
            mu_surprise_loss = torch.stack(mu_surprises).mean() if mu_surprises else torch.tensor(0.0, device=self.device)
            
            diversity_loss = self._calculate_diversity_loss(proto_weights)
            meta_loss = self.config.mu_surprise_loss_weight * mu_surprise_loss + self.config.diversity_loss_weight * diversity_loss

            if torch.isfinite(meta_loss):
                meta_loss.backward()
                self.optimizer_proto.step()
                self.optimizer_gate.step()

            if torch.abs(last_meta_loss - meta_loss).item() < self.config.inner_loop_convergence_tolerance:
                patience_counter += 1
            else:
                patience_counter = 0
            
            if patience_counter >= self.config.inner_loop_patience:
                break
            
            last_meta_loss = meta_loss.detach()
            final_mu_surprise_loss = mu_surprise_loss.detach()
            final_diversity_loss = diversity_loss.detach()
        
        return {"mu_surprise_loss": final_mu_surprise_loss, "diversity_loss": final_diversity_loss}

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

        self.model.train()
        inner_loop_results = self._inner_loop_step(input_ids, labels)
        
        self.optimizer_main.zero_grad(set_to_none=True)
        
        logits, _, _, raw_weights, _, _, layer_taus, _ = self.model(input_ids)
        main_loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, self.config.model.vocab_size), labels[:, 1:].contiguous().view(-1), ignore_index=-100)
        task_loss = main_loss

        if torch.isfinite(task_loss):
            task_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.optimizer_main.param_groups[0]['params'], 1.0)
            torch.nn.utils.clip_grad_norm_(self.optimizer_main.param_groups[1]['params'], 1.0)
            self.optimizer_main.step()

        with torch.no_grad():
            token_acc = self._calculate_token_accuracy(logits, labels).item()
            pi_score = torch.exp(-main_loss).item()
            avg_tau = layer_taus.mean().item()
            all_gates = [p for name, p in self.model.named_parameters() if 'gate_param' in name]
            avg_gate = torch.mean(torch.stack([g.mean() for g in all_gates])).item() if all_gates else 0.0
            all_protos = [p for name, p in self.model.named_parameters() if 'proto_weight' in name]
            avg_proto = torch.mean(torch.stack([p.mean() for p in all_protos])).item() if all_protos else 0.0
            activation_rate = torch.cat([rw.view(-1) for rw in raw_weights]).gt(0).float().mean().item() if raw_weights else 0.0

            metrics = {
                "main_loss": main_loss.item(), "mu_surprise_loss": inner_loop_results["mu_surprise_loss"].item(),
                "div_loss": inner_loop_results["diversity_loss"].item(),
                "token_acc": token_acc, "pi_score": pi_score, "avg_tau": avg_tau, "avg_gate": avg_gate, "avg_proto": avg_proto,
                "activation_rate": activation_rate
            }

        elapsed_time = time.time() - start_time
        if self.global_step % self.config.log_interval == 0:
            self.observer.log_step(epoch, self.global_step, metrics, elapsed_time)
            self._save_checkpoint()

        if self.global_step > 0 and self.global_step % self.config.eval_interval == 0:
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
        self.optimizer_main.zero_grad()
        self.optimizer_proto.zero_grad()
        self.optimizer_gate.zero_grad()
        for epoch in range(self.config.num_epochs):
            self._train_epoch(epoch)
        self.console.print("[bold green]Training Finished.[/bold green]")

def main():
    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
