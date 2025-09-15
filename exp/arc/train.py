import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.nn as nn
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
        base_params = [p for name, p in self.model.named_parameters() if 'mu_' not in name and 'proto_' not in name]

        self.optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': self.config.base_learning_rate},
            {'params': mu_params, 'lr': self.config.base_learning_rate},
            {'params': proto_params, 'lr': self.config.proto_learning_rate}
        ], weight_decay=self.config.weight_decay)


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


    def _calculate_ibs_loss(self, main_loss: torch.Tensor, block_outputs: list[torch.Tensor], ffn_inputs: list[torch.Tensor], layer_taus: torch.Tensor) -> torch.Tensor:
        intermediate_tensors = block_outputs + ffn_inputs
        
        grads = torch.autograd.grad(main_loss, intermediate_tensors, retain_graph=True, allow_unused=True)
        
        surprise_norms = [g.norm(p=2) for g in grads if g is not None]
        
        if not surprise_norms:
            return torch.tensor(0.0, device=self.device)
            
        surprise_loss = torch.stack(surprise_norms).mean()
        entropy_loss = layer_taus.mean()
        
        return surprise_loss + self.config.entropy_loss_weight * entropy_loss


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
        self.optimizer.zero_grad(set_to_none=True)

        logits, _, _, raw_weights, _, proto_weights, layer_taus, block_outputs, ffn_inputs, block_raw_weights, _ = self.model(input_ids)
        main_loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, self.config.model.vocab_size), labels[:, 1:].contiguous().view(-1), ignore_index=-100)
        
        ibs_loss = self._calculate_ibs_loss(main_loss, block_outputs, ffn_inputs, layer_taus)
        total_loss = main_loss + self.config.ibs_loss_weight * ibs_loss

        if torch.isfinite(total_loss):
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for group in self.optimizer.param_groups for p in group['params']], 1.0)
            self.optimizer.step()

        complexity_cost = 0.0
        for p_group in self.optimizer.param_groups:
            for p in p_group['params']:
                if p.grad is not None:
                    complexity_cost += p.grad.norm(p=2).item() ** 2
        complexity_cost = complexity_cost ** 0.5
        
        with torch.no_grad():
            seq_len = input_ids.shape[1]
            token_counts = torch.bincount(input_ids.flatten(), minlength=self.tokenizer.vocab_size)
            token_probs = token_counts / seq_len
            token_probs = token_probs[token_probs > 0]
            seq_entropy = -torch.sum(token_probs * torch.log2(token_probs)).item()

            token_acc = self._calculate_token_accuracy(logits, labels).item()
            pi_score = torch.exp(-(main_loss + complexity_cost)).item()
            avg_tau = layer_taus.mean().item()
            proto_norm = torch.mean(torch.stack([p.norm(p=2) for p in proto_weights])).item() if proto_weights else 0.0
            avg_proto = torch.mean(torch.stack([p.mean() for p in proto_weights])).item() if proto_weights else 0.0
            
            activation_rate_avg = torch.cat([rw.view(-1) for rw in raw_weights]).gt(0).float().mean().item() if raw_weights else 0.0
            
            # Layer-wise activation rates
            act_rate_l0 = torch.cat([rw.view(-1) for rw in block_raw_weights[0]]).gt(0).float().mean().item() if block_raw_weights else 0.0
            act_rate_ln = torch.cat([rw.view(-1) for rw in block_raw_weights[-1]]).gt(0).float().mean().item() if block_raw_weights else 0.0


            metrics = {
                "main_loss": main_loss.item(), "ibs_loss": ibs_loss.item(),
                "token_acc": token_acc, "pi_score": pi_score, "avg_tau": avg_tau, "avg_proto": avg_proto, "proto_norm": proto_norm,
                "activation_rate_avg": activation_rate_avg,
                "activation_rate_l0": act_rate_l0,
                "activation_rate_ln": act_rate_ln,
                "seq_len": float(seq_len), "seq_entropy": seq_entropy,
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
