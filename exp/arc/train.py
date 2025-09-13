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
        self.prior_std = 1.0

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
        # self.model = torch.jit.script(self.model)

        # Separate parameter groups for differential learning rates
        prior_params = []
        base_params = []

        for name, param in self.model.named_parameters():
            if 'sigma_weight' in name or 'gate_param' in name:
                prior_params.append(param)
            else:
                base_params.append(param)

        param_groups = [
            {'params': base_params, 'lr': self.config.base_learning_rate, 'weight_decay': self.config.weight_decay},
            {'params': prior_params, 'lr': self.config.prior_learning_rate, 'weight_decay': self.config.weight_decay}
        ]

        self.optimizer = torch.optim.AdamW(param_groups)
        self.consistency_tools = ConsistencyTools()
        self.evaluator = EvaluationStep(self.model, self.serializer, self.deserializer, self.observer, self.device)

        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.global_step = 0

    def _save_checkpoint(self):
        state = {
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
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

    def _run_tf_step(self, task_data: dict[str, Any], view_idx: int) -> tuple[torch.Tensor, dict[str, Any]] | None:
        kl_epsilon = self.config.kl_prior_epsilon

        input_grid = torch.tensor(task_data['test'][0]['input'])
        output_grid = torch.tensor(task_data['test'][0]['output'])

        transforms = self.consistency_tools.get_transforms()
        input_grid_aug = transforms[view_idx](input_grid)
        output_grid_aug = transforms[view_idx](output_grid)

        view_task_data = {'train': task_data['train'], 'test': [{'input': input_grid_aug.tolist(), 'output': output_grid_aug.tolist()}]}
        input_ids_list, labels_list = self.serializer.serialize_task_with_context(view_task_data)

        if len(input_ids_list) > self.config.model.max_position_embeddings:
            return None

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)
        labels = torch.tensor([labels_list], dtype=torch.long, device=self.device)

        logits, masked_outputs, computation_outputs, raw_weights, kl_loss, _, _ = self.model(
            input_ids, self.prior_std, kl_epsilon
        )

        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        main_loss = F.cross_entropy(
            shifted_logits.contiguous().view(-1, self.config.model.vocab_size),
            shifted_labels.contiguous().view(-1), ignore_index=-100
        )

        sml_loss = torch.tensor(0.0, device=self.device)
        if computation_outputs:
            active_elements = sum((rw > 1e-5).float().sum().item() for rw in raw_weights)
            total_elements = sum(rw.numel() for rw in raw_weights)
            activation_rate = active_elements / total_elements if total_elements > 0 else 0.0

            masked_grads = torch.autograd.grad(
                main_loss, computation_outputs, retain_graph=True, allow_unused=True
            )

            for grad_tensor in masked_grads:
                if grad_tensor is not None:
                    surprise_per_neuron = grad_tensor.view(-1, grad_tensor.shape[-1]).norm(p=2, dim=0)
                    active_surprise = surprise_per_neuron[surprise_per_neuron > 1e-9]
                    if active_surprise.numel() > 0:
                        weighted_surprise = active_surprise * activation_rate
                        sml_loss += (-torch.log(weighted_surprise + 1e-9) * weighted_surprise).sum()

        with torch.no_grad():
            avg_tau = torch.distributions.Categorical(logits=logits).entropy().mean()

        w_sml = -torch.log(avg_tau + kl_epsilon)
        total_loss = main_loss + kl_loss + w_sml * sml_loss

        # --- Metric Calculation ---
        with torch.no_grad():
            token_acc = self._calculate_token_accuracy(logits, labels).item()
            active_elements = sum((rw > 0).float().sum().item() for rw in raw_weights)
            total_elements = sum(rw.numel() for rw in raw_weights)
            activation_rate = active_elements / total_elements if total_elements > 0 else 0.0

            pi_score = torch.exp(-(main_loss + kl_loss + sml_loss)).item()

            metrics = {
                "main_loss": main_loss.item(),
                "sml_loss": sml_loss.item(),
                "kl_loss": kl_loss.item(),
                "total_loss": total_loss.item(),
                "token_acc": token_acc,
                "activation_rate": activation_rate,
                "avg_tau": avg_tau.item(),
                "prior_std": self.prior_std,
                "pi_score": pi_score
            }
            all_sigmas = [p for name, p in self.model.named_parameters() if 'sigma_weight' in name]
            metrics["avg_sigma"] = (torch.mean(torch.stack([F.softplus(s).mean() for s in all_sigmas])).item() if all_sigmas else 0.0)
            all_gates = [p for name, p in self.model.named_parameters() if 'gate_param' in name]
            metrics["avg_gate"] = (torch.mean(torch.stack([g.mean() for g in all_gates])).item() if all_gates else 0.0)

        return total_loss, metrics

    def _train_epoch(self, epoch: int):
        self.model.train()
        for _i, batch in enumerate(self.train_loader):
            if not batch: continue

            task_data = batch['task_data'][0]

            for view_idx in range(8): # Treat each view as a separate training step
                start_time = time.time()

                step_output = self._run_tf_step(task_data, view_idx)
                if step_output is None:
                    continue

                loss, metrics = step_output

                if loss.item() == 0.0: continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Update prior_std based on the dual-layer mechanism
                avg_sigma = metrics.get('avg_sigma', 0.0)
                avg_gate = metrics.get('avg_gate', 0.0)
                beta = torch.sigmoid(torch.tensor(avg_gate - avg_sigma)).item()
                current_tau = metrics.get('avg_tau', self.prior_std)
                self.prior_std = beta * self.prior_std + (1 - beta) * current_tau

                torch.cuda.empty_cache()
                elapsed_time = time.time() - start_time

                if self.global_step > 0 and self.global_step % self.config.log_interval == 0:
                    self.observer.log_step(epoch, self.global_step, metrics, elapsed_time)
                    self._save_checkpoint()

                if self.global_step > 0 and (self.global_step % self.config.eval_interval == 0):
                    self.evaluator.run(self.eval_loader, self.global_step, quick_eval=True)

                self.global_step += 1

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
