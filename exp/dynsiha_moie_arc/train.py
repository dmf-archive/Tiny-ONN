import glob
import os
import time
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from rich.console import Console
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import itertools

from .config import TrainConfig
from .consistency import ConsistencyTools
from .data import ArcCollator, ArcDataset, GridDeserializer, GridSerializer
from .model import ArcTransformer
from .observer import Observer
from .tokenizer import ArcPositionalTokenizer


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)

        self.console = Console()
        self.observer = Observer(self.console, config)
        self.prior_std = 1.0

        self.tokenizer = ArcPositionalTokenizer()
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
        self.model = ArcTransformer(self.config.model).to(self.device)
        
        sigma_params = [p for name, p in self.model.named_parameters() if 'sigma_weight' in name]
        other_params = [p for name, p in self.model.named_parameters() if 'sigma_weight' not in name]

        optimizer_grouped_parameters = [
            {'params': other_params},
            {'params': sigma_params, 'lr': config.kl_learning_rate}
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.consistency_tools = ConsistencyTools()
        
        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.global_step = 0

    def _save_checkpoint(self):
        state = {'step': self.global_step, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}
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
        masked_labels = labels.view(-1)
        active_mask = masked_labels != -100
        if not active_mask.any():
            return torch.tensor(0.0)
        active_logits = logits.view(-1, logits.size(-1))[active_mask]
        active_labels = masked_labels[active_mask]
        preds = torch.argmax(active_logits, dim=-1)
        return (preds == active_labels).float().mean()

    def _run_tf_step(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        task_data = batch['task_data'][0]
        
        input_grids_aug = self.consistency_tools.apply_transforms(torch.tensor(task_data['test'][0]['input']))
        output_grids_aug = self.consistency_tools.apply_transforms(torch.tensor(task_data['test'][0]['output']))
        
        # Pre-calculate global tau from all views for a stable prior
        all_logits_no_grad = []
        with torch.no_grad():
            for i in range(8):
                view_task_data = { 'train': task_data['train'], 'test': [{'input': input_grids_aug[i].tolist(), 'output': output_grids_aug[i].tolist()}] }
                input_ids_list, _ = self.serializer.serialize_task_with_context(view_task_data)
                if len(input_ids_list) > self.config.model.max_position_embeddings: continue
                input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)
                logits, _, _ = self.model(input_ids, 1.0, self.config.kl_prior_epsilon)
                all_logits_no_grad.append(logits)
        
        if not all_logits_no_grad: return torch.tensor(0.0), {}

        with torch.no_grad():
            avg_tau = torch.mean(torch.stack([torch.distributions.Categorical(logits=l).entropy().mean() for l in all_logits_no_grad]))
            self.prior_std = torch.clamp(avg_tau.detach(), min=0.01, max=3.0).item()
        
        # Grouped gradient accumulation
        num_groups = 4
        group_size = 8 // num_groups
        total_loss_for_backward = torch.tensor(0.0, device=self.device)
        
        # Metrics to aggregate across groups
        agg_metrics = {"main_loss": 0.0, "gate_loss": 0.0, "kl_loss": 0.0, "total_loss": 0.0, "token_acc": 0.0, "activation_rate": 0.0}

        for group_idx in range(num_groups):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size
            
            group_main_loss = torch.tensor(0.0, device=self.device)
            group_kl_loss = torch.tensor(0.0, device=self.device)
            group_masked_outputs = []
            group_active_elements, group_total_elements = 0, 0
            group_logits, group_labels = [], []

            for i in range(start_idx, end_idx):
                view_task_data = { 'train': task_data['train'], 'test': [{'input': input_grids_aug[i].tolist(), 'output': output_grids_aug[i].tolist()}] }
                input_ids_list, labels_list = self.serializer.serialize_task_with_context(view_task_data)
                if len(input_ids_list) > self.config.model.max_position_embeddings: continue

                input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)
                labels = torch.tensor([labels_list], dtype=torch.long, device=self.device)

                logits, masked_outputs, kl_loss = self.model(input_ids, self.prior_std, self.config.kl_prior_epsilon)
                
                group_main_loss += F.cross_entropy(logits.view(-1, self.config.model.vocab_size), labels.view(-1), ignore_index=-100)
                group_kl_loss += kl_loss
                group_masked_outputs.extend(masked_outputs)
                group_logits.append(logits)
                group_labels.append(labels)

                with torch.no_grad():
                    for mo in masked_outputs:
                        group_active_elements += (mo.abs() > 1e-5).float().sum()
                        group_total_elements += mo.numel()

            if not group_masked_outputs: continue

            group_activation_rate = (group_active_elements / group_total_elements).item() if group_total_elements > 0 else 0.0
            
            group_gate_loss = torch.tensor(0.0, device=self.device)
            surprise_grads = torch.autograd.grad(group_main_loss, group_masked_outputs, retain_graph=True, allow_unused=True)
            for grad_tensor in surprise_grads:
                if grad_tensor is not None:
                    surprise_per_neuron = grad_tensor.view(-1, grad_tensor.shape[-1]).norm(p=2, dim=0)
                    active_surprise = surprise_per_neuron[surprise_per_neuron > 1e-9]
                    if active_surprise.numel() > 0:
                        weighted_surprise = active_surprise * group_activation_rate
                        group_gate_loss += (-torch.log(weighted_surprise + 1e-9) * weighted_surprise).sum()

            with torch.no_grad():
                w_gate = -torch.log(avg_tau.detach() + self.config.kl_prior_epsilon)
            
            group_total_loss = group_main_loss + w_gate * group_gate_loss + group_kl_loss
            total_loss_for_backward += group_total_loss

            # Aggregate metrics
            with torch.no_grad():
                agg_metrics["main_loss"] += group_main_loss.item()
                agg_metrics["gate_loss"] += (w_gate * group_gate_loss).item()
                agg_metrics["kl_loss"] += group_kl_loss.item()
                agg_metrics["total_loss"] += group_total_loss.item()
                agg_metrics["token_acc"] += torch.mean(torch.stack([self._calculate_token_accuracy(l, la) for l, la in zip(group_logits, group_labels)])).item()
                agg_metrics["activation_rate"] += group_activation_rate

        # Normalize aggregated metrics
        for key in agg_metrics: agg_metrics[key] /= num_groups
        
        with torch.no_grad():
            all_sigmas = [p for name, p in self.model.named_parameters() if 'sigma_weight' in name]
            all_gates = [p for name, p in self.model.named_parameters() if 'gate_param' in name]
            agg_metrics["avg_sigma"] = torch.mean(torch.stack([F.softplus(s).mean() for s in all_sigmas])).item() if all_sigmas else 0
            agg_metrics["avg_gate"] = torch.mean(torch.stack([g.mean() for g in all_gates])).item() if all_gates else 0
            agg_metrics["avg_tau"] = avg_tau.item()
            agg_metrics["prior_std"] = self.prior_std

        return total_loss_for_backward, agg_metrics

    def _train_epoch(self, epoch: int):
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            if not batch: continue
            
            start_time = time.time()
            
            loss, metrics = self._run_tf_step(batch)

            if loss.item() == 0.0: continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            torch.cuda.empty_cache() # Clean cache after each full step
            elapsed_time = time.time() - start_time
            
            if self.global_step > 0 and self.global_step % self.config.log_interval == 0:
                self.observer.log_step(epoch, self.global_step, metrics, elapsed_time)
            
            if self.global_step > 0 and self.global_step % self.config.eval_interval == 0:
                self.evaluate(quick_eval=True)
                self._save_checkpoint()

            self.global_step += 1
        
        self.console.print(f"[bold yellow]End of Epoch {epoch}.[/bold yellow]")
    
    @torch.no_grad()
    def evaluate(self, quick_eval: bool = True):
        self.model.eval()
        torch.cuda.empty_cache() # Clean cache before evaluation
        eval_title = "Quick Eval" if quick_eval and len(self.eval_loader) > 10 else "Full Eval"
        num_samples_to_eval = 10 if quick_eval and len(self.eval_loader) > 10 else len(self.eval_loader)
        self.console.print(f"\n[bold cyan]--- Running {eval_title} ({num_samples_to_eval} samples) ---[/bold cyan]")
        
        total_grid_acc, evaluated_count = 0, 0
        visualized = False
        
        for i, batch in enumerate(itertools.islice(self.eval_loader, num_samples_to_eval)):
            if not batch: continue

            task_data = batch['task_data'][0]
            input_grid_raw = torch.tensor(task_data['test'][0]['input'])
            target_grid_raw = torch.tensor(task_data['test'][0]['output'])

            prompt_ids = self.serializer.serialize_for_inference(task_data)
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            
            h, w = target_grid_raw.shape
            max_new_tokens = int(h * w * 1.5) + 30

            generated_ids = self.model.generate(
                input_ids=prompt_tensor,
                max_new_tokens=max_new_tokens,
                top_p=0.05,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            pred_grid = self.deserializer.deserialize(generated_ids[0].tolist())

            if not visualized:
                self.observer.visualize_evaluation_sample(input_grid_raw, target_grid_raw, pred_grid, self.global_step)
                visualized = True
            
            is_correct = 1 if pred_grid is not None and torch.equal(pred_grid, target_grid_raw) else 0
            total_grid_acc += is_correct
            evaluated_count += 1

        avg_grid_acc = total_grid_acc / evaluated_count if evaluated_count > 0 else 0

        eval_metrics = {
            "eval_grid_acc": avg_grid_acc,
            "total_count": float(evaluated_count)
        }
        self.observer.log_eval_summary(eval_metrics, self.global_step)
        self.model.train()

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
