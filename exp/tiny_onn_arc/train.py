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
from .model import ArcTransformer, DynMoE
from .observer import Observer
from .tokenizer import ArcChatMLTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)

        self.console = Console()
        self.observer = Observer(self.console, config)

        self.tokenizer = ArcChatMLTokenizer()
        self.serializer = GridSerializer(self.tokenizer)
        self.deserializer = GridDeserializer(self.tokenizer)

        train_dataset = ArcDataset(data_path=config.data.data_path, split="training")
        eval_dataset = ArcDataset(data_path=config.data.data_path, split="evaluation")
        collator = ArcCollator(tokenizer=self.tokenizer)

        self.train_loader = DataLoader(
            train_dataset, batch_size=config.data.batch_size, collate_fn=collator,
            num_workers=config.data.num_workers, shuffle=True
        )
        self.eval_loader = DataLoader(
            eval_dataset, batch_size=1, collate_fn=collator, num_workers=config.data.num_workers, shuffle=True
        )

        model_config_dict = {
            "vocab_size": self.tokenizer.vocab_size,
            "hidden_size": config.model.hidden_size,
            "num_hidden_layers": config.model.num_layers,
            "num_attention_heads": config.model.num_attention_heads,
            "num_key_value_heads": config.model.num_key_value_heads,
            "max_position_embeddings": config.model.max_position_embeddings,
            "rope_theta": config.model.rope_theta,
            "rms_norm_eps": config.model.rms_norm_eps,
            "attention_dropout": config.model.attention_dropout,
            "pad_token_id": self.tokenizer.pad_token_id,
            "moe": config.model.moe
        }
        qwen3_config = Qwen3Config(**model_config_dict)
        self.model = ArcTransformer(qwen3_config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.consistency_tools = ConsistencyTools()
        
        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.global_step = 0
        self.last_grid_acc = 0.0

    def _save_checkpoint(self):
        state = {
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        filepath = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        torch.save(state, filepath)
        self.console.print(f"[bold green]Saved checkpoint to {filepath}[/bold green]")

        checkpoints = sorted(self.checkpoint_dir.glob("*.pt"), key=os.path.getmtime)
        if len(checkpoints) > self.config.max_checkpoints:
            os.remove(checkpoints[0])

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

    def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    def _calculate_token_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        masked_labels = labels.view(-1)
        active_logits = logits.view(-1, logits.size(-1))[masked_labels != -100]
        active_labels = masked_labels[masked_labels != -100]
        
        if active_labels.numel() == 0:
            return torch.tensor(0.0)

        preds = torch.argmax(active_logits, dim=-1)
        return (preds == active_labels).float().mean()

    def _calculate_loss_and_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        main_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        token_acc = self._calculate_token_accuracy(logits, labels)
        pred_ids = torch.argmax(logits, dim=-1)
        
        correct_grids = 0
        for i in range(logits.shape[0]):
            true_answer_mask = labels[i] != -100
            if not torch.any(true_answer_mask): continue
            true_answer = labels[i][true_answer_mask]
            
            pred_answer_full = pred_ids[i]
            if pred_answer_full.shape[0] >= true_answer_mask.shape[0]:
                pred_answer = pred_answer_full[true_answer_mask]
            else: continue
            if torch.equal(true_answer, pred_answer):
                correct_grids += 1
        
        grid_acc = torch.tensor(correct_grids / logits.shape[0])
        return main_loss, token_acc, grid_acc

    def _run_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        outputs = self.model(**batch)
        logits = outputs.logits
        total_loss = outputs.loss
        
        _, token_acc, grid_acc = self._calculate_loss_and_metrics(logits, batch["labels"])

        metrics = { "main_loss": total_loss, "token_acc": token_acc, "grid_acc": grid_acc }
        if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
             metrics["moe_avg_k"] = outputs.router_logits.item()

        return total_loss, metrics

    def _train_epoch(self, epoch: int):
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            start_time = time.time()
            
            effective_batch = self._prepare_batch(batch)
            loss, metrics = self._run_step(effective_batch)

            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                elapsed_time = time.time() - start_time
                self.observer.log_step(epoch, self.global_step, metrics, elapsed_time, self.last_grid_acc)
                
                if self.global_step > 0 and self.global_step % self.config.save_interval == 0:
                    self.evaluate(quick_eval=True)
                    self._save_checkpoint()
                self.global_step += 1
        
        self.console.print(f"[bold yellow]End of Epoch {epoch}.[/bold yellow]")
    
    @torch.no_grad()
    def evaluate(self, quick_eval: bool = True):
        self.model.eval()
        num_samples_to_eval = 10 if quick_eval and len(self.eval_loader) > 10 else len(self.eval_loader)
        eval_title = "Quick Eval" if quick_eval else "Full Eval"
        self.console.print(f"\n[bold cyan]--- Running {eval_title} ({num_samples_to_eval} samples) ---[/bold cyan]")
        total_grid_acc, evaluated_count = 0, 0
        
        for i, batch in enumerate(itertools.islice(self.eval_loader, num_samples_to_eval)):
            input_grid_raw = torch.tensor(batch["input_grids"][0])
            target_grid_raw = torch.tensor(batch["output_grids"][0])
            
            effective_batch = self._prepare_batch(batch)
            problem_ids = effective_batch["input_ids"]

            target_len = len(batch["output_grids"][0]) * len(batch["output_grids"][0][0]) + len(batch["output_grids"][0])
            generated_ids = self.model.generate(
                input_ids=problem_ids,
                max_new_tokens=target_len + 10, eos_token_id=self.tokenizer.eos_token_id
            )
            pred_grid = self.deserializer.deserialize(generated_ids[0].tolist())
            
            is_correct = 1 if pred_grid is not None and torch.equal(pred_grid, target_grid_raw) else 0
            total_grid_acc += is_correct
            evaluated_count += 1
            
            if i == 0:
                self.observer.visualize_evaluation_sample(input_grid_raw, target_grid_raw, pred_grid, self.global_step)
        
        avg_grid_acc = total_grid_acc / evaluated_count if evaluated_count > 0 else 0
        self.last_grid_acc = avg_grid_acc
        self.observer.log_eval_summary({"grid_acc": avg_grid_acc, "total_count": evaluated_count}, self.global_step)
        self.model.train()

    def train(self):
        self._load_checkpoint()
        self.console.print("[bold green]Starting Training...[/bold green]")
        for epoch in range(self.config.num_epochs):
            self._train_epoch(epoch)
        self.console.print("[bold green]Training Finished.[/bold green]")

def main():
    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
