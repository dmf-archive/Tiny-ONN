import torch
import torch.nn as nn
import time
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from .config import TrainConfig
from .data import get_arc_dataloader, ARCTokenizer, ARCProcessor
from .observer import ARCObserver
from .evaluator import ArcEvaluator
from .shaper import RoutingShaper
from ...optimizers.ars2_neo import SingleDeviceARS2Neo

class ARCTrainer:
    def __init__(self, model: nn.Module, config: TrainConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        self.tokenizer = ARCTokenizer()
        self.processor = ARCProcessor(self.tokenizer)
        self.observer = ARCObserver()
        self.evaluator = ArcEvaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            processor=self.processor,
            observer=self.observer,
            device=config.device
        )
        self.shaper = RoutingShaper(w_meta=0.1)
        
        self.train_loader = get_arc_dataloader(config.data_dir, config.batch_size, split="training")
        self.eval_loader = get_arc_dataloader(config.data_dir, 1, split="evaluation")
        
        self.optimizer = self._setup_optimizer()
        self.global_step = 0
        self.checkpoint_dir = Path("checkpoints/arc_v2")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.last_routing_diagnostics = {}
        self.performance_stats = {}

    def _setup_optimizer(self) -> SingleDeviceARS2Neo:
        ars2_params = []
        adamw_params = []
        for p in self.model.parameters():
            if p.requires_grad:
                if p.ndim >= 2:
                    ars2_params.append(p)
                else:
                    adamw_params.append(p)
        
        param_groups = [
            {"params": ars2_params, "is_rmsuon_group": True},
            {"params": adamw_params, "is_rmsuon_group": False}
        ]
        
        return SingleDeviceARS2Neo(
            param_groups,
            lr=self.config.lr,
            weight_decay=0.01,
            rho=0.1,
            k=1  # Synchronous SAM by default
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        start_time = time.perf_counter()
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        diff_mask = batch.get("diff_mask")
        if diff_mask is not None:
            diff_mask = diff_mask.to(self.device)
        
        last_outputs = {}

        def closure():
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, labels=labels, diff_mask=diff_mask)
            
            if isinstance(outputs, dict):
                main_loss = outputs["loss"]
                routing_info = outputs.get("routing_info")
                logits = outputs.get("logits")
            elif hasattr(outputs, "loss"):
                main_loss = outputs.loss
                routing_info = getattr(outputs, "routing_info", None)
                logits = getattr(outputs, "logits", None)
            else:
                raise ValueError(f"Unsupported model output type: {type(outputs)}")
            
            last_outputs['logits'] = logits.detach() if logits is not None else None

            if routing_info:
                # FARS: Fisher-Aware Routing Shaping (Module-level)
                meta_loss = self.shaper.calculate_meta_loss(routing_info, model=self.model, optimizer=self.optimizer)
                total_loss = main_loss + meta_loss
                self.last_routing_diagnostics = {
                    **self.shaper.get_routing_diagnostics(routing_info),
                    "meta_loss": meta_loss.item(),
                    "shaper_ms": self.shaper.performance_stats.get('shaper_calc_time_ms', 0)
                }
            else:
                total_loss = main_loss
                self.last_routing_diagnostics = {}
                
            return total_loss

        loss = self.optimizer.step(closure)
        self.global_step += 1
        
        # Calculate metrics using observer
        if last_outputs.get('logits') is not None:
            step_metrics = self.observer.calculate_metrics(
                loss=loss.item(),
                logits=last_outputs['logits'],
                labels=labels,
                optimizer_diagnostics=self.optimizer.diagnostics
            )
            self.performance_stats.update(step_metrics)

        self.performance_stats['step_ms'] = (time.perf_counter() - start_time) * 1000
        if self.device.type == 'cuda':
            self.performance_stats['gpu_mem_mb'] = torch.cuda.memory_allocated(self.device) / 1024**2
            
        return loss.item()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_correct = 0
        num_tasks = min(len(self.eval_loader), 10)
        
        dataset = self.eval_loader.dataset
        for i in range(num_tasks):
            task_data = dataset.tasks[i]
            is_correct = self.evaluator.evaluate_task(task_data, self.config.generation)
            if is_correct:
                total_correct += 1
        
        accuracy = total_correct / num_tasks
        self.observer.log_metrics({"eval_accuracy": accuracy}, step=self.global_step)
        return accuracy

    def save_checkpoint(self, epoch: int):
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, path)

    def train(self):
        self.observer.console.print(f"[bold green]Starting ARC-2 Training on {self.device}...[/bold green]")
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(self.train_loader):
                loss = self.train_step(batch)
                epoch_loss += loss
                
                if step % 10 == 0:
                    self.observer.log_metrics(
                        {
                            "loss": loss, 
                            **self.optimizer.diagnostics, 
                            **self.last_routing_diagnostics,
                            **self.performance_stats
                        }, 
                        step=self.global_step, 
                        epoch=epoch
                    )
            
            avg_loss = epoch_loss / len(self.train_loader)
            self.observer.log_metrics({"avg_epoch_loss": avg_loss}, step=self.global_step, epoch=epoch)
            
            self.evaluate()
            self.save_checkpoint(epoch)
