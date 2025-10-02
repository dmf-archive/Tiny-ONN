import collections
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from torch.utils.data import DataLoader

from .config import GenerationConfig, TrainConfig
from .consistency import ConsistencyTools
from .data import ArcCollator, GridDeserializer, GridSerializer, InMemoryArcDataset
from .evaluation import EvaluationStep
from .model import ArcTransformer, mas_normalize
from .observer import Observer
from .tokenizer import ArcColorTokenizer


@torch.jit.script
def _jsd_from_distributions(p_dist_unnorm: torch.Tensor, q_dist_unnorm: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-9
    p_dist = p_dist_unnorm / (p_dist_unnorm.sum(dim=-1, keepdim=True) + epsilon)
    q_dist = q_dist_unnorm / (q_dist_unnorm.sum(dim=-1, keepdim=True) + epsilon)
    m_dist = 0.5 * (p_dist + q_dist)
    kl_p_m = torch.sum(p_dist * (torch.log(p_dist + epsilon) - torch.log(m_dist + epsilon)), dim=-1)
    kl_q_m = torch.sum(q_dist * (torch.log(q_dist + epsilon) - torch.log(m_dist + epsilon)), dim=-1)
    return (0.5 * kl_p_m + 0.5 * kl_q_m).mean()


def _get_task_representation(tensors: list[torch.Tensor]) -> torch.Tensor | None:
    if not tensors or any(t.ndim != 3 for t in tensors) or tensors[0].shape[0] > 1:
        return None

    matrix = torch.cat(tensors, dim=2).squeeze(0).float()

    if matrix.numel() == 0:
        return None

    if matrix.shape[0] < 2:
        mean_vec = matrix.mean(dim=0)
        std_vec = torch.zeros_like(mean_vec)
    else:
        mean_vec = matrix.mean(dim=0)
        std_vec = matrix.std(dim=0, unbiased=False)

    return torch.cat([mean_vec, std_vec], dim=0)


class LearningDynamics:
    def __init__(
        self,
        config: TrainConfig,
        computation_params: list,
        routing_params_with_names: list[tuple[str, nn.Parameter]],
        optimizer_comp: torch.optim.Optimizer,
        optimizer_route: torch.optim.Optimizer,
    ):
        self.config = config
        self.computation_params = computation_params
        self.routing_params_with_names = routing_params_with_names
        self.routing_params = [p for _, p in routing_params_with_names]
        self.optimizer_comp = optimizer_comp
        self.optimizer_route = optimizer_route

    @staticmethod
    @torch.jit.script
    def _calculate_jsd_loss(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        p_dist_unnorm = mas_normalize(p_logits)
        q_dist_unnorm = mas_normalize(q_logits).detach()
        return _jsd_from_distributions(p_dist_unnorm, q_dist_unnorm)

    @staticmethod
    @torch.jit.script
    def _calculate_mu_surprise_norms(
        computation_output_grads: list[torch.Tensor], spl_inputs: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        mu_surprise_norms = []
        for grad_act, spl_in in zip(computation_output_grads, spl_inputs):
            grad_act_sq = torch.sum(grad_act**2, dim=(0, 1))
            spl_in_sq = torch.sum(spl_in**2, dim=(0, 1))
            
            term1 = (spl_in_sq * grad_act_sq.sum()).sqrt()
            term2 = (grad_act_sq * spl_in_sq.sum()).sqrt()
            
            norm = (term1 + term2) / 2.0
            mu_surprise_norms.append(norm)
        return mu_surprise_norms

    def compute_and_apply_gradients(
        self, main_loss: torch.Tensor, model_outputs: dict, device: torch.device, last_routing_logits: list[torch.Tensor] | None, last_spl_inputs: list[torch.Tensor] | None
    ) -> dict[str, Any]:
        self.optimizer_comp.zero_grad()
        self.optimizer_route.zero_grad()

        computation_outputs = model_outputs["computation_outputs"]
        clean_spl_inputs = [s for s in model_outputs["spl_inputs"] if s is not None]

        params_to_grad = self.computation_params + computation_outputs
        all_grads = torch.autograd.grad(main_loss, params_to_grad, retain_graph=True, allow_unused=True)
        comp_grads, intermediate_grads = all_grads[: len(self.computation_params)], all_grads[len(self.computation_params) :]

        with torch.no_grad():
            for param, grad in zip(self.computation_params, comp_grads):
                if grad is not None:
                    param.grad = grad.clone()

        valid_intermediate_grads = [g for g in intermediate_grads if g is not None]
        mu_surprise_norms = self._calculate_mu_surprise_norms(valid_intermediate_grads, clean_spl_inputs)
        
        with torch.no_grad():
            all_goodness = [
                F.relu(mas_normalize(torch.norm(output_grad, p=2, dim=(0, 1))) - mas_normalize(mu_surprise))
                for mu_surprise, output_grad in zip(mu_surprise_norms, valid_intermediate_grads)
            ]

        meta_losses = [
            self._calculate_jsd_loss(logit, good)
            for logit, good in zip(model_outputs["routing_logits"], all_goodness)
            if logit.numel() > 0 and good.numel() > 0 and logit.shape[-1] == good.shape[-1]
        ]
        avg_route_jsd_loss = torch.stack(meta_losses).mean() if meta_losses else torch.tensor(0.0, device=device)
        total_meta_loss = self.config.w_route_jsd * avg_route_jsd_loss

        avg_carc_loss = torch.tensor(0.0, device=device)
        if last_routing_logits and last_spl_inputs:
            reps = [
                _get_task_representation([t for t in tensor_list if t is not None])
                for tensor_list in [model_outputs["routing_logits"], last_routing_logits, clean_spl_inputs, last_spl_inputs]
            ]
            if all(isinstance(rep, torch.Tensor) for rep in reps):
                sim_p = F.cosine_similarity(reps[0].unsqueeze(0), reps[1].unsqueeze(0))
                sim_x = F.cosine_similarity(reps[2].unsqueeze(0), reps[3].unsqueeze(0))
                avg_carc_loss = torch.abs(sim_p - sim_x.detach()).mean()
                total_meta_loss += self.config.w_carc * avg_carc_loss

        if total_meta_loss > 0:
            meta_grads = torch.autograd.grad(total_meta_loss, self.routing_params, allow_unused=True)
            with torch.no_grad():
                for param, grad in zip(self.routing_params, meta_grads):
                    if grad is not None:
                        param.grad = grad.clone()

        torch.nn.utils.clip_grad_norm_(self.computation_params, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.routing_params, max_norm=1.0)
        self.optimizer_comp.step()
        self.optimizer_route.step()

        return {
            "route_jsd_loss": avg_route_jsd_loss, "carc_loss": avg_carc_loss, "mu_surprises": mu_surprise_norms,
            "goodness_scores": all_goodness, "proto_weights": [p.detach() for name, p in self.routing_params_with_names if "proto_weight" in name],
        }


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config, self.device = config, torch.device(config.device)
        torch.manual_seed(config.seed)
        self.console = Console()
        self.observer = Observer(self.console, config)
        self.tokenizer, self.serializer = ArcColorTokenizer(), GridSerializer(ArcColorTokenizer())
        self.consistency_tools = ConsistencyTools()
        self._setup_data()
        self._setup_model_and_optimizer()
        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.global_step, self.epoch, self.start_task_idx, self.start_view_idx = 0, 0, 0, 0
        self.log_cycle_goodness: list[list[torch.Tensor]] = []
        self.log_cycle_act_rates: list[list[float]] = []
        self.last_spl_inputs: list[torch.Tensor] | None = None
        self.last_routing_logits: list[torch.Tensor] | None = None
        self.replay_queue: collections.deque = collections.deque(maxlen=100)
        self.new_sample_counter = 0

    def _setup_data(self):
        collator = ArcCollator(self.tokenizer, max_len=self.config.model.max_position_embeddings)
        self.train_loader = DataLoader(
            InMemoryArcDataset(data_path=self.config.data.data_path, split="training"),
            batch_size=self.config.data.batch_size, collate_fn=collator, num_workers=self.config.data.num_workers, shuffle=False,
        )
        self.eval_loader = DataLoader(
            InMemoryArcDataset(data_path=self.config.data.data_path, split="evaluation"),
            batch_size=1, collate_fn=collator, num_workers=self.config.data.num_workers, shuffle=False,
        )

    def _setup_model_and_optimizer(self):
        self.config.model.vocab_size = self.tokenizer.vocab_size
        self.model = ArcTransformer(self.config.model, device=self.device).to(self.device)
        computation_params = [p for name, p in self.model.named_parameters() if "proto_weight" not in name and "gate_param" not in name]
        routing_params_with_names = [(name, p) for name, p in self.model.named_parameters() if "proto_weight" in name or "gate_param" in name]
        self.optimizer_comp = torch.optim.AdamW(computation_params, lr=self.config.lr)
        self.optimizer_route = torch.optim.AdamW([p for _, p in routing_params_with_names], lr=self.config.lr)
        self.dynamics = LearningDynamics(self.config, computation_params, routing_params_with_names, self.optimizer_comp, self.optimizer_route)
        self.evaluator = EvaluationStep(self.model, self.serializer, GridDeserializer(self.tokenizer), self.observer, self.device, self.train_loader.dataset, self.config)

    @staticmethod
    def _prepare_batch(task_data: dict, view_idx: int, device: torch.device, serializer: GridSerializer, consistency_tools: ConsistencyTools, max_len: int) -> dict[str, torch.Tensor] | None:
        
        original_train = task_data["train"]
        original_test = task_data["test"]
        
        transformed_train = []
        for pair in original_train:
            input_grid = torch.tensor(pair["input"], device=device)
            output_grid = torch.tensor(pair["output"], device=device)
            transform = consistency_tools.get_transforms()[view_idx]
            transformed_train.append({
                "input": transform(input_grid).cpu().tolist(),
                "output": transform(output_grid).cpu().tolist(),
            })

        test_input_grid = torch.tensor(original_test[0]["input"], device=device)
        test_output_grid = torch.tensor(original_test[0]["output"], device=device)
        transform = consistency_tools.get_transforms()[view_idx]
        transformed_test = [{
            "input": transform(test_input_grid).cpu().tolist(),
            "output": transform(test_output_grid).cpu().tolist(),
        }]

        augmented_task = {"train": transformed_train, "test": transformed_test}
        
        ids, labels, coords = serializer.serialize_task(augmented_task)
        if len(ids) > max_len: return None
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=device), "labels": torch.tensor([labels], dtype=torch.long, device=device),
            "coords": torch.tensor([coords], dtype=torch.long, device=device),
            "sample_entropy": torch.tensor([ArcCollator._calculate_sample_entropy(labels)], dtype=torch.float32, device=device),
        }

    def _train_step(self, batch: dict, epoch: int, task_idx: int | str, view_idx: int, last_view_routing_logits: list | None) -> tuple | None:
        start_time = time.time()
        self.model.train()
        with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16):
            model_outputs = self.model(batch["input_ids"], coords=batch["coords"], return_dict=True)
            main_loss = F.cross_entropy(model_outputs["logits"][:, :-1, :].contiguous().view(-1, self.config.model.vocab_size), batch["labels"][:, 1:].contiguous().view(-1), ignore_index=-100)

        if not torch.isfinite(main_loss):
            self.console.print(f"[bold red]NaN detected in main_loss at step {self.global_step}. Aborting step.[/bold red]")
            return None

        signals = self.dynamics.compute_and_apply_gradients(main_loss, model_outputs, self.device, self.last_routing_logits, self.last_spl_inputs)
        model_outputs.update({"labels": batch["labels"], "sample_entropy": batch["sample_entropy"]})
        metrics = self.observer.calculate_metrics(main_loss, model_outputs, signals, batch["input_ids"], self.model)

        if "goodness_scores" in signals and signals["goodness_scores"]: self.log_cycle_goodness.append(signals["goodness_scores"])
        if "act_rates" in metrics: self.log_cycle_act_rates.append(metrics["act_rates"])

        consistency_metrics = self._calculate_consistency_metrics(model_outputs.get("routing_logits"), last_view_routing_logits, model_outputs.get("spl_inputs"), self.last_spl_inputs)

        signals["raw_weights"] = model_outputs.get("raw_weights")
        self.observer.maybe_log_and_visualize(
            epoch, self.global_step, task_idx if isinstance(task_idx, int) else -1, view_idx, metrics, time.time() - start_time,
            signals, self.evaluator, self.eval_loader, task_idx if isinstance(task_idx, int) else -1, self._save_checkpoint,
            {k: v for k, v in (consistency_metrics or {}).items() if isinstance(v, (float, int))}, self._reinitialize_dead_prototypes_if_needed
            {k: v for k, v in (consistency_metrics or {}).items() if isinstance(v, (float, int))},
        )

        self.global_step += 1
        self.last_routing_logits = [r.detach() for r in model_outputs.get("routing_logits", []) if r is not None]
        self.last_spl_inputs = [s.detach() for s in model_outputs.get("spl_inputs", []) if s is not None]
        torch.cuda.empty_cache()
        return metrics, model_outputs.get("raw_weights"), model_outputs.get("routing_logits"), signals

    def _calculate_consistency_metrics(self, current_logits: list | None, prev_logits: list | None, current_inputs: list | None, prev_inputs: list | None) -> dict | None:
        if not all([current_logits, prev_logits, current_inputs, prev_inputs]): return None
        reps = [_get_task_representation([t for t in tensor_list if t is not None]) for tensor_list in [current_logits, prev_logits, current_inputs, prev_inputs]]
        if not all(isinstance(rep, torch.Tensor) for rep in reps): return None
        return {"sim_p": F.cosine_similarity(reps[0].unsqueeze(0), reps[1].unsqueeze(0)).item(), "sim_x": F.cosine_similarity(reps[2].unsqueeze(0), reps[3].unsqueeze(0)).item(), "rep_p": reps[0], "rep_x": reps[2]}

    def _train_epoch(self, epoch: int):
        dataset = self.train_loader.dataset
        for task_idx in range(self.start_task_idx, len(dataset)):
            task_data, last_view_routing_logits = dataset[task_idx], None
            start_view = self.start_view_idx if task_idx == self.start_task_idx else 0
            for view_idx in range(start_view, 8):
                converged = False
                batch = self._prepare_batch(task_data, view_idx, self.device, self.serializer, self.consistency_tools, self.config.model.max_position_embeddings)
                if not batch:
                    self.console.print(f"[yellow]Skipping Task {task_idx} View {view_idx} due to excessive length.[/yellow]")
                    continue

                for step in range(500):
                    result = self._train_step(batch, epoch, task_idx, view_idx, last_view_routing_logits)
                    if not result: break
                    metrics, _, routing_logits, _ = result
                    if metrics["main_loss"] <= 0.01 and metrics["token_acc"] >= 1.0:
                        self.console.print(f"Task {task_idx} view {view_idx} converged in {step + 1} steps.")
                        last_view_routing_logits = [r.detach() for r in routing_logits if r is not None]
                        self.replay_queue.append((task_data, view_idx, task_idx))
                        converged = True
                        break
                if not converged: self.console.print(f"[red]Task {task_idx} view {view_idx} hit MAX_STEPS.[/red]")

                self.new_sample_counter += 1
                if self.new_sample_counter % 7 == 0 and self.replay_queue:
                    replay_task, replay_view_idx, replay_task_idx = random.choice(self.replay_queue)
                    self.console.print(f"[cyan]--- Replaying sample (Task -{replay_task_idx} View {replay_view_idx}) ---[/cyan]")
                    replay_batch = self._prepare_batch(replay_task, replay_view_idx, self.device, self.serializer, self.consistency_tools, self.config.model.max_position_embeddings)
                    if replay_batch:
                        self._train_step(replay_batch, epoch, f"-{replay_task_idx}", replay_view_idx, None)

            self.start_view_idx = 0
        self.start_task_idx = 0


    def train(self):
        self._load_checkpoint()
        self.console.print("[bold green]Starting Training...[/bold green]")
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            self._train_epoch(epoch)
        self.console.print("[bold green]Training Finished.[/bold green]")

    def _save_checkpoint(self, task_idx: int, view_idx: int):
        state = {
            "epoch": self.epoch, "step": self.global_step, "task_idx": task_idx, "view_idx": view_idx,
            "model_state_dict": self.model.state_dict(), "optimizer_comp_state_dict": self.optimizer_comp.state_dict(),
            "optimizer_route_state_dict": self.optimizer_route.state_dict(),
        }
        path = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        torch.save(state, path)
        ckpts = sorted(self.checkpoint_dir.glob("*.pt"), key=os.path.getmtime)
        if len(ckpts) > self.config.max_checkpoints: os.remove(ckpts[0])

    def _load_checkpoint(self):
        ckpts = sorted(self.checkpoint_dir.glob("*.pt"), key=os.path.getmtime, reverse=True)
        if not ckpts:
            self.console.print("[bold yellow]No checkpoint found.[/bold yellow]")
            return
        for path in ckpts:
            try:
                ckpt = torch.load(path, map_location=self.device)
                self.model.load_state_dict({k.replace(".sbl", ".spl"): v for k, v in ckpt["model_state_dict"].items()})
                self.optimizer_comp.load_state_dict(ckpt["optimizer_comp_state_dict"])
                self.optimizer_route.load_state_dict(ckpt["optimizer_route_state_dict"])
                self.global_step, self.epoch, self.start_task_idx, self.start_view_idx = (ckpt["step"], ckpt["epoch"], ckpt["task_idx"], ckpt["view_idx"])
                self.console.print(f"[bold green]Loaded checkpoint from {path} at step {self.global_step}.[/bold green]")
                return
            except Exception as e:
                self.console.print(f"[bold red]Corrupted checkpoint {path}: {e}. Trying next.[/bold red]")
        self.console.print("[bold yellow]No valid checkpoint found.[/bold yellow]")


def main():
    config = TrainConfig()
    config.generation = GenerationConfig()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
