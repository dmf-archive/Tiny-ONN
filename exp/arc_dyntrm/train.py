import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from rich.console import Console
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import ArcCollator, GridDeserializer, GridSerializer, InMemoryArcDataset
from .evaluation import EvaluationStep
from .model import ArcTransformer, ExpertMLP
from .observer import Observer
from .tokenizer import ArcColorTokenizer


@torch.jit.script
def mas_normalize_jit(logits: torch.Tensor) -> torch.Tensor:
    max_abs_val = torch.max(torch.abs(logits), dim=-1, keepdim=True).values
    scaled_logits = logits / (max_abs_val + 1e-9)
    return scaled_logits

def _calculate_exact_grad_norms_jit(
    captured_raw_output_grads: list[torch.Tensor],
    all_active_inputs: list[torch.Tensor],
    all_w1_out_active: list[torch.Tensor],
    all_token_indices: list[torch.Tensor],
    all_expert_indices: list[torch.Tensor],
    routing_logits: list[torch.Tensor],
    attn_expert_library: list[ExpertMLP],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    all_goodness_logits: list[torch.Tensor] = []
    all_mu_grad_norms: list[torch.Tensor] = []

    num_modules = len(captured_raw_output_grads)
    for i in range(num_modules):
        grad_raw_output = captured_raw_output_grads[i]
        logits = routing_logits[i]
        
        S, P, D_h = grad_raw_output.shape
        B, T, H, _ = logits.shape
        S = B * T * H

        grad_raw_output_reshaped = grad_raw_output.view(B, T, H, P, D_h)
        output_grad_norm = torch.sqrt(torch.sum(grad_raw_output_reshaped**2, dim=-1) + 1e-9)

        routing_logits_flat = logits.reshape(S, P)
        active_mask = routing_logits_flat > 1e-6
        token_indices, expert_indices = torch.where(active_mask)
        
        if token_indices.numel() == 0:
            all_mu_grad_norms.append(torch.zeros_like(logits))
            all_goodness_logits.append(torch.zeros_like(logits))
            continue

        active_routing_logits = routing_logits_flat[token_indices, expert_indices]
        
        grad_output_flat = grad_raw_output.view(S * P, D_h)
        flat_indices = token_indices * P + expert_indices
        grad_output_active = grad_output_flat.gather(0, flat_indices.unsqueeze(-1).expand(-1, D_h))

        grad_expert_output = grad_output_active * active_routing_logits.unsqueeze(-1)

        active_inputs = all_active_inputs[i]
        w1_out_active = all_w1_out_active[i]
        
        sigmoid_w1 = torch.sigmoid(w1_out_active)
        silu_derivative = sigmoid_w1 * (1 + w1_out_active * (1 - sigmoid_w1))
        
        grad_w1_out = grad_expert_output * silu_derivative
        
        grad_w1_norm_sq = torch.sum(grad_w1_out**2, dim=-1)
        
        silu_w1_out = F.silu(w1_out_active)
        grad_w2_norm_sq = torch.sum(grad_expert_output**2, dim=-1)
        
        total_grad_norm_sq = grad_w1_norm_sq + grad_w2_norm_sq
        mu_grad_norm_active = torch.sqrt(total_grad_norm_sq + 1e-9)
        
        mu_grad_norm = torch.zeros_like(logits.view(-1))
        flat_indices = token_indices * P + expert_indices
        mu_grad_norm.scatter_add_(0, flat_indices, mu_grad_norm_active)
        mu_grad_norm = mu_grad_norm.view_as(logits)
        all_mu_grad_norms.append(mu_grad_norm)

        norm_logits = mas_normalize_jit(logits.detach())
        norm_mu_grad = mas_normalize_jit(mu_grad_norm.detach())
        norm_output_grad = mas_normalize_jit(output_grad_norm.detach())
        
        goodness_logits = norm_output_grad * (norm_logits - norm_mu_grad)
        all_goodness_logits.append(goodness_logits)

    return all_goodness_logits, all_mu_grad_norms


class LearningDynamics:
    def __init__(
        self,
        config: TrainConfig,
        model: nn.Module,
        computation_params: list,
        routing_params_with_names: list[tuple[str, nn.Parameter]],
        optimizer_comp: torch.optim.Optimizer,
        optimizer_route: torch.optim.Optimizer,
        accelerator: Accelerator,
    ):
        self.config = config
        self.model = model
        self.computation_params = computation_params
        self.routing_params_with_names = routing_params_with_names
        self.routing_params = [p for _, p in routing_params_with_names]
        self.optimizer_comp = optimizer_comp
        self.optimizer_route = optimizer_route
        self.accelerator = accelerator

    @staticmethod
    def _calculate_meta_loss(goodness_logits: torch.Tensor) -> torch.Tensor:
        dist = F.softmax(goodness_logits, dim=-1)
        log_dist = torch.log(dist + 1e-9)
        entropy = -torch.sum(dist * log_dist, dim=-1)
        return entropy.mean()

    def compute_and_apply_gradients(
        self,
        main_loss: torch.Tensor,
        model_outputs: dict,
        device: torch.device,
        captured_raw_inputs: list[torch.Tensor],
        captured_raw_output_grads: list[torch.Tensor],
    ) -> dict[str, Any]:

        self.optimizer_comp.zero_grad()
        self.optimizer_route.zero_grad()

        self.accelerator.backward(main_loss, retain_graph=True)

        routing_logits = model_outputs.get("all_routing_logits", [])

        if not captured_raw_output_grads:
            torch.nn.utils.clip_grad_value_(self.computation_params, clip_value=1.0)
            self.optimizer_comp.step()
            return {"meta_loss": torch.tensor(0.0), "mu_grad_norms": [], "goodness_logits": [], "proto_weights": []}

        all_active_inputs = model_outputs.get("all_active_inputs", [])
        all_w1_out_active = model_outputs.get("all_w1_out_active", [])
        all_token_indices = model_outputs.get("all_token_indices", [])
        all_expert_indices = model_outputs.get("all_expert_indices", [])
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        all_goodness_logits, all_mu_grad_norms = _calculate_exact_grad_norms_jit(
            captured_raw_output_grads,
            all_active_inputs,
            all_w1_out_active,
            all_token_indices,
            all_expert_indices,
            routing_logits,
            unwrapped_model.attn_expert_library,
        )

        meta_losses = []
        for i in range(len(all_goodness_logits)):
            meta_losses.append(self._calculate_meta_loss(all_goodness_logits[i]))

        avg_meta_loss = torch.stack(meta_losses).mean() if meta_losses else torch.tensor(0.0, device=device)
        total_meta_loss = self.config.w_meta * avg_meta_loss

        if total_meta_loss.requires_grad:
            self.accelerator.backward(total_meta_loss)

        torch.nn.utils.clip_grad_value_(self.computation_params, clip_value=1.0)
        self.optimizer_comp.step()
        self.optimizer_route.step()

        return {
            "meta_loss": avg_meta_loss,
            "mu_grad_norms": all_mu_grad_norms,
            "goodness_logits": all_goodness_logits,
            "proto_weights": [p.detach() for name, p in self.routing_params_with_names if "proto" in name],
        }

class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        torch.manual_seed(config.seed)

        self.console = Console()
        self.observer = Observer(self.console, config)
        self.tokenizer = ArcColorTokenizer()
        self.serializer = GridSerializer(self.tokenizer)
        self.deserializer = GridDeserializer(self.tokenizer)

        self._setup_data()
        self._setup_model_and_optimizer()

        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.checkpoint_dir.mkdir(exist_ok=True)

        self.global_step, self.epoch, self.start_task_idx, self.start_view_idx = 0, 0, 0, 0
        self.total_tasks_processed = 0
        self.curriculum_stage = 1
        self.captured_raw_inputs: list[torch.Tensor] = []
        self.captured_raw_output_grads: list[torch.Tensor] = []

    def _setup_data(self):
        collator = ArcCollator(self.tokenizer, max_len=self.config.model.max_position_embeddings)
        train_dataset = InMemoryArcDataset(
            data_path=self.config.data.data_path,
            tokenizer=self.tokenizer,
            split="training",
            warmup_ratio=self.config.data.warmup_dataset_ratio,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.data.batch_size, collate_fn=collator, num_workers=self.config.data.num_workers, shuffle=False, pin_memory=True,
        )
        self.eval_loader = DataLoader(
            InMemoryArcDataset(data_path=self.config.data.data_path, tokenizer=self.tokenizer, split="evaluation"),
            batch_size=1, collate_fn=collator, num_workers=self.config.data.num_workers, shuffle=False, pin_memory=True,
        )

    def _setup_model_and_optimizer(self):
        self.config.model.vocab_size = self.tokenizer.vocab_size
        model = ArcTransformer(self.config.model, self.config.generation, device=self.device)

        computation_params = [p for name, p in model.named_parameters() if "proto" not in name and "gate" not in name]
        routing_params_with_names = [(name, p) for name, p in model.named_parameters() if "proto" in name or "gate" in name]

        optimizer_comp = torch.optim.AdamW(computation_params, lr=self.config.lr)
        optimizer_route = torch.optim.AdamW([p for _, p in routing_params_with_names], lr=self.config.lr)

        self.model, self.optimizer_comp, self.optimizer_route, self.train_loader, self.eval_loader = self.accelerator.prepare(
            model, optimizer_comp, optimizer_route, self.train_loader, self.eval_loader
        )

        self.dynamics = LearningDynamics(
            self.config, self.model, computation_params, routing_params_with_names, self.optimizer_comp, self.optimizer_route, self.accelerator
        )
        self.evaluator = EvaluationStep(self.model, self.serializer, self.deserializer, self.observer, self.device, self.train_loader.dataset, self.config)

    def _prepare_batch(self, task_data: dict, view_idx: int, max_len: int) -> dict[str, torch.Tensor] | None:
        all_colors = set()
        grids_cpu_lists = []

        train_pairs = task_data["train"].copy()
        random.shuffle(train_pairs)

        for pair in train_pairs:
            grids_cpu_lists.extend([pair["input"], pair["output"]])
            for row in pair["input"]: all_colors.update(row)
            for row in pair["output"]: all_colors.update(row)

        grids_cpu_lists.extend([task_data["test"][0]["input"], task_data["test"][0]["output"]])
        for row in task_data["test"][0]["input"]: all_colors.update(row)
        if "output" in task_data["test"][0]:
            for row in task_data["test"][0]["output"]: all_colors.update(row)

        active_colors = [c for c in all_colors if c != 0]
        color_map_cpu = torch.arange(10, dtype=torch.long)
        if len(active_colors) >= 2:
            c1, c2 = random.sample(active_colors, 2)
            color_map_cpu[c1], color_map_cpu[c2] = c2, c1

        grids_cpu_tensors = [torch.tensor(g, dtype=torch.long) for g in grids_cpu_lists]
        augmented_grids = _augment_and_map_kernel(grids_cpu_tensors, view_idx, color_map_cpu)
        augmented_grids_list = [g.tolist() for g in augmented_grids]

        transformed_train, ptr = [], 0
        for _ in train_pairs:
            transformed_train.append({"input": augmented_grids_list[ptr], "output": augmented_grids_list[ptr + 1]})
            ptr += 2

        transformed_test = [{"input": augmented_grids_list[ptr], "output": augmented_grids_list[ptr + 1]}]
        augmented_task = {"train": transformed_train, "test": transformed_test}

        ids, labels, _ = self.serializer.serialize_task(augmented_task)
        if len(ids) > max_len:
            return None

        device = self.accelerator.device
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
            "labels": torch.tensor([labels], dtype=torch.long, device=device),
            "sample_entropy": torch.tensor([ArcCollator._calculate_sample_entropy(labels)], dtype=torch.float32, device=device),
        }

    def _train_step(self, batch: dict, epoch: int, task_idx: int | str, view_idx: int) -> tuple | None:
        start_time = time.time()
        self.model.train()

        self.captured_raw_inputs.clear()
        self.captured_raw_output_grads.clear()

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            model_outputs = self.model(
                batch["input_ids"],
                return_dict=True,
                captured_raw_inputs=self.captured_raw_inputs,
                captured_raw_output_grads=self.captured_raw_output_grads,
            )

        main_loss = F.cross_entropy(
            model_outputs["logits"][:, :-1, :].contiguous().view(-1, self.config.model.vocab_size),
            batch["labels"][:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )

        if not torch.isfinite(main_loss):
            self.console.print(f"[bold red]NaN detected in main_loss at step {self.global_step}. Aborting step.[/bold red]")
            return None

        signals = self.dynamics.compute_and_apply_gradients(
            main_loss, model_outputs, self.device, self.captured_raw_inputs, self.captured_raw_output_grads
        )

        model_outputs.update({"labels": batch["labels"], "sample_entropy": batch["sample_entropy"]})
        metrics = self.observer.calculate_metrics(main_loss, model_outputs, signals, batch["input_ids"], self.model)

        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)

            proto_weights_dict = {
                "q": unwrapped_model.block.attn.proto_q,
                "k": unwrapped_model.block.attn.proto_k,
                "v": unwrapped_model.block.attn.proto_v,
            }

            visualization_signals = {
                "all_router_logits": model_outputs.get("all_routing_logits", []),
                "goodness_logits": signals.get("goodness_logits", []),
                "proto_weights": proto_weights_dict,
            }

            self.observer.maybe_log_and_visualize(
                epoch, self.global_step, task_idx if isinstance(task_idx, int) else -1, view_idx,
                metrics, time.time() - start_time, visualization_signals, self.evaluator, self.eval_loader,
                self.total_tasks_processed, self._save_checkpoint, self.advance_curriculum, self.curriculum_stage,
            )

        self.global_step += 1
        return metrics, model_outputs.get("all_routing_logits", []), signals

    def advance_curriculum(self):
        if self.curriculum_stage == 1:
            self.curriculum_stage = 2
            self.train_loader.dataset.set_stage(2)
            self.console.print("[bold magenta]Curriculum stage advanced to 2. Using full dataset.[/bold magenta]")
            self.start_task_idx, self.start_view_idx = 0, 0
            self.epoch = 0

    def _train_epoch(self, epoch: int):
        dataset = self.train_loader.dataset
        for task_idx, task_data in enumerate(dataset):
            if task_idx < self.start_task_idx:
                continue

            start_view = self.start_view_idx if task_idx == self.start_task_idx else 0

            all_views = list(range(8))
            selected_views = random.sample(all_views, self.config.num_augmentation_views) if start_view == 0 else all_views[start_view:]

            for view_idx in selected_views:
                self.start_view_idx = view_idx
                batch_cpu = self._prepare_batch(task_data, view_idx, self.config.model.max_position_embeddings)
                if not batch_cpu:
                    if self.accelerator.is_main_process:
                        self.console.print(f"[yellow]Skipping Task {task_idx} View {view_idx} due to excessive length.[/yellow]")
                    continue

                batch = batch_cpu
                for step in range(self.config.max_steps_per_view):
                    result = self._train_step(batch, epoch, task_idx, view_idx)
                    if not result: break
                    metrics, _, _ = result
                    if metrics["main_loss"] <= 0.05 and metrics["token_acc"] >= 0.999:
                        if self.accelerator.is_main_process:
                            self.console.print(f"Task {task_idx} view {view_idx} converged in {step + 1} steps.")
                        break
            self.start_view_idx = 0
            self.total_tasks_processed = task_idx + 1
        self.start_task_idx = 0

    def train(self):
        self._load_checkpoint()
        if self.accelerator.is_main_process:
            self.console.print("[bold green]Starting Training...[/bold green]")
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            self._train_epoch(epoch)
        if self.accelerator.is_main_process:
            self.console.print("[bold green]Training Finished.[/bold green]")

    def _save_checkpoint(self, task_idx: int, view_idx: int):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save_state(self.checkpoint_dir / f"checkpoint_{self.global_step}")

            ckpts = sorted(self.checkpoint_dir.glob("checkpoint_*"), key=os.path.getmtime)
            if len(ckpts) > self.config.max_checkpoints:
                shutil.rmtree(ckpts[0])

    def _load_checkpoint(self):
        ckpts = sorted(self.checkpoint_dir.glob("checkpoint_*"), key=os.path.getmtime, reverse=True)
        if not ckpts:
            if self.accelerator.is_main_process:
                self.console.print("[bold yellow]No checkpoint found.[/bold yellow]")
            return

        self.accelerator.load_state(ckpts[0])
        if self.accelerator.is_main_process:
            self.console.print(f"[bold green]Loaded checkpoint from {ckpts[0]}.[/bold green]")

def _augment_and_map_kernel(grids: list[torch.Tensor], transform_idx: int, color_map: torch.Tensor) -> list[torch.Tensor]:
    transformed_grids = []
    for x in grids:
        if transform_idx == 0: transformed_x = x
        elif transform_idx == 1: transformed_x = torch.rot90(x, 1, [0, 1])
        elif transform_idx == 2: transformed_x = torch.rot90(x, 2, [0, 1])
        elif transform_idx == 3: transformed_x = torch.rot90(x, 3, [0, 1])
        elif transform_idx == 4: transformed_x = torch.flip(x, [0])
        elif transform_idx == 5: transformed_x = torch.flip(x, [1])
        elif transform_idx == 6: transformed_x = torch.transpose(x, 0, 1)
        else: transformed_x = torch.rot90(torch.flip(x, [0]), 1, [0, 1])
        transformed_grids.append(color_map[transformed_x])
    return transformed_grids

def main():
    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
