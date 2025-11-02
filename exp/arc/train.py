import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from safetensors.torch import load_file, save_file
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import ArcCollator, GridDeserializer, GridSerializer, InMemoryArcDataset
from .evaluation import EvaluationStep
from .model import ArcTransformer, SparseProtoLinear
from .observer import Observer
from .tokenizer import ArcColorTokenizer
from .fisher import Compute_H_D, Compute_S_D, update_running_avg, MinMaxNormalization, kron


@torch.jit.script
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


@torch.jit.script
def _augment_and_map_kernel(grids: list[torch.Tensor], transform_idx: int, color_map: torch.Tensor) -> list[torch.Tensor]:
    transformed_grids = []
    for x in grids:
        if transform_idx == 0:
            transformed_x = x
        elif transform_idx == 1:
            transformed_x = torch.rot90(x, 1, [0, 1])
        elif transform_idx == 2:
            transformed_x = torch.rot90(x, 2, [0, 1])
        elif transform_idx == 3:
            transformed_x = torch.rot90(x, 3, [0, 1])
        elif transform_idx == 4:
            transformed_x = torch.flip(x, [0])
        elif transform_idx == 5:
            transformed_x = torch.flip(x, [1])
        elif transform_idx == 6:
            transformed_x = torch.transpose(x, 0, 1)
        else:
            transformed_x = torch.rot90(torch.flip(x, [0]), 1, [0, 1])
        transformed_grids.append(color_map[transformed_x])
    return transformed_grids


@torch.jit.script
def mas_normalize_jit(logits: torch.Tensor) -> torch.Tensor:
    max_abs_val = torch.max(torch.abs(logits), dim=-1, keepdim=True).values
    scaled_logits = logits / (max_abs_val + 1e-9)
    return scaled_logits


class LearningDynamics:
    def __init__(
        self,
        config: TrainConfig,
        model: nn.Module,
        computation_params: list,
        routing_params_with_names: list[tuple[str, nn.Parameter]],
        optimizer_comp: torch.optim.Optimizer,
        optimizer_route: torch.optim.Optimizer,
    ):
        self.config = config
        self.model = model
        self.computation_params = computation_params
        self.routing_params_with_names = routing_params_with_names
        self.routing_params = [p for _, p in routing_params_with_names]
        self.optimizer_comp = optimizer_comp
        self.optimizer_route = optimizer_route

        self.H_D: Dict[nn.Module, torch.Tensor] = {}
        self.S_D: Dict[nn.Module, torch.Tensor] = {}
        self.modules: List[nn.Module] = []
        self.captured_inputs: Dict[nn.Module, torch.Tensor] = {}
        self.captured_grad_outputs: Dict[nn.Module, torch.Tensor] = {}
        self.steps = 0
        self._prepare_model()

    def _save_input(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
        if torch.is_grad_enabled():
            self.captured_inputs[module] = input[0].data.detach()

    def _save_grad_output(self, module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
         if torch.is_grad_enabled():
            self.captured_grad_outputs[module] = grad_output[0].data.detach()

    def _prepare_model(self):
        for module in self.model.modules():
            if isinstance(module, SparseProtoLinear):
                self.modules.append(module)
                module.register_forward_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)

    def _update_fisher_diagonals(self):
        for module in self.modules:
            if module in self.captured_inputs and module in self.captured_grad_outputs:
                h = self.captured_inputs[module]
                s = self.captured_grad_outputs[module]
                
                H_D_i = Compute_H_D.linear(h, module)
                S_D_i = Compute_S_D.linear(s, module)

                if module not in self.H_D:
                    self.H_D[module] = torch.ones_like(H_D_i)
                    self.S_D[module] = torch.ones_like(S_D_i)

                update_running_avg(MinMaxNormalization(H_D_i), self.H_D[module], 0.9)
                update_running_avg(MinMaxNormalization(S_D_i), self.S_D[module], 0.9)
        
        self.captured_inputs.clear()
        self.captured_grad_outputs.clear()

    def compute_and_apply_gradients(
        self,
        main_loss: torch.Tensor,
        model_outputs: dict,
        device: torch.device,
    ) -> dict[str, Any]:
        
        self.optimizer_comp.zero_grad()
        self.optimizer_route.zero_grad()

        # Decoupled learning:
        # 1. Store main loss gradients for computation params
        comp_grads = [p.grad.clone() for p in self.computation_params if p.grad is not None]
        
        # 2. Zero all gradients before meta-loss backward
        self.optimizer_comp.zero_grad()
        self.optimizer_route.zero_grad()
        
        # 3. L_meta Calculation
        # 3a. Entropy Term
        entropy_losses = []
        for routing_logits in model_outputs["routing_logits"]:
            if routing_logits.numel() > 0:
                probs = F.softmax(routing_logits, dim=-1)
                log_probs = F.log_softmax(routing_logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                entropy_losses.append(entropy)
        l_entropy = torch.stack(entropy_losses).mean() if entropy_losses else torch.tensor(0.0, device=device)

        # 3b. Exploration Term
        explore_losses = []
        if self.steps % 10 == 0:
            self._update_fisher_diagonals()
        
        for module in self.modules:
            if module in self.H_D and module in self.S_D:
                fisher_diag = torch.kron(self.H_D[module], self.S_D[module]) + 1e-8
                log_det_F = torch.log(fisher_diag).sum()
                explore_losses.append(-log_det_F)

        l_explore = torch.stack(explore_losses).mean() if explore_losses else torch.tensor(0.0, device=device)
        
        avg_meta_loss = self.config.w_meta * (l_entropy - 0.1 * l_explore)

        # 4. Backward pass for meta-loss (updates routing params)
        if avg_meta_loss.requires_grad:
            avg_meta_loss.backward()

        # 5. Restore main loss gradients for computation params
        for p, g in zip([p for p in self.computation_params if p.grad is not None], comp_grads):
            p.grad = g

        # 6. Clip and step both optimizers
        torch.nn.utils.clip_grad_value_(self.computation_params, clip_value=1.0)
        
        # Manually clip routing param grads if they exist
        for p in self.routing_params:
            if p.grad is not None:
                p.grad.data.clamp_(-1.0, 1.0)
            
        self.optimizer_comp.step()
        self.optimizer_route.step()
        
        self.steps += 1

        return {
            "meta_loss": avg_meta_loss,
            "mu_grad_norms": [],
            "goodness_logits": [],
            "proto_weights": [p.detach() for name, p in self.routing_params_with_names if "proto_weight" in name],
        }


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config, self.device = config, torch.device(config.device)
        torch.manual_seed(config.seed)
        self.console = Console()
        self.observer = Observer(self.console, config)
        self.tokenizer, self.serializer = ArcColorTokenizer(), GridSerializer(ArcColorTokenizer())
        self._setup_data()
        self._setup_model_and_optimizer()
        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.global_step, self.epoch, self.start_task_idx, self.start_view_idx = 0, 0, 0, 0
        self.total_tasks_processed = 0
        self.curriculum_stage = 1

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
        self.model = ArcTransformer(self.config.model, device=self.device).to(self.device)
        computation_params = [p for name, p in self.model.named_parameters() if "proto_weight" not in name and "gate_param" not in name]
        routing_params_with_names = [(name, p) for name, p in self.model.named_parameters() if "proto_weight" in name or "gate_param" in name]

        self.optimizer_comp = torch.optim.AdamW(computation_params, lr=self.config.lr)
        self.optimizer_route = torch.optim.AdamW([p for _, p in routing_params_with_names], lr=self.config.lr)
        self.dynamics = LearningDynamics(self.config, self.model, computation_params, routing_params_with_names, self.optimizer_comp, self.optimizer_route)
        self.evaluator = EvaluationStep(self.model, self.serializer, GridDeserializer(self.tokenizer), self.observer, self.device, self.train_loader.dataset, self.config)

    def _prepare_batch(self, task_data: dict, view_idx: int, max_len: int) -> dict[str, torch.Tensor] | None:
        all_colors = set()
        grids_cpu_lists = []

        train_pairs = task_data["train"].copy()
        random.shuffle(train_pairs)

        for pair in train_pairs:
            grids_cpu_lists.extend([pair["input"], pair["output"]])
            for row in pair["input"]:
                all_colors.update(row)
            for row in pair["output"]:
                all_colors.update(row)

        grids_cpu_lists.extend([task_data["test"][0]["input"], task_data["test"][0]["output"]])
        for row in task_data["test"][0]["input"]:
            all_colors.update(row)
        if "output" in task_data["test"][0]:
            for row in task_data["test"][0]["output"]:
                all_colors.update(row)

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

        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
            "sample_entropy": torch.tensor([ArcCollator._calculate_sample_entropy(labels)], dtype=torch.float32),
        }

    def _train_step(self, batch: dict, epoch: int, task_idx: int | str, view_idx: int) -> tuple | None:
        start_time = time.time()
        self.model.train()

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION), torch.autocast(
            device_type=self.config.device, dtype=torch.float32
        ):
            model_outputs = self.model(
                batch["input_ids"],
                coords=None,
                return_dict=True,
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
            main_loss, model_outputs, self.device
        )
        model_outputs.update({"labels": batch["labels"], "sample_entropy": batch["sample_entropy"]})
        metrics = self.observer.calculate_metrics(main_loss, model_outputs, signals, batch["input_ids"], self.model)

        signals["masked_routing_logits"] = model_outputs.get("masked_routing_logits")
        signals["routing_logits"] = model_outputs.get("routing_logits")
        self.observer.maybe_log_and_visualize(
            epoch,
            self.global_step,
            task_idx if isinstance(task_idx, int) else -1,
            view_idx,
            metrics,
            time.time() - start_time,
            signals,
            self.evaluator,
            self.eval_loader,
            self.total_tasks_processed,
            self._save_checkpoint,
            self.advance_curriculum,
            self.curriculum_stage,
        )

        self.global_step += 1
        return metrics, model_outputs.get("masked_routing_logits"), model_outputs.get("routing_logits"), signals


    def advance_curriculum(self):
        if self.curriculum_stage == 1:
            self.curriculum_stage = 2
            self.train_loader.dataset.set_stage(2)
            self.console.print("[bold magenta]Curriculum stage advanced to 2. Using full dataset.[/bold magenta]")
            self.start_task_idx, self.start_view_idx = 0, 0
            self.epoch = 0

    def _train_epoch(self, epoch: int):
        dataset = self.train_loader.dataset
        num_tasks_in_stage = len(dataset)
        
        for task_idx, task_data in enumerate(dataset):
            if task_idx < self.start_task_idx:
                continue

            start_view = self.start_view_idx if task_idx == self.start_task_idx else 0
            
            all_views = list(range(8))
            if start_view > 0:
                selected_views = all_views[start_view:]
            else:
                selected_views = random.sample(all_views, self.config.num_augmentation_views)

            for view_idx in selected_views:
                self.start_view_idx = view_idx
                batch_cpu = self._prepare_batch(task_data, view_idx, self.config.model.max_position_embeddings)
                if not batch_cpu:
                    self.console.print(
                        f"[yellow]Skipping Task {task_idx} View {view_idx} due to excessive length.[/yellow]"
                    )
                    continue

                batch = {k: v.to(self.device) for k, v in batch_cpu.items() if isinstance(v, torch.Tensor)}
                for step in range(self.config.max_steps_per_view):
                    result = self._train_step(batch, epoch, task_idx, view_idx)
                    if not result:
                        break
                    metrics, _, _, _ = result
                    if metrics["main_loss"] <= 0.05 and metrics["token_acc"] >= 0.999:
                        self.console.print(f"Task {task_idx} view {view_idx} converged in {step + 1} steps.")
                        break
            self.start_view_idx = 0
            torch.cuda.empty_cache()
            self.total_tasks_processed = task_idx + 1
        self.start_task_idx = 0

    def train(self):
        self._load_checkpoint()
        self.console.print("[bold green]Starting Training...[/bold green]")
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            self._train_epoch(epoch)
        self.console.print("[bold green]Training Finished.[/bold green]")

    def _save_checkpoint(self, task_idx: int, view_idx: int):
        checkpoint_dir = self.checkpoint_dir / f"checkpoint_{self.global_step}"
        checkpoint_dir.mkdir(exist_ok=True)

        model_path = checkpoint_dir / "model.safetensors"
        save_file(self.model.state_dict(), model_path)

        trainer_state = {
            "epoch": self.epoch, "step": self.global_step, "task_idx": task_idx, "view_idx": view_idx,
            "total_tasks_processed": self.total_tasks_processed,
            "optimizer_comp_state_dict": self.optimizer_comp.state_dict(),
            "optimizer_route_state_dict": self.optimizer_route.state_dict(),
        }
        state_path = checkpoint_dir / "trainer_state.pt"
        torch.save(trainer_state, state_path)

        ckpts = sorted(self.checkpoint_dir.glob("checkpoint_*"), key=os.path.getmtime)
        if len(ckpts) > self.config.max_checkpoints:
            oldest_ckpt = ckpts[0]
            if oldest_ckpt.is_dir():
                shutil.rmtree(oldest_ckpt)
            else:
                os.remove(oldest_ckpt)

    def _load_checkpoint(self):
        ckpts = sorted(self.checkpoint_dir.glob("checkpoint_*"), key=os.path.getmtime, reverse=True)
        if not ckpts:
            self.console.print("[bold yellow]No checkpoint found.[/bold yellow]")
            return

        for path in ckpts:
            try:
                if path.is_dir():
                    model_path = path / "model.safetensors"
                    state_path = path / "trainer_state.pt"
                    if not model_path.exists() or not state_path.exists():
                        continue

                    model_state_dict = load_file(model_path, device=str(self.device))
                    self.model.load_state_dict(model_state_dict)
                    state = torch.load(state_path, map_location=self.device)
                    self.optimizer_comp.load_state_dict(state["optimizer_comp_state_dict"])
                    self.optimizer_route.load_state_dict(state["optimizer_route_state_dict"])
                    self.global_step, self.epoch, self.start_task_idx, self.start_view_idx = (state["step"], state["epoch"], state["task_idx"], state["view_idx"])
                    self.total_tasks_processed = state["total_tasks_processed"]

                elif path.is_file():
                    ckpt = torch.load(path, map_location=self.device)
                    self.model.load_state_dict({k.replace(".sbl", ".spl"): v for k, v in ckpt["model_state_dict"].items()})
                    self.optimizer_comp.load_state_dict(ckpt["optimizer_comp_state_dict"])
                    self.optimizer_route.load_state_dict(ckpt["optimizer_route_state_dict"])
                    self.global_step, self.epoch, self.start_task_idx, self.start_view_idx = (ckpt["step"], ckpt["epoch"], ckpt["task_idx"], ckpt["view_idx"])
                    self.total_tasks_processed = ckpt["total_tasks_processed"]

                self.console.print(f"[bold green]Loaded checkpoint from {path} at step {self.global_step}.[/bold green]")
                return

            except Exception as e:
                self.console.print(f"[bold red]Corrupted or invalid checkpoint {path}: {e}. Trying next.[/bold red]")

        self.console.print("[bold yellow]No valid checkpoint found.[/bold yellow]")


def main():
    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
