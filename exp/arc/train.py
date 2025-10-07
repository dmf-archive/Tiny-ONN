import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader

from .config import TrainConfig
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
    return 0.5 * kl_p_m + 0.5 * kl_q_m

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
def _calculate_goodness_jit(
    masked_outputs: list[torch.Tensor],
    captured_spl_grad_outputs: list[torch.Tensor],
    captured_spl_inputs: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    all_goodness: list[torch.Tensor] = []
    all_goodness_logits: list[torch.Tensor] = []
    all_mu_surprises: list[torch.Tensor] = []

    num_modules = len(masked_outputs)
    for i in range(num_modules):
        grad_output = captured_spl_grad_outputs[i]
        x = captured_spl_inputs[i]
        masked_output = masked_outputs[i]
        
        b_contrib_token = torch.abs(masked_output)
        b_rel_token = torch.abs(grad_output)

        per_token_grad_weight = torch.einsum("bso,bsi->bsoi", grad_output, x)
        c_learn_token = torch.norm(per_token_grad_weight, p=2, dim=-1)
        
        all_mu_surprises.append(torch.mean(c_learn_token, dim=(0, 1)))

        benefit_eff_token = mas_normalize(b_contrib_token)
        benefit_rel_token = mas_normalize(b_rel_token)
        synergistic_benefit_token = mas_normalize(benefit_eff_token * benefit_rel_token)

        learning_cost_token = mas_normalize(c_learn_token)

        goodness_logits_token = mas_normalize(synergistic_benefit_token / (learning_cost_token + 1e-9))
        
        all_goodness_logits.append(goodness_logits_token)
        all_goodness.append(F.relu(goodness_logits_token))

    return all_goodness, all_goodness_logits, all_mu_surprises


class LearningDynamics:
    def __init__(
        self,
        config: TrainConfig,
        model: nn.Module,
        computation_params: list,
        routing_params_with_names: list[tuple[str, nn.Parameter]],
        optimizer_comp: torch.optim.Optimizer,
        optimizer_route: torch.optim.Optimizer,
        spl_modules: list[nn.Module],
    ):
        self.config = config
        self.model = model
        self.computation_params = computation_params
        self.routing_params_with_names = routing_params_with_names
        self.routing_params = [p for _, p in routing_params_with_names]
        self.optimizer_comp = optimizer_comp
        self.optimizer_route = optimizer_route
        self.spl_modules = spl_modules

    @staticmethod
    @torch.jit.script
    def _calculate_jsd_loss(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        p_dist_unnorm = F.relu(mas_normalize(p_logits))
        q_dist_unnorm = mas_normalize(q_logits).detach()
        return _jsd_from_distributions(p_dist_unnorm, q_dist_unnorm)

    def compute_and_apply_gradients(
        self,
        main_loss: torch.Tensor,
        model_outputs: dict,
        device: torch.device,
        captured_spl_inputs: list[torch.Tensor],
        captured_spl_grad_outputs: list[torch.Tensor],
        masked_outputs: list[torch.Tensor],
    ) -> dict[str, Any]:
        self.optimizer_comp.zero_grad()
        self.optimizer_route.zero_grad()

        main_loss.backward(retain_graph=True)

        with torch.no_grad():
            for param in self.routing_params:
                if param.grad is not None:
                    param.grad.zero_()

        all_goodness, all_goodness_logits, all_mu_surprises = _calculate_goodness_jit(
            masked_outputs, captured_spl_grad_outputs, captured_spl_inputs
        )

        meta_losses = [
            self._calculate_jsd_loss(logit, good)
            for logit, good in zip(model_outputs["routing_logits"], all_goodness)
            if logit.numel() > 0 and good.numel() > 0 and logit.shape == good.shape
        ]
        
        avg_route_jsd_loss = torch.stack(meta_losses).mean() if meta_losses else torch.tensor(0.0, device=device)
        total_meta_loss = self.config.w_route_jsd * avg_route_jsd_loss

        if total_meta_loss > 0:
            meta_grads = torch.autograd.grad(total_meta_loss, self.routing_params, allow_unused=True)
            with torch.no_grad():
                for param, grad in zip(self.routing_params, meta_grads):
                    if grad is not None:
                        param.grad = grad

        for i, module in enumerate(self.spl_modules):
            if i < len(all_goodness):
                goodness = all_goodness[i]
                goodness_mask = torch.mean(goodness, dim=(0, 1)).detach()

                if module.mu_weight.grad is not None:
                    module.mu_weight.grad.mul_(goodness_mask.unsqueeze(1))
                if module.mu_bias.grad is not None:
                    module.mu_bias.grad.mul_(goodness_mask)

        torch.nn.utils.clip_grad_value_(self.computation_params, clip_value=1.0)
        self.optimizer_comp.step()
        self.optimizer_route.step()

        return {
            "route_jsd_loss": avg_route_jsd_loss,
            "mu_surprises": all_mu_surprises,
            "goodness_scores": all_goodness,
            "goodness_logits": all_goodness_logits,
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
        self.captured_spl_inputs: list[torch.Tensor] = []
        self.captured_spl_grad_outputs: list[torch.Tensor] = []

    def _setup_data(self):
        collator = ArcCollator(self.tokenizer, max_len=self.config.model.max_position_embeddings)
        self.train_loader = DataLoader(
            InMemoryArcDataset(data_path=self.config.data.data_path, split="training"),
            batch_size=self.config.data.batch_size, collate_fn=collator, num_workers=self.config.data.num_workers, shuffle=False, pin_memory=True,
        )
        self.eval_loader = DataLoader(
            InMemoryArcDataset(data_path=self.config.data.data_path, split="evaluation"),
            batch_size=1, collate_fn=collator, num_workers=self.config.data.num_workers, shuffle=False, pin_memory=True,
        )

    def _setup_model_and_optimizer(self):
        self.config.model.vocab_size = self.tokenizer.vocab_size
        self.model = ArcTransformer(self.config.model, device=self.device).to(self.device)
        computation_params = [p for name, p in self.model.named_parameters() if "proto_weight" not in name and "gate_param" not in name]
        routing_params_with_names = [(name, p) for name, p in self.model.named_parameters() if "proto_weight" in name or "gate_param" in name]
        
        spl_modules = []
        for block in self.model.blocks:
            spl_modules.append(block.attn.spl_q)
            spl_modules.append(block.attn.spl_k)
            spl_modules.append(block.attn.spl_v)
            spl_modules.append(block.attn.spl_o)

        self.optimizer_comp = torch.optim.AdamW(computation_params, lr=self.config.lr)
        self.optimizer_route = torch.optim.AdamW([p for _, p in routing_params_with_names], lr=self.config.lr)
        self.dynamics = LearningDynamics(self.config, self.model, computation_params, routing_params_with_names, self.optimizer_comp, self.optimizer_route, spl_modules)
        self.evaluator = EvaluationStep(self.model, self.serializer, GridDeserializer(self.tokenizer), self.observer, self.device, self.train_loader.dataset, self.config)

    def _prepare_batch(self, task_data: dict, view_idx: int, max_len: int) -> dict[str, torch.Tensor] | None:
        all_colors = set()
        grids_cpu_lists = []
        for pair in task_data["train"]:
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
        for _ in task_data["train"]:
            transformed_train.append({"input": augmented_grids_list[ptr], "output": augmented_grids_list[ptr + 1]})
            ptr += 2

        transformed_test = [{"input": augmented_grids_list[ptr], "output": augmented_grids_list[ptr + 1]}]
        augmented_task = {"train": transformed_train, "test": transformed_test}

        ids, labels, coords = self.serializer.serialize_task(augmented_task)
        if len(ids) > max_len:
            return None

        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
            "coords": torch.tensor([coords], dtype=torch.long),
            "sample_entropy": torch.tensor([ArcCollator._calculate_sample_entropy(labels)], dtype=torch.float32),
        }

    def _train_step(self, batch: dict, epoch: int, task_idx: int | str, view_idx: int) -> tuple | None:
        start_time = time.time()
        self.model.train()
        
        self.captured_spl_inputs.clear()
        self.captured_spl_grad_outputs.clear()

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            with torch.autocast(device_type=self.config.device, dtype=torch.float32):
                model_outputs = self.model(
                    batch["input_ids"],
                    coords=batch["coords"],
                    return_dict=True,
                    captured_spl_inputs=self.captured_spl_inputs,
                    captured_spl_grad_outputs=self.captured_spl_grad_outputs,
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
            main_loss, model_outputs, self.device, self.captured_spl_inputs, self.captured_spl_grad_outputs, model_outputs["masked_outputs"]
        )
        model_outputs.update({"labels": batch["labels"], "sample_entropy": batch["sample_entropy"]})
        metrics = self.observer.calculate_metrics(main_loss, model_outputs, signals, batch["input_ids"], self.model)

        signals["raw_weights"] = model_outputs.get("raw_weights")
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
            task_idx if isinstance(task_idx, int) else -1,
            self._save_checkpoint,
        )

        self.global_step += 1
        torch.cuda.empty_cache()
        return metrics, model_outputs.get("raw_weights"), model_outputs.get("routing_logits"), signals


    def _train_epoch(self, epoch: int):
        dataset = self.train_loader.dataset
        for task_idx in range(self.start_task_idx, len(dataset)):
            task_data = dataset[task_idx]
            start_view = self.start_view_idx if task_idx == self.start_task_idx else 0

            all_views = list(range(8))
            if start_view > 0:
                selected_views = all_views[start_view:]
            else:
                selected_views = random.sample(all_views, self.config.num_augmentation_views)

            for view_idx in selected_views:
                batch_cpu = self._prepare_batch(task_data, view_idx, self.config.model.max_position_embeddings)
                if not batch_cpu:
                    self.console.print(
                        f"[yellow]Skipping Task {task_idx} View {view_idx} due to excessive length.[/yellow]"
                    )
                    continue

                batch = {k: v.to(self.device) for k, v in batch_cpu.items() if isinstance(v, torch.Tensor)}
                converged = False
                for step in range(100):
                    result = self._train_step(batch, epoch, task_idx, view_idx)
                    if not result:
                        break
                    metrics, raw_weights, routing_logits, _ = result
                    if metrics["main_loss"] <= 0.05 and metrics["token_acc"] >= 0.999:
                        self.console.print(f"Task {task_idx} view {view_idx} converged in {step + 1} steps.")
                        converged = True
                        break
                if not converged:
                    self.console.print(f"[red]Task {task_idx} view {view_idx} hit MAX_STEPS.[/red]")
            self.start_view_idx = 0
        self.start_task_idx, self.start_view_idx = 0, 0


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
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
