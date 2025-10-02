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
    def _calculate_mu_grads(
        computation_output_grads: list[torch.Tensor], spl_inputs: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        mu_grads_per_token: list[torch.Tensor] = []
        for grad_act, spl_in in zip(computation_output_grads, spl_inputs, strict=False):
            mu_grads_per_token.append(torch.einsum("bsd,bsh->bsdh", grad_act, spl_in))
        return mu_grads_per_token

    def compute_and_apply_gradients(
        self,
        main_loss: torch.Tensor,
        model_outputs: dict,
        device: torch.device,
        last_routing_logits: list[torch.Tensor] | None,
        last_spl_inputs: list[torch.Tensor] | None,
    ) -> dict[str, Any]:
        self.optimizer_comp.zero_grad()
        self.optimizer_route.zero_grad()

        computation_outputs = model_outputs["computation_outputs"]
        clean_spl_inputs = [s for s in model_outputs["spl_inputs"] if s is not None]

        params_to_grad = self.computation_params + computation_outputs
        all_grads = torch.autograd.grad(main_loss, params_to_grad, retain_graph=True, allow_unused=True)
        comp_grads = all_grads[: len(self.computation_params)]
        intermediate_grads = all_grads[len(self.computation_params) :]

        with torch.no_grad():
            for param, grad in zip(self.computation_params, comp_grads):
                if grad is not None:
                    param.grad = grad.clone()

        mu_grads_unclipped = self._calculate_mu_grads(
            [g for g in intermediate_grads if g is not None], clean_spl_inputs
        )
        all_goodness = []
        with torch.no_grad():
            valid_intermediate_grads = [g for g in intermediate_grads if g is not None]
            for mu_grad, output_grad in zip(mu_grads_unclipped, valid_intermediate_grads):
                norm_importance = mas_normalize(torch.norm(output_grad, p=2, dim=(0, 1)))
                norm_surprise = mas_normalize(torch.norm(mu_grad, p=2, dim=(0, 1, 3)))
                goodness = F.relu(norm_importance - norm_surprise)
                all_goodness.append(goodness)

        meta_losses = []
        for logit, good in zip(model_outputs["routing_logits"], all_goodness):
            if not (logit.numel() > 0 and good.numel() > 0 and logit.shape[-1] == good.shape[-1]):
                continue
            meta_losses.append(self._calculate_jsd_loss(logit, good))

        avg_route_jsd_loss = torch.tensor(0.0, device=device)
        if meta_losses:
            avg_route_jsd_loss = torch.stack(meta_losses).mean()

        total_meta_loss = self.config.w_route_jsd * avg_route_jsd_loss

        carc_losses = []
        if last_routing_logits is not None and last_spl_inputs is not None:
            for curr_l, last_l, curr_in, last_in in zip(
                model_outputs["routing_logits"], last_routing_logits, clean_spl_inputs, last_spl_inputs
            ):
                if not (
                    curr_l is not None
                    and last_l is not None
                    and curr_in is not None
                    and last_in is not None
                    and curr_l.shape == last_l.shape
                    and curr_in.shape == last_in.shape
                ):
                    continue

                sim_p = F.cosine_similarity(curr_l.mean(dim=1), last_l.mean(dim=1), dim=-1).mean()
                sim_x = F.cosine_similarity(curr_in.mean(dim=1), last_in.mean(dim=1), dim=-1).mean()
                carc_losses.append(torch.abs(sim_p - sim_x.detach()))

        avg_carc_loss = torch.tensor(0.0, device=device)
        if carc_losses:
            avg_carc_loss = torch.stack(carc_losses).mean()
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
            "route_jsd_loss": avg_route_jsd_loss,
            "carc_loss": avg_carc_loss,
            "mu_surprises": [torch.norm(g, p=2, dim=(0, 1, 3)) for g in mu_grads_unclipped],
            "goodness_scores": all_goodness,
            "proto_weights": [p.detach() for name, p in self.routing_params_with_names if "proto_weight" in name],
        }


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        self.console = Console()
        self.observer = Observer(self.console, config)
        self.tokenizer = ArcColorTokenizer()
        self.serializer = GridSerializer(self.tokenizer)
        self.consistency_tools = ConsistencyTools()

        self._setup_data()
        self._setup_model_and_optimizer()

        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.global_step, self.epoch, self.start_task_idx, self.start_view_idx = 0, 0, 0, 0
        self.log_cycle_goodness: list[list[torch.Tensor]] = []
        self.log_cycle_act_rates: list[list[float]] = []
        self.last_spl_inputs: list[torch.Tensor] | None = None
        self.replay_queue: collections.deque = collections.deque(maxlen=100)
        self.new_sample_counter = 0
        self.last_routing_logits: list[torch.Tensor] | None = None

    def _setup_data(self):
        train_dataset = InMemoryArcDataset(data_path=self.config.data.data_path, split="training")
        eval_dataset = InMemoryArcDataset(data_path=self.config.data.data_path, split="evaluation")
        collator = ArcCollator(self.tokenizer, max_len=self.config.model.max_position_embeddings)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.data.batch_size,
            collate_fn=collator,
            num_workers=self.config.data.num_workers,
            shuffle=False,
        )
        self.eval_loader = DataLoader(
            eval_dataset, batch_size=1, collate_fn=collator, num_workers=self.config.data.num_workers, shuffle=False
        )

    def _setup_model_and_optimizer(self):
        self.config.model.vocab_size = self.tokenizer.vocab_size
        self.model = ArcTransformer(self.config.model, device=self.device).to(self.device)

        computation_params = [
            p for name, p in self.model.named_parameters() if "proto_weight" not in name and "gate_param" not in name
        ]
        routing_params_with_names = [
            (name, p) for name, p in self.model.named_parameters() if "proto_weight" in name or "gate_param" in name
        ]
        routing_params = [p for _, p in routing_params_with_names]
        self.optimizer_comp = torch.optim.AdamW(computation_params, lr=self.config.lr)
        self.optimizer_route = torch.optim.AdamW(routing_params, lr=self.config.lr)

        self.dynamics = LearningDynamics(
            self.config, computation_params, routing_params_with_names, self.optimizer_comp, self.optimizer_route
        )
        self.evaluator = EvaluationStep(
            self.model,
            self.serializer,
            GridDeserializer(self.tokenizer),
            self.observer,
            self.device,
            self.train_loader.dataset,
            self.config,
        )

    @staticmethod
    def _prepare_batch(
        mini_task: dict,
        view_idx: int,
        device: torch.device,
        serializer: GridSerializer,
        consistency_tools: ConsistencyTools,
        max_len: int,
    ) -> dict[str, torch.Tensor] | None:
        input_grid = torch.tensor(mini_task["input"], device=device)
        output_grid = torch.tensor(mini_task["output"], device=device)
        transform = consistency_tools.get_transforms()[view_idx]
        augmented = {"input": transform(input_grid).cpu().tolist(), "output": transform(output_grid).cpu().tolist()}
        ids, labels, coords = serializer.serialize_mini_task(augmented)
        if len(ids) > max_len:
            return None

        return {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
            "labels": torch.tensor([labels], dtype=torch.long, device=device),
            "coords": torch.tensor([coords], dtype=torch.long, device=device),
            "sample_entropy": torch.tensor(
                [ArcCollator._calculate_sample_entropy(labels)], dtype=torch.float32, device=device
            ),
        }

    def _run_step(
        self, mini_task: dict, view_idx: int, epoch: int, task_idx: int | str, last_view_routing_logits: list | None
    ) -> tuple | None:
        start_time = time.time()
        batch = self._prepare_batch(
            mini_task,
            view_idx,
            self.device,
            self.serializer,
            self.consistency_tools,
            self.config.model.max_position_embeddings,
        )
        if not batch:
            return None

        self.model.train()
        with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16):
            model_outputs = self.model(batch["input_ids"], coords=batch["coords"], return_dict=True)
            main_loss = F.cross_entropy(
                model_outputs["logits"][:, :-1, :].contiguous().view(-1, self.config.model.vocab_size),
                batch["labels"][:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

        if not torch.isfinite(main_loss):
            self.console.print(f"[bold red]NaN detected in main_loss at step {self.global_step}. Aborting step.[/bold red]")
            return None

        signals = self.dynamics.compute_and_apply_gradients(
            main_loss, model_outputs, self.device, self.last_routing_logits, self.last_spl_inputs
        )
        model_outputs.update({"labels": batch["labels"], "sample_entropy": batch["sample_entropy"]})
        metrics = self.observer.calculate_metrics(main_loss, model_outputs, signals, batch["input_ids"], self.model)

        self._update_cycle_metrics(metrics, signals, model_outputs)
        consistency_metrics = self._calculate_consistency_metrics(
            model_outputs.get("routing_logits"),
            last_view_routing_logits,
            model_outputs.get("spl_inputs"),
            self.last_spl_inputs,
        )

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
            consistency_metrics,
            self._reinitialize_dead_prototypes_if_needed,
        )

        self.global_step += 1
        routing_logits = model_outputs.get("routing_logits")
        if routing_logits:
            self.last_routing_logits = [r.detach() for r in routing_logits if r is not None]
        spl_inputs = model_outputs.get("spl_inputs")
        if spl_inputs:
            self.last_spl_inputs = [s.detach() for s in spl_inputs if s is not None]
        torch.cuda.empty_cache()
        return metrics, model_outputs.get("raw_weights"), routing_logits, signals

    def _update_cycle_metrics(self, metrics: dict, signals: dict, model_outputs: dict):
        if "goodness_scores" in signals and signals["goodness_scores"]:
            self.log_cycle_goodness.append(signals["goodness_scores"])
        if "act_rates" in metrics:
            self.log_cycle_act_rates.append(metrics["act_rates"])

    def _calculate_consistency_metrics(
        self,
        current_logits: list | None,
        prev_logits: list | None,
        current_inputs: list | None,
        prev_inputs: list | None,
    ) -> dict | None:
        if not all([current_logits, prev_logits, current_inputs, prev_inputs]):
            return None

        sim_ps, sim_xs = [], []
        for curr_l, prev_l, curr_in, prev_in in zip(
            current_logits, prev_logits, current_inputs, prev_inputs  # type: ignore[arg-type]
        ):
            if curr_l is not None and prev_l is not None and curr_l.shape == prev_l.shape:
                sim_ps.append(F.cosine_similarity(curr_l.mean(dim=1), prev_l.mean(dim=1), dim=-1).mean())
            if curr_in is not None and prev_in is not None and curr_in.shape == prev_in.shape:
                sim_xs.append(F.cosine_similarity(curr_in.mean(dim=1), prev_in.mean(dim=1), dim=-1).mean())

        if not sim_ps or not sim_xs:
            return None
        return {"sim_p": torch.stack(sim_ps).mean().item(), "sim_x": torch.stack(sim_xs).mean().item()}

    def _train_epoch(self, epoch: int):
        dataset = self.train_loader.dataset
        for task_idx in range(self.start_task_idx, len(dataset)):
            mini_task = dataset[task_idx]
            last_view_routing_logits = None
            start_view = self.start_view_idx if task_idx == self.start_task_idx else 0
            for view_idx in range(start_view, 8):
                for step in range(500):
                    result = self._run_step(mini_task, view_idx, epoch, task_idx, last_view_routing_logits)
                    if not result:
                        break
                    metrics, _, routing_logits, _ = result
                    if metrics["main_loss"] <= 0.01 and metrics["token_acc"] >= 1.0:
                        self.console.print(f"Task {task_idx} view {view_idx} converged in {step + 1} steps.")
                        last_view_routing_logits = routing_logits
                        self.replay_queue.append((mini_task, view_idx, task_idx))
                        break
                else:
                    self.console.print(f"[red]Task {task_idx} view {view_idx} hit MAX_STEPS.[/red]")

                self.new_sample_counter += 1
                if self.new_sample_counter % 7 == 0 and self.replay_queue:
                    replay_task, replay_view_idx, replay_task_idx = random.choice(self.replay_queue)
                    self.console.print(
                        f"[cyan]--- Replaying sample (Task -{replay_task_idx} View {replay_view_idx}) ---[/cyan]"
                    )
                    self._run_step(replay_task, replay_view_idx, epoch, f"-{replay_task_idx}", None)

            self.start_view_idx = 0
        self.start_task_idx = 0

    def _reinitialize_dead_prototypes_if_needed(self):
        if not self.log_cycle_goodness or not self.last_spl_inputs or not self.log_cycle_act_rates:
            self.console.print(
                f"[bold yellow]Reinitialization check: 0 reinitialized at step {self.global_step} (No data).[/bold yellow]"
            )
            return

        num_spl_modules = len(self.log_cycle_goodness[0])
        avg_goodness = [
            torch.stack([g[i] for g in self.log_cycle_goodness if i < len(g) and g[i].numel() > 0]).mean(0)
            if any(i < len(g) and g[i].numel() > 0 for g in self.log_cycle_goodness)
            else torch.empty(0, device=self.device)
            for i in range(num_spl_modules)
        ]

        layer_act_rates = [[step_rates[i] for step_rates in self.log_cycle_act_rates] for i in range(num_spl_modules)]
        avg_act_rates = [sum(rates) / len(rates) if rates else 0.0 for rates in layer_act_rates]

        reinit_count = self._reinitialize_dead_prototypes(avg_goodness, self.last_spl_inputs, avg_act_rates)
        self.console.print(
            f"[bold yellow]Reinitialization check: {reinit_count} reinitialized at step {self.global_step}.[/bold yellow]"
        )
        self.log_cycle_goodness.clear()
        self.log_cycle_act_rates.clear()

    @staticmethod
    def _find_redundant_mask(module: nn.Module, act_rate: float) -> torch.Tensor:
        num_experts = module.proto_weight.shape[0]
        dead_mask = torch.zeros(num_experts, dtype=torch.bool, device=module.proto_weight.device)
        if num_experts < 2:
            return dead_mask

        t_max, t_min = 0.999, 0.99
        threshold = t_max - act_rate * (t_max - t_min)
        sim_matrix = F.cosine_similarity(
            module.proto_weight.unsqueeze(1), module.proto_weight.unsqueeze(0), dim=-1
        )

        visited = torch.zeros(num_experts, dtype=torch.bool, device=dead_mask.device)
        for i in range(num_experts):
            if visited[i]:
                continue

            similar_indices = torch.where(sim_matrix[i] > threshold)[0]
            if len(similar_indices) > 1:
                survivor_idx = similar_indices[torch.randint(0, len(similar_indices), (1,)).item()]
                for idx in similar_indices:
                    if idx != survivor_idx:
                        dead_mask[idx] = True
                    visited[idx] = True
            else:
                visited[i] = True
        return dead_mask

    @staticmethod
    def _respawn_prototypes(module: nn.Module, dead_mask: torch.Tensor, spl_inputs: torch.Tensor) -> int:
        num_dead = dead_mask.sum().item()
        if not num_dead:
            return 0

        proto_weight, mu_weight = module.proto_weight, module.mu_weight
        spl_input_flat = spl_inputs.reshape(-1, proto_weight.shape[1])
        if spl_input_flat.shape[0] > 0:
            indices = torch.randint(0, spl_input_flat.shape[0], (num_dead,))
            new_protos = spl_input_flat[indices]
            noise = torch.randn_like(new_protos) * spl_input_flat.std(0) * 0.1
            proto_weight.data[dead_mask] = (new_protos + noise).to(proto_weight.dtype)

            new_mu = torch.empty_like(mu_weight.data[dead_mask])
            nn.init.kaiming_uniform_(new_mu, a=math.sqrt(5))
            mu_weight.data[dead_mask] = new_mu.to(mu_weight.dtype)
        return num_dead

    def _reinitialize_dead_prototypes(
        self, avg_goodness: list[torch.Tensor], spl_inputs: list[torch.Tensor], avg_act_rates: list[float]
    ) -> int:
        reinit_count, spl_idx = 0, 0
        with torch.no_grad():
            for module in self.model.modules():
                if "SparseProtoLinear" in module.__class__.__name__:
                    if spl_idx < len(spl_inputs) and spl_idx < len(avg_act_rates):
                        act_rate = avg_act_rates[spl_idx]
                        dead_mask = self._find_redundant_mask(module, act_rate)
                        reinit_count += self._respawn_prototypes(module, dead_mask, spl_inputs[spl_idx])
                    spl_idx += 1
        return reinit_count

    def train(self):
        self._load_checkpoint()
        self.console.print("[bold green]Starting Training...[/bold green]")
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            self._train_epoch(epoch)
        self.console.print("[bold green]Training Finished.[/bold green]")

    def _save_checkpoint(self, task_idx: int, view_idx: int):
        state = {
            "epoch": self.epoch,
            "step": self.global_step,
            "task_idx": task_idx,
            "view_idx": view_idx,
            "model_state_dict": self.model.state_dict(),
            "optimizer_comp_state_dict": self.optimizer_comp.state_dict(),
            "optimizer_route_state_dict": self.optimizer_route.state_dict(),
        }
        path = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        torch.save(state, path)
        ckpts = sorted(self.checkpoint_dir.glob("*.pt"), key=os.path.getmtime)
        if len(ckpts) > self.config.max_checkpoints:
            os.remove(ckpts[0])

    def _load_checkpoint(self):
        ckpts = sorted(self.checkpoint_dir.glob("*.pt"), key=os.path.getmtime, reverse=True)
        if not ckpts:
            self.console.print("[bold yellow]No checkpoint found.[/bold yellow]")
            return
        for path in ckpts:
            try:
                ckpt = torch.load(path, map_location=self.device)

                state_dict = ckpt["model_state_dict"]
                converted_state_dict = {k.replace(".sbl", ".spl"): v for k, v in state_dict.items()}
                self.model.load_state_dict(converted_state_dict)

                self.optimizer_comp.load_state_dict(ckpt["optimizer_comp_state_dict"])
                self.optimizer_route.load_state_dict(ckpt["optimizer_route_state_dict"])
                self.global_step, self.epoch, self.start_task_idx, self.start_view_idx = (
                    ckpt["step"],
                    ckpt["epoch"],
                    ckpt["task_idx"],
                    ckpt["view_idx"],
                )
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
