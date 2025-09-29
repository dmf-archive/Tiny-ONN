import math
import os
import time
from pathlib import Path
from typing import Any
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from torch.utils.data import DataLoader

from .config import TrainConfig
from .consistency import ConsistencyTools
from .data import ArcCollator, GridDeserializer, GridSerializer, InMemoryArcDataset
from .evaluation import EvaluationStep
from .model import ArcTransformer, mas_normalize
from .observer import Observer
from .tokenizer import ArcColorTokenizer


class LearningDynamics:
    def __init__(self, model: ArcTransformer, optimizer: torch.optim.Optimizer, config: TrainConfig):
        self.model = model
        self.optimizer = optimizer
        self.config = config

    @staticmethod
    @torch.jit.script
    def _calculate_mu_grads(
        computation_output_grads: list[torch.Tensor | None], sbl_inputs: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]:
        mu_grads_per_token: list[torch.Tensor | None] = []
        for grad_act, sbl_in in zip(computation_output_grads, sbl_inputs):
            if grad_act is not None and sbl_in is not None:
                per_token_grad = torch.einsum("bsd,bsh->bsdh", grad_act, sbl_in)
                mu_grads_per_token.append(per_token_grad)
            else:
                mu_grads_per_token.append(None)
        return mu_grads_per_token

    def _compute_goodness_per_module(
        self, mu_grads_per_token: torch.Tensor | None, computation_output_grads: torch.Tensor | None
    ) -> torch.Tensor | None:
        if mu_grads_per_token is None or computation_output_grads is None:
            return None

        surprise_vec = torch.norm(mu_grads_per_token, p=2, dim=(0, 1, 3)).detach()
        importance_vec = torch.norm(computation_output_grads, p=2, dim=(0, 1)).detach()

        norm_surprise = mas_normalize(surprise_vec)
        norm_importance = mas_normalize(importance_vec)
        
        goodness = norm_importance - norm_surprise
        return goodness

    def compute_and_apply_gradients(
        self, main_loss: torch.Tensor, model_outputs: dict, device: torch.device
    ) -> dict[str, Any]:
        sbl_inputs = model_outputs["sbl_inputs"]
        masked_outputs = model_outputs["masked_outputs"]
        routing_logits = model_outputs["routing_logits"]

        main_loss_grads = torch.autograd.grad(
            main_loss,
            masked_outputs + sbl_inputs,
            retain_graph=True,
            allow_unused=True,
        )
        masked_output_grads = main_loss_grads[: len(masked_outputs)]
        sbl_input_grads = main_loss_grads[len(masked_outputs) :]
        
        all_goodness = []
        for masked_grad, sbl_in_grad in zip(masked_output_grads, sbl_input_grads):
            if masked_grad is not None and sbl_in_grad is not None:
                importance_vec = torch.norm(masked_grad, p=2, dim=(0, 1)).detach()
                surprise_vec = torch.norm(sbl_in_grad, p=2, dim=(0, 1)).detach()

                norm_importance = mas_normalize(importance_vec)
                norm_surprise = mas_normalize(surprise_vec)
                
                goodness = norm_importance - norm_surprise
                all_goodness.append(goodness)
            else:
                all_goodness.append(torch.empty(0, device=device))

        total_route_jsd_loss = torch.tensor(0.0, device=device)
        num_valid_losses = 0
        for logits, goodness in zip(routing_logits, all_goodness, strict=True):
            if logits.numel() > 0 and goodness.numel() > 0 and logits.shape[-1] == goodness.shape[-1]:
                p_dist = mas_normalize(logits)
                q_dist = mas_normalize(goodness).detach()

                m_dist = 0.5 * (p_dist + q_dist)
                
                kl_p_m = torch.sum(p_dist * (torch.log(p_dist + 1e-9) - torch.log(m_dist + 1e-9)), dim=-1)
                kl_q_m = torch.sum(q_dist * (torch.log(q_dist + 1e-9) - torch.log(m_dist + 1e-9)), dim=-1)

                jsd_loss = (0.5 * kl_p_m + 0.5 * kl_q_m).mean()

                if torch.isfinite(jsd_loss):
                    total_route_jsd_loss += jsd_loss
                    num_valid_losses += 1
        
        avg_route_jsd_loss = total_route_jsd_loss / num_valid_losses if num_valid_losses > 0 else torch.tensor(0.0, device=device)
        
        total_loss = main_loss + self.config.w_route_kl * avg_route_jsd_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        proto_weights = [p.detach() for name, p in self.model.named_parameters() if 'proto_weight' in name]
        
        return {
            "route_kl_loss": avg_route_jsd_loss,
            "mu_surprises": sbl_input_grads,
            "goodness_scores": all_goodness,
            "proto_weights": proto_weights,
        }


def data_driven_init(trainer: "Trainer"):
    if hasattr(trainer.model, "is_data_initialized") and trainer.model.is_data_initialized:
        return
    trainer.console.print("[bold yellow]Performing one-time data-driven initialization...[/bold yellow]")

    mini_task = trainer.train_loader.dataset[0]
    batch = prepare_batch(
        mini_task,
        0,
        trainer.device,
        trainer.serializer,
        trainer.consistency_tools,
        trainer.config.model.max_position_embeddings,
    )
    if not batch:
        trainer.console.print("[bold red]Failed to get a valid batch for init.[/bold red]")
        return

    with torch.no_grad():
        initial_embeddings = trainer.model.embedding(batch["input_ids"]).squeeze(0)

    for name, param in trainer.model.named_parameters():
        if "proto_weight" in name:
            num_experts, in_features = param.shape
            new_data = torch.empty_like(param.data)

            if in_features == initial_embeddings.shape[1]:
                indices = torch.randint(0, initial_embeddings.shape[0], (num_experts,))
                new_data = initial_embeddings[indices]
            elif in_features > initial_embeddings.shape[1]:
                num_repeats = (in_features + initial_embeddings.shape[1] - 1) // initial_embeddings.shape[1]
                for i in range(num_experts):
                    indices = torch.randint(0, initial_embeddings.shape[0], (num_repeats,))
                    full_embed = torch.cat([initial_embeddings[idx] for idx in indices], dim=-1)
                    new_data[i] = full_embed[:in_features]
            param.data.copy_(new_data)

    trainer.model.is_data_initialized = True
    trainer.console.print("[bold green]Data-driven initialization complete.[/bold green]")


def prepare_batch(
    mini_task: dict, view_idx: int, device: torch.device, serializer, consistency_tools, max_len
) -> dict[str, torch.Tensor] | None:
    input_grid = torch.tensor(mini_task["input"], device=device)
    output_grid = torch.tensor(mini_task["output"], device=device)
    transform = consistency_tools.get_transforms()[view_idx]
    augmented = {"input": transform(input_grid).cpu().tolist(), "output": transform(output_grid).cpu().tolist()}
    ids, labels = serializer.serialize_mini_task(augmented)
    if len(ids) > max_len:
        return None
    
    entropy = ArcCollator._calculate_sample_entropy(labels)
    
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "labels": torch.tensor([labels], dtype=torch.long, device=device),
        "sample_entropy": torch.tensor([entropy], dtype=torch.float32, device=device)
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

        train_dataset = InMemoryArcDataset(data_path=config.data.data_path, split="training")
        eval_dataset = InMemoryArcDataset(data_path=config.data.data_path, split="evaluation")
        collator = ArcCollator(self.tokenizer, max_len=config.model.max_position_embeddings)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            collate_fn=collator,
            num_workers=config.data.num_workers,
            shuffle=False,
        )
        self.eval_loader = DataLoader(
            eval_dataset, batch_size=1, collate_fn=collator, num_workers=config.data.num_workers, shuffle=False
        )

        self.config.model.vocab_size = self.tokenizer.vocab_size
        self.model = ArcTransformer(self.config.model, device=self.device).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.dynamics = LearningDynamics(self.model, self.optimizer, config)
        self.evaluator = EvaluationStep(
            self.model,
            self.serializer,
            GridDeserializer(self.tokenizer),
            self.observer,
            self.device,
            train_dataset,
            self.config,
        )

        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.global_step, self.epoch, self.start_task_idx, self.start_view_idx = 0, 0, 0, 0
        self.log_cycle_act_rates = []
        self.replay_queue = collections.deque(maxlen=100)
        self.new_sample_counter = 0

    def _run_step(
        self,
        mini_task: dict,
        view_idx: int,
        epoch: int,
        task_idx: int,
        last_view_routing_logits: list[torch.Tensor] | None,
    ):
        start_time = time.time()
        batch = prepare_batch(
            mini_task,
            view_idx,
            self.device,
            self.serializer,
            self.consistency_tools,
            self.config.model.max_position_embeddings,
        )
        if not batch:
            return None, None, None, None
        input_ids, labels, sample_entropy = batch["input_ids"], batch["labels"], batch["sample_entropy"]

        self.model.train()

        with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16):
            model_outputs = self.model(input_ids, return_dict=True)
            main_loss = F.cross_entropy(
                model_outputs["logits"][:, :-1, :].contiguous().view(-1, self.config.model.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

        if not torch.isfinite(main_loss):
            self.console.print(f"[bold red]NaN detected in main_loss at step {self.global_step}. Aborting step.[/bold red]")
            return None, None

        signals = self.dynamics.compute_and_apply_gradients(main_loss, model_outputs, self.device)
        model_outputs["labels"] = labels
        model_outputs["sample_entropy"] = sample_entropy
        
        metrics = self.observer.calculate_metrics(main_loss, model_outputs, signals, input_ids, self.model)
        if "act_rates" in metrics:
            self.log_cycle_act_rates.append(metrics["act_rates"])

        consistency_metrics = None
        if last_view_routing_logits is not None:
            all_cos_sims, all_euc_dists = [], []
            for curr_logits, prev_logits in zip(model_outputs["routing_logits"], last_view_routing_logits, strict=True):
                if curr_logits.numel() > 0 and prev_logits.numel() > 0:
                    # FIX: Use mas_normalize instead of softmax for consistency with the model's activation function
                    curr_mean_dist = mas_normalize(curr_logits).mean(dim=1).squeeze(0)
                    prev_mean_dist = mas_normalize(prev_logits).mean(dim=1).squeeze(0)
                    
                    if curr_mean_dist.shape != prev_mean_dist.shape: continue

                    cos_sim = F.cosine_similarity(curr_mean_dist, prev_mean_dist, dim=0)
                    euc_dist = torch.cdist(curr_mean_dist.unsqueeze(0).to(torch.float32), prev_mean_dist.unsqueeze(0).to(torch.float32))
                    
                    all_cos_sims.append(cos_sim)
                    all_euc_dists.append(euc_dist)

            if all_cos_sims:
                consistency_metrics = {
                    "cos_sim": torch.stack(all_cos_sims).mean().item(),
                    "euc_dist": torch.stack(all_euc_dists).mean().item()
                }
        
        signals["raw_weights"] = model_outputs["raw_weights"]
        self.observer.maybe_log_and_visualize(
            epoch,
            self.global_step,
            task_idx,
            view_idx,
            metrics,
            time.time() - start_time,
            signals,
            self.evaluator,
            self.eval_loader,
            task_idx,
            self._save_checkpoint,
            consistency_metrics,
            self._reinitialize_dead_prototypes_if_needed,
        )

        self.global_step += 1
        torch.cuda.empty_cache()
        return metrics, model_outputs["raw_weights"], model_outputs["routing_logits"], signals

    def _train_epoch(self, epoch: int):
        dataset = self.train_loader.dataset
        for task_idx in range(self.start_task_idx, len(dataset)):
            mini_task = dataset[task_idx]
            last_view_routing_logits = None
            start_view = self.start_view_idx if task_idx == self.start_task_idx else 0
            for view_idx in range(start_view, 8):
                step, MAX_STEPS = 0, 500
                view_last_raw_weights = None
                while step < MAX_STEPS:
                    metrics, raw_weights, routing_logits, signals = self._run_step(
                        mini_task, view_idx, epoch, task_idx, last_view_routing_logits
                    ) or (None, None, None, None)
                    
                    if raw_weights:
                        view_last_raw_weights = raw_weights
                    if not metrics:
                        self.console.print(f"[yellow]Skipping task {task_idx} view {view_idx}.[/yellow]")
                        break
                    if metrics.get("token_acc", 0.0) >= 1.0:
                        self.console.print(f"Task {task_idx} view {view_idx} converged in {step + 1} steps.")
                        if view_last_raw_weights:
                            last_view_routing_logits = routing_logits
                        self.replay_queue.append((mini_task, view_idx, task_idx))
                        break
                    step += 1
                if step == MAX_STEPS:
                    self.console.print(f"[red]Task {task_idx} view {view_idx} hit MAX_STEPS.[/red]")

                self.new_sample_counter += 1
                if self.new_sample_counter % 7 == 0 and self.replay_queue:
                    replay_task, replay_view_idx, replay_task_idx = random.choice(self.replay_queue)
                    self.console.print(f"[cyan]--- Replaying sample (Task -{replay_task_idx} View {replay_view_idx}) ---[/cyan]")
                    self._run_step(replay_task, replay_view_idx, epoch, f"-{replay_task_idx}", None)

            self.start_view_idx = 0
        self.start_task_idx = 0
    
    def _reinitialize_dead_prototypes_if_needed(self):
        if not self.log_cycle_act_rates: return

        avg_act_rates = []
        num_spl_layers = len(self.log_cycle_act_rates[0])
        for i in range(num_spl_layers):
            layer_rates = [step_rates[i] for step_rates in self.log_cycle_act_rates if i < len(step_rates)]
            if layer_rates:
                avg_act_rates.append(sum(layer_rates) / len(layer_rates))
            else:
                avg_act_rates.append(0.0)
        
        reinit_count = self._reinitialize_dead_prototypes(avg_act_rates)
        if reinit_count > 0:
            self.console.print(
                f"[bold yellow]Reinitialized {reinit_count} dead prototypes at step {self.global_step}.[/bold yellow]"
            )
        self.log_cycle_act_rates = []

    def _reinitialize_dead_prototypes(self, avg_act_rates: list[float], k: int = 1) -> int:
        reinitialized_count = 0
        spl_layer_idx = 0
        with torch.no_grad():
            for block in self.model.blocks:
                for module_name, module in block.named_modules():
                    if isinstance(module, torch.nn.Module) and hasattr(module, "proto_weight"):
                        if spl_layer_idx >= len(avg_act_rates): continue
                        
                        act_rate = avg_act_rates[spl_layer_idx]
                        proto_threshold = 0.99 * (1.0 - act_rate) + 0.01 * 0.90
                        
                        proto_weight = module.proto_weight
                        mu_weight = module.mu_weight
                        num_experts = proto_weight.shape[0]
                        
                        proto_sim = F.cosine_similarity(proto_weight.unsqueeze(1), proto_weight.unsqueeze(0), dim=-1)
                        
                        dead_mask = torch.zeros(num_experts, dtype=torch.bool, device=self.device)
                        visited = torch.zeros(num_experts, dtype=torch.bool, device=self.device)
                        
                        for i in range(num_experts):
                            if visited[i]: continue
                            
                            similar_indices = torch.where(proto_sim[i] > proto_threshold)[0]
                            
                            if len(similar_indices) > k:
                                survivor_idx = similar_indices[torch.randint(0, len(similar_indices), (1,)).item()]
                                
                                for idx in similar_indices:
                                    if idx != survivor_idx:
                                        dead_mask[idx] = True
                                    visited[idx] = True
                            else:
                                visited[i] = True
                        
                        num_dead = dead_mask.sum().item()

                        if num_dead > 0:
                            reinitialized_count += num_dead
                            live_mask = ~dead_mask
                            
                            if live_mask.any():
                                live_protos = proto_weight[live_mask]
                                center = live_protos.mean(dim=0)
                                distances = torch.norm(live_protos - center, p=2, dim=-1)
                                radius = distances.mean() if distances.numel() > 0 else 1.0
                                
                                new_protos = center + torch.randn_like(proto_weight[dead_mask]) * radius
                                proto_weight.data[dead_mask] = new_protos

                                new_mu = torch.empty_like(mu_weight.data[dead_mask])
                                torch.nn.init.kaiming_uniform_(new_mu, a=math.sqrt(5))
                                mu_weight.data[dead_mask] = new_mu
                            else:
                                torch.nn.init.kaiming_uniform_(proto_weight.data, a=math.sqrt(5))
                                torch.nn.init.kaiming_uniform_(mu_weight.data, a=math.sqrt(5))
                        
                        spl_layer_idx += 1

        return reinitialized_count

    def train(self):
        self._load_checkpoint()
        data_driven_init(self)
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
            "optimizer_state_dict": self.optimizer.state_dict(),
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
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                self.global_step = ckpt["step"]
                self.epoch = ckpt["epoch"]
                self.start_task_idx = ckpt["task_idx"]
                self.start_view_idx = ckpt["view_idx"]
                self.console.print(f"[bold green]Loaded checkpoint from {path} at step {self.global_step}.[/bold green]")
                return
            except Exception as e:
                self.console.print(f"[bold red]Corrupted checkpoint {path}: {e}. Trying next.[/bold red]")
                os.remove(path)
        self.console.print("[bold yellow]No valid checkpoint found.[/bold yellow]")


def main():
    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
