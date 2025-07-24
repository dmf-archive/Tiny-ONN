import os
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    get_scheduler,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from tiny_onn.modular import TinyOnnForCausalLM, TinyOnnMoE

from .callbacks import MatplotlibVisualizer, TensorBoardLogger
from .config import TrainConfig


class TrainerEngine:
    def __init__(
        self,
        config: TrainConfig,
        model: PreTrainedModel,
        optimizer_experts: Optimizer,
        optimizer_router: Optimizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        device: torch.device,
    ):
        self.config = config
        self.model = cast(TinyOnnForCausalLM, model)
        self.optimizer_experts = optimizer_experts
        self.optimizer_router = optimizer_router
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.grad_scaler = torch.amp.GradScaler(enabled=self.config.training.use_amp)
        self.teacher_model = None
        if config.model.teacher_model_name:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                config.model.teacher_model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
            self.teacher_model.eval()

        self.output_dir = Path(config.training.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.img_dir = self.output_dir / "img"

        self.logger = TensorBoardLogger(self.log_dir)
        self.visualizer = MatplotlibVisualizer(
            self.img_dir, config.model.base_model_name
        )
        self.metrics_cache: defaultdict[str, list[float]] = defaultdict(list)
        self.expert_data_cache: defaultdict[str, list[np.ndarray]] = defaultdict(list)
        self.global_step = 0
        self.start_epoch = 0

        self.lr_scheduler_experts = get_scheduler(
            name="linear",
            optimizer=self.optimizer_experts,
            num_warmup_steps=config.training.lr_scheduler_warmup_steps,
            num_training_steps=len(self.train_dataloader)
            * config.training.num_train_epochs,
        )
        self.lr_scheduler_router = get_scheduler(
            name="linear",
            optimizer=self.optimizer_router,
            num_warmup_steps=config.training.lr_scheduler_warmup_steps,
            num_training_steps=len(self.train_dataloader)
            * config.training.num_train_epochs,
        )

    def _get_expert_params(self) -> list[torch.Tensor]:
        expert_params: list[torch.Tensor] = []
        for layer in self.model.model.layers:
            moe_layer = cast(TinyOnnMoE, layer.mlp)
            for expert in moe_layer.experts:
                expert_params.extend(list(expert.parameters()))
        return expert_params

    def _hyper_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        inputs = {k: v.to(self.device) for k, v in batch.items() if v is not None}
        if self.model.config.num_experts_per_tok == -1:
            inputs["surprise_budget"] = self.config.training.pi_alpha

        # Stage 1: Expert Forward and Backward Pass
        self.optimizer_experts.zero_grad()
        self.optimizer_router.zero_grad()

        with torch.amp.autocast("cuda", enabled=self.config.training.use_amp):
            student_outputs: CausalLMOutputWithPast = self.model(**inputs)
            main_loss = student_outputs.loss

        assert main_loss is not None
        self.grad_scaler.scale(main_loss).backward(retain_graph=True)
        self.grad_scaler.step(self.optimizer_experts)

        # Stage 2: Router Meta-Learning
        num_tokens = inputs.get("input_ids", torch.empty(0)).numel()
        num_experts = self.model.config.num_experts_per_layer
        
        all_router_logits = []
        all_optimal_expert_indices = []
        total_surprise = 0.0
        num_active_tokens = 0

        for layer in self.model.model.layers:
            moe_layer = cast(TinyOnnMoE, layer.mlp)
            if not moe_layer.last_expert_token_indices or not moe_layer.last_expert_inputs:
                continue

            per_token_surprise = torch.full((num_tokens, num_experts), float("inf"), device=self.device)
            
            expert_params_for_grad = [p for idx in moe_layer.last_expert_inputs for p in moe_layer.experts[idx].parameters()]
            
            if not expert_params_for_grad:
                continue

            expert_grads = torch.autograd.grad(outputs=main_loss, inputs=expert_params_for_grad, retain_graph=True)
            
            param_counter = 0
            for expert_idx, expert_input in moe_layer.last_expert_inputs.items():
                token_indices = moe_layer.last_expert_token_indices[expert_idx]
                expert = moe_layer.experts[expert_idx]
                
                w1_grad = expert_grads[param_counter]
                w3_grad = expert_grads[param_counter+1]
                w2_grad = expert_grads[param_counter+2]
                param_counter += 3
                
                # Reconstruct per-token gradients using einsum
                # This is the core of the per-token surprise calculation
                act_fn_derivative = expert.act_fn(expert.w1(expert_input)) * expert.w3(expert_input)
                
                # Einsum notation: b: batch (token), i: intermediate_dim, j: hidden_dim
                # Calculate per-token gradients for w1 and w3
                grad_w1_contribution = torch.einsum("bi,bj->bij", act_fn_derivative, expert_input)
                grad_w3_contribution = torch.einsum("bi,bj->bij", expert.act_fn(expert.w1(expert_input)), expert_input)

                # The gradient w.r.t. w1 and w3 is influenced by w2
                grad_w1_per_token = torch.matmul(grad_w1_contribution, w2_grad.T)
                grad_w3_per_token = torch.matmul(grad_w3_contribution, w2_grad.T)

                # Einsum notation for w2: o: output_dim (hidden_dim), i: intermediate_dim, b: batch (token)
                grad_w2_per_token = torch.einsum("oi,bi->bio", w2_grad, act_fn_derivative)
                
                surprise = torch.linalg.norm(grad_w1_per_token.flatten(1), dim=1) + \
                           torch.linalg.norm(grad_w3_per_token.flatten(1), dim=1) + \
                           torch.linalg.norm(grad_w2_per_token.flatten(1), dim=1)
                
                per_token_surprise[token_indices, expert_idx] = surprise

            valid_surprise_mask = ~torch.isinf(per_token_surprise).all(dim=1)
            if valid_surprise_mask.any():
                optimal_indices = torch.argmin(per_token_surprise[valid_surprise_mask], dim=1)
                
                current_surprise = per_token_surprise[valid_surprise_mask]
                total_surprise += current_surprise[torch.arange(current_surprise.size(0)), optimal_indices].sum().item()
                num_active_tokens += valid_surprise_mask.sum().item()
                
                if moe_layer.last_router_logits is not None:
                    router_logits_for_loss = moe_layer.last_router_logits[valid_surprise_mask]
                    all_router_logits.append(router_logits_for_loss)
                    all_optimal_expert_indices.append(optimal_indices)

        router_loss = torch.tensor(0.0, device=self.device)
        if all_router_logits:
            concatenated_logits = torch.cat(all_router_logits, dim=0).float()
            concatenated_indices = torch.cat(all_optimal_expert_indices, dim=0)
            router_loss = F.cross_entropy(concatenated_logits, concatenated_indices)

        if router_loss.requires_grad:
            self.grad_scaler.scale(router_loss).backward()

        self.grad_scaler.step(self.optimizer_router)
        self.grad_scaler.update()

        self.lr_scheduler_experts.step()
        self.lr_scheduler_router.step()

        # Metrics Calculation
        with torch.no_grad():
            labels = inputs.get("labels")
            main_acc = 0.0
            if student_outputs.logits is not None and labels is not None:
                main_acc = ((student_outputs.logits.argmax(-1) == labels).float().mean().item())

            tau = 0.0
            if student_outputs.logits is not None:
                tau = torch.distributions.Categorical(logits=student_outputs.logits.float()).entropy().mean().item()

            gating_acc = 0.0
            avg_surprise = 0.0

            pi_score = 0.0
            if tau > 0:
                pi_score_tensor = torch.exp(
                    -self.config.training.pi_alpha * (main_loss / tau + self.config.training.pi_gamma * avg_surprise)
                )
                pi_score = pi_score_tensor.item()

        metrics = {
            "main_loss": main_loss.item(),
            "router_loss": router_loss.item() if isinstance(router_loss, torch.Tensor) else router_loss,
            "main_acc": main_acc,
            "gating_acc": gating_acc,
            "surprise": avg_surprise,
            "tau": tau,
            "pi_score": pi_score,
        }
        for k, v in metrics.items():
            self.metrics_cache[k].append(v)
        return metrics

    def _save_checkpoint(self, epoch: int):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = (
            self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        )

        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_experts_state_dict": self.optimizer_experts.state_dict(),
                "optimizer_router_state_dict": self.optimizer_router.state_dict(),
                "lr_scheduler_experts_state_dict": self.lr_scheduler_experts.state_dict(),
                "lr_scheduler_router_state_dict": self.lr_scheduler_router.state_dict(),
            },
            checkpoint_path,
        )

        checkpoints = sorted(
            self.checkpoint_dir.glob("*.pt"), key=os.path.getmtime
        )
        if len(checkpoints) > self.config.logging.rolling_checkpoint_count:
            os.remove(checkpoints[0])

    def _load_checkpoint(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_experts.load_state_dict(
            checkpoint["optimizer_experts_state_dict"]
        )
        self.optimizer_router.load_state_dict(
            checkpoint["optimizer_router_state_dict"]
        )
        self.lr_scheduler_experts.load_state_dict(
            checkpoint["lr_scheduler_experts_state_dict"]
        )
        self.lr_scheduler_router.load_state_dict(
            checkpoint["lr_scheduler_router_state_dict"]
        )
        self.global_step = checkpoint["global_step"]
        self.start_epoch = checkpoint["epoch"]
        print(
            f"Resumed from checkpoint {checkpoint_path} at epoch {self.start_epoch} and global step {self.global_step}"
        )

    def train(self, max_steps: int | None = None):
        for epoch in range(
            self.start_epoch, self.config.training.num_train_epochs
        ):
            self.model.train()
            progress_bar = tqdm(
                self.train_dataloader, desc=f"Epoch {epoch + 1}"
            )
            for batch in progress_bar:
                if max_steps is not None and self.global_step >= max_steps:
                    self.logger.close()
                    return

                metrics = self._hyper_step(batch)
                progress_bar.set_postfix(metrics)

                if self.global_step % self.config.logging.log_interval == 0:
                    self.logger.log_metrics(metrics, self.global_step)

                if self.global_step % self.config.logging.eval_interval == 0:
                    self.visualizer.update_plots(
                        self.metrics_cache,
                        self.expert_data_cache,  # type: ignore
                        self.global_step,
                        self.model,
                    )

                if (
                    self.global_step % self.config.logging.checkpoint_interval
                    == 0
                ):
                    self._save_checkpoint(epoch)

                self.global_step += 1

        self.logger.close()

    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()

        avg_loss = total_loss / len(self.eval_dataloader)
        return {
            "eval_loss": avg_loss,
            "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
        }
