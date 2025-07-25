import os
from pathlib import Path
from typing import Any, cast

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

from .callbacks import TensorBoardLogger
from .config import TrainConfig
from .plotter import LivePlotter


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
        self.grad_scaler = None
        self.last_main_loss: float | None = None

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
        self.plotter = LivePlotter(self.img_dir)

        self.metrics_history: list[dict[str, Any]] = []
        self.expert_data: dict[str, np.ndarray] = {
            "selected_experts_steps": np.array([], dtype=int),
            "selected_experts_values": np.array([], dtype=int),
            "optimal_experts_steps": np.array([], dtype=int),
            "optimal_experts_values": np.array([], dtype=int),
        }

        self.global_step = 0
        self.start_epoch = 0

        self.surprise_ema = {"mean": 0.0, "var": 1.0, "count": 0}
        self.cost_ema = {"mean": 0.0, "var": 1.0, "count": 0}

        num_training_steps = len(self.train_dataloader) * config.training.num_train_epochs
        self.lr_scheduler_experts = get_scheduler(
            name="linear",
            optimizer=self.optimizer_experts,
            num_warmup_steps=config.training.lr_scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )
        self.lr_scheduler_router = get_scheduler(
            name="linear",
            optimizer=self.optimizer_router,
            num_warmup_steps=config.training.lr_scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _update_expert_data(
        self,
        all_selected_experts: list[tuple[int, torch.Tensor]],
        optimal_indices: torch.Tensor,
        all_router_logits: list[torch.Tensor]
    ):
        num_experts_per_layer = self.model.config.num_experts_per_layer

        # Selected experts
        selected_experts_list = []
        for layer_idx, experts_tensor in all_selected_experts:
            global_expert_ids = (experts_tensor + layer_idx * num_experts_per_layer).cpu().numpy()
            selected_experts_list.append(global_expert_ids)

        if selected_experts_list:
            all_selected = np.concatenate(selected_experts_list).flatten()
            self.expert_data["selected_experts_values"] = np.concatenate([self.expert_data["selected_experts_values"], all_selected])
            steps = np.full(all_selected.shape, self.global_step)
            self.expert_data["selected_experts_steps"] = np.concatenate([self.expert_data["selected_experts_steps"], steps])

        # Optimal experts
        token_to_layer_map = []
        for i, layer in enumerate(self.model.model.layers):
            moe_layer = cast(TinyOnnMoE, layer.mlp)
            if moe_layer.last_router_logits is not None:
                num_layer_tokens = moe_layer.last_router_logits.shape[0]
                token_to_layer_map.extend([i] * num_layer_tokens)

        flat_optimal_indices = optimal_indices.cpu().numpy()
        global_optimal_experts = np.array([
            local_expert_idx + token_to_layer_map[i] * num_experts_per_layer
            for i, local_expert_idx in enumerate(flat_optimal_indices)
        ])

        self.expert_data["optimal_experts_values"] = np.concatenate([self.expert_data["optimal_experts_values"], global_optimal_experts])
        steps = np.full(global_optimal_experts.shape, self.global_step)
        self.expert_data["optimal_experts_steps"] = np.concatenate([self.expert_data["optimal_experts_steps"], steps])


    def _hyper_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        inputs = {k: v.to(self.device) for k, v in batch.items() if v is not None}

        self.optimizer_experts.zero_grad()
        self.optimizer_router.zero_grad()

        surprise_context: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        forward_kwargs = {"surprise_context": surprise_context}

        student_outputs: CausalLMOutputWithPast = self.model(**inputs, **forward_kwargs) # type: ignore
        main_loss = student_outputs.loss
        assert main_loss is not None

        main_loss.backward(retain_graph=True)

        all_router_logits, all_selected_experts, all_k_per_token = [], [], []
        layer_token_offsets: dict[int, int] = {}
        current_offset = 0
        for i, layer in enumerate(self.model.model.layers):
            moe_layer = cast(TinyOnnMoE, layer.mlp)
            if moe_layer.last_router_logits is not None:
                all_router_logits.append(moe_layer.last_router_logits)
                layer_token_offsets[i] = current_offset
                current_offset += moe_layer.last_router_logits.shape[0]
            if moe_layer.last_selected_experts is not None:
                all_selected_experts.append((i, moe_layer.last_selected_experts))
            if moe_layer.last_k_per_token is not None:
                all_k_per_token.append(moe_layer.last_k_per_token)

        router_loss = torch.tensor(0.0, device=self.device)
        gating_acc = 0.0
        per_token_surprise = torch.tensor([], device=self.device)

        if all_router_logits:
            concatenated_logits = torch.cat(all_router_logits, dim=0)
            per_token_surprise = torch.full_like(concatenated_logits, float("inf"))

            for (layer_idx, expert_idx), (token_indices, surprise) in surprise_context.items():
                if layer_idx in layer_token_offsets:
                    offset = layer_token_offsets[layer_idx]
                    original_dtype = per_token_surprise.dtype
                    per_token_surprise[offset + token_indices, expert_idx] = surprise.to(original_dtype)

            optimal_indices = torch.argmin(per_token_surprise, dim=1)
            smk_loss = F.cross_entropy(concatenated_logits, optimal_indices)
            
            avg_k_tensor = torch.cat([k.float() for k in all_k_per_token]).mean()
            avg_k_loss = avg_k_tensor.clone()
            
            router_loss = smk_loss + self.config.training.router_loss_lambda * avg_k_loss

            with torch.no_grad():
                predicted_indices = torch.argmin(concatenated_logits, dim=1)
                gating_acc = (predicted_indices == optimal_indices).float().mean().item()
                self._update_expert_data(all_selected_experts, optimal_indices, all_router_logits)

        if router_loss.requires_grad:
            router_loss.backward()

        self.optimizer_experts.step()
        self.optimizer_router.step()

        self.lr_scheduler_experts.step()
        self.lr_scheduler_router.step()

        with torch.no_grad():
            if student_outputs.logits is not None and "labels" in inputs:
                main_acc = (student_outputs.logits.argmax(-1) == inputs["labels"]).float().mean().item()
                tau = torch.distributions.Categorical(logits=student_outputs.logits.float()).entropy().mean().item()
            else:
                main_acc = 0.0
                tau = 1.0
            valid_surprise = per_token_surprise[torch.isfinite(per_token_surprise)]
            surprise_val = valid_surprise.mean().item() if valid_surprise.numel() > 0 else 0.0

            cost = main_loss.item() / tau
            pi_score = torch.exp(torch.tensor(-self.config.training.pi_alpha * cost)).item()

            avg_k_float = avg_k_tensor.item() if all_k_per_token else 0.0

        metrics: dict[str, Any] = {
            "type": "train",
            "step": self.global_step,
            "main_loss": main_loss.item(),
            "router_loss": router_loss.item(),
            "smk_loss": smk_loss.item(),
            "avg_k_loss": avg_k_loss.item(),
            "main_acc": main_acc,
            "gating_acc": gating_acc,
            "surprise": surprise_val,
            "tau": float(tau),
            "pi_score": pi_score,
            "avg_k": avg_k_float,
        }
        self.metrics_history.append(metrics)
        self.last_main_loss = main_loss.item()

        if torch.cuda.is_available():
            metrics["cuda_mem_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            metrics["cuda_mem_peak_mb"] = torch.cuda.max_memory_allocated() / 1024**2

        return {k: v for k, v in metrics.items() if isinstance(v, int | float)}


    def _save_checkpoint(self, epoch: int):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save({
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_experts_state_dict": self.optimizer_experts.state_dict(),
            "optimizer_router_state_dict": self.optimizer_router.state_dict(),
            "lr_scheduler_experts_state_dict": self.lr_scheduler_experts.state_dict(),
            "lr_scheduler_router_state_dict": self.lr_scheduler_router.state_dict(),
        }, checkpoint_path)
        checkpoints = sorted(self.checkpoint_dir.glob("*.pt"), key=os.path.getmtime)
        if len(checkpoints) > self.config.logging.rolling_checkpoint_count:
            os.remove(checkpoints[0])

    def _load_checkpoint(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_experts.load_state_dict(checkpoint["optimizer_experts_state_dict"])
        self.optimizer_router.load_state_dict(checkpoint["optimizer_router_state_dict"])
        self.lr_scheduler_experts.load_state_dict(checkpoint["lr_scheduler_experts_state_dict"])
        self.lr_scheduler_router.load_state_dict(checkpoint["lr_scheduler_router_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.start_epoch = checkpoint["epoch"]
        print(f"Resumed from checkpoint {checkpoint_path} at epoch {self.start_epoch} and global step {self.global_step}")

    def train(self, max_steps: int | None = None):
        self.model.train()
        for epoch in range(self.start_epoch, self.config.training.num_train_epochs):
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            for batch in progress_bar:
                if max_steps is not None and self.global_step >= max_steps:
                    self.logger.close()
                    return

                metrics = self._hyper_step(batch)
                progress_bar.set_postfix(metrics)

                if self.global_step % self.config.logging.log_interval == 0:
                    self.logger.log_metrics(metrics, self.global_step)

                if self.global_step > 0 and self.global_step % self.config.logging.eval_interval == 0:
                    eval_metrics = self.evaluate()
                    self.metrics_history.append(eval_metrics)
                    self.logger.log_metrics(eval_metrics, self.global_step)

                if self.global_step > 0 and self.global_step % self.config.logging.plot_interval == 0:
                    self.plotter.plot_metrics_dashboard(self.metrics_history, self.global_step)
                    self.plotter.plot_expert_dashboard(self.expert_data, self.global_step, self.model.config)

                if self.global_step > 0 and self.global_step % self.config.logging.checkpoint_interval == 0:
                    self._save_checkpoint(epoch)

                self.global_step += 1

        self.logger.close()

    def evaluate(self) -> dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_tokens = torch.tensor(0, device=self.device)
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                inputs = {k: v.to(self.device) for k, v in batch.items() if v is not None}
                outputs = self.model(**inputs)
                if outputs.loss is not None:
                    loss = outputs.loss
                    num_tokens = inputs["attention_mask"].sum()
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens
                    if outputs.logits is not None and "labels" in inputs:
                        total_acc += ((outputs.logits.argmax(-1) == inputs["labels"]) * inputs["attention_mask"]).sum().item()

        avg_loss = total_loss / total_tokens.item() if total_tokens.item() > 0 else 0
        avg_acc = total_acc / total_tokens.item() if total_tokens.item() > 0 else 0

        self.model.train()
        return {"main_loss": avg_loss, "main_acc": avg_acc, "type": "eval", "step": self.global_step}
