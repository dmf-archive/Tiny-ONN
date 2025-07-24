import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, cast

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
        self.grad_scaler = None
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

        self.surprise_ema = {"mean": 0.0, "var": 1.0, "count": 0}
        self.cost_ema = {"mean": 0.0, "var": 1.0, "count": 0}

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

    def _hyper_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        inputs = {k: v.to(self.device) for k, v in batch.items() if v is not None}

        self.optimizer_experts.zero_grad()
        self.optimizer_router.zero_grad()

        surprise_context: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        forward_kwargs = {"surprise_context": surprise_context}
        if self.model.config.num_experts_per_tok == -1:
            if self.last_main_loss is None:
                # On the first step, use a default budget
                surprise_budget = 0.5
            else:
                # Dynamically adjust budget based on last step's loss
                # Clamp the loss to a reasonable range before sigmoid to ensure sensitivity
                clamped_loss = torch.clamp(torch.tensor(self.last_main_loss), 0.1, 5.0)
                surprise_budget = torch.sigmoid(clamped_loss).item()
            forward_kwargs["surprise_budget"] = surprise_budget

        inputs.update(forward_kwargs) # type: ignore

        student_outputs: CausalLMOutputWithPast = self.model(**inputs)
        main_loss = student_outputs.loss
        assert main_loss is not None

        main_loss.backward(retain_graph=True)

        # --- Surprise Processing and Router Loss Calculation ---
        all_router_logits: List[torch.Tensor] = []
        all_selected_experts: List[Tuple[int, torch.Tensor]] = []
        all_k_per_token: List[torch.Tensor] = []
        layer_token_offsets: Dict[int, int] = {}
        
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
        per_token_surprise = torch.tensor([], device=self.device) # Initialize empty tensor

        if all_router_logits:
            concatenated_logits = torch.cat(all_router_logits, dim=0)
            per_token_surprise = torch.full_like(concatenated_logits, float("inf"))

            for (layer_idx, expert_idx), (token_indices, surprise) in surprise_context.items():
                if layer_idx in layer_token_offsets:
                    row_offset = layer_token_offsets[layer_idx]
                    per_token_surprise[row_offset + token_indices, expert_idx] = surprise.to(per_token_surprise.dtype)

            optimal_indices = torch.argmin(per_token_surprise, dim=1)
            router_loss = F.cross_entropy(concatenated_logits, optimal_indices)
            
            with torch.no_grad():
                predicted_indices = torch.argmin(concatenated_logits, dim=1)
                gating_acc = (predicted_indices == optimal_indices).float().mean().item()
                
                num_experts_per_layer = self.model.config.num_experts_per_layer
                
                selected_experts_to_plot = []
                for layer_idx, experts_tensor in all_selected_experts:
                    global_expert_ids = experts_tensor + layer_idx * num_experts_per_layer
                    selected_experts_to_plot.append(global_expert_ids.reshape(-1, 1))
                if selected_experts_to_plot:
                    selected_experts_np = torch.cat(selected_experts_to_plot, dim=0).cpu().numpy()
                    step_col = np.full((selected_experts_np.shape[0], 1), self.global_step)
                    self.expert_data_cache["selected_experts"].append(np.hstack([step_col, selected_experts_np]))

                token_to_layer_map = []
                for i, layer in enumerate(self.model.model.layers):
                    moe_layer = cast(TinyOnnMoE, layer.mlp)
                    if moe_layer.last_router_logits is not None:
                        num_layer_tokens = moe_layer.last_router_logits.shape[0]
                        token_to_layer_map.extend([i] * num_layer_tokens)
                
                global_optimal_experts = []
                flat_optimal_indices = optimal_indices.cpu().numpy()
                for i, local_expert_idx in enumerate(flat_optimal_indices):
                    layer_idx = token_to_layer_map[i]
                    global_expert_idx = local_expert_idx + layer_idx * num_experts_per_layer
                    global_optimal_experts.append(global_expert_idx)
                
                optimal_experts_np = np.array(global_optimal_experts).reshape(-1, 1)
                step_col_optimal = np.full((optimal_experts_np.shape[0], 1), self.global_step)
                self.expert_data_cache["optimal_experts"].append(np.hstack([step_col_optimal, optimal_experts_np]))

        if router_loss.requires_grad:
            router_loss.backward()
        
        self.optimizer_experts.step()
        self.optimizer_router.step()
        
        self.lr_scheduler_experts.step()
        self.lr_scheduler_router.step()

        with torch.no_grad():
            labels = inputs.get("labels")
            main_acc = 0.0
            tau = 1.0
            if student_outputs.logits is not None and labels is not None:
                main_acc = (student_outputs.logits.argmax(-1) == labels).float().mean().item()
                tau = torch.distributions.Categorical(logits=student_outputs.logits.float()).entropy().mean().item()

            if per_token_surprise.numel() > 0:
                valid_surprise = per_token_surprise[torch.isfinite(per_token_surprise)]
            else:
                valid_surprise = torch.tensor([], device=self.device)
            surprise_val = valid_surprise.mean().item() if valid_surprise.numel() > 0 else 0.0
            # Update EMAs for surprise and cost
            def update_ema(ema_dict, value, beta=0.99):
                if ema_dict["count"] == 0:
                    ema_dict["mean"] = value
                    ema_dict["var"] = 0
                else:
                    ema_dict["mean"] = beta * ema_dict["mean"] + (1 - beta) * value
                    ema_dict["var"] = beta * ema_dict["var"] + (1 - beta) * (value - ema_dict["mean"])**2
                ema_dict["count"] += 1
    
            cost = main_loss.item() / tau
            update_ema(self.surprise_ema, surprise_val)
            update_ema(self.cost_ema, cost)
    
            # Calculate Z-scores
            surprise_z = (surprise_val - self.surprise_ema["mean"]) / (self.surprise_ema["var"]**0.5 + 1e-8)
            cost_z = (cost - self.cost_ema["mean"]) / (self.cost_ema["var"]**0.5 + 1e-8)
            
            pi_score_tensor_input = torch.tensor(-self.config.training.pi_alpha * (cost_z + self.config.training.pi_gamma * surprise_z), device=self.device)
            pi_score = torch.exp(pi_score_tensor_input).item()

        if all_k_per_token:
            avg_k = np.mean([k.float().mean().item() for k in all_k_per_token])
        else:
            avg_k = 0.0
        
        metrics: Dict[str, float] = {
            "main_loss": main_loss.item(),
            "router_loss": router_loss.item(),
            "main_acc": main_acc,
            "gating_acc": gating_acc,
            "surprise": surprise_val,
            "tau": float(tau),
            "pi_score": pi_score,
            "avg_k": avg_k,
        }
        for k, v in metrics.items():
            self.metrics_cache[k].append(v)
            self.last_main_loss = main_loss.item()
            return metrics

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
        for epoch in range(self.start_epoch, self.config.training.num_train_epochs):
            self.model.train()
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            for batch in progress_bar:
                if max_steps is not None and self.global_step >= max_steps:
                    self.logger.close()
                    return
                metrics = self._hyper_step(batch)
                progress_bar.set_postfix(metrics)
                if self.global_step % self.config.logging.log_interval == 0:
                    self.logger.log_metrics(metrics, self.global_step)
                
                # Always update plots
                expert_data_to_plot = {
                    k: np.vstack(v) for k, v in self.expert_data_cache.items() if v
                }
                if expert_data_to_plot: # Only plot if there is data
                    self.visualizer.update_plots(
                        self.metrics_cache, expert_data_to_plot, self.global_step, self.model
                    )
                if self.global_step % self.config.logging.checkpoint_interval == 0:
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
        return {"eval_loss": avg_loss, "perplexity": torch.exp(torch.tensor(avg_loss)).item()}
