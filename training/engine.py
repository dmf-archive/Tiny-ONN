import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    get_scheduler,
)

from .callbacks import MatplotlibVisualizer, TensorBoardLogger
from .config import TrainConfig
from .hooks import GradientInterceptor


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
        self.model = model
        self.optimizer_experts = optimizer_experts
        self.optimizer_router = optimizer_router
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.grad_scaler = GradScaler()
        self.interceptor = GradientInterceptor(model)
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
        self.visualizer = MatplotlibVisualizer(self.img_dir, config.model.base_model_name)

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

    def _hyper_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.optimizer_experts.zero_grad()
        self.optimizer_router.zero_grad()

        inputs = {k: v.to(self.device) for k, v in batch.items() if v is not None}

        with autocast():
            student_outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_router_logits=True,
            )
            if self.teacher_model:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**inputs)

                distillation_loss = F.kl_div(
                    F.log_softmax(
                        student_outputs.logits / self.config.training.distillation_temperature,
                        dim=-1,
                    ),
                    F.softmax(
                        teacher_outputs.logits / self.config.training.distillation_temperature,
                        dim=-1,
                    ),
                    reduction="batchmean",
                )
                main_loss = (
                    self.config.training.distillation_alpha * distillation_loss
                    + (1 - self.config.training.distillation_alpha) * student_outputs.loss
                )
            else:
                main_loss = student_outputs.loss

        self.grad_scaler.scale(main_loss).backward(retain_graph=True)
        self.grad_scaler.step(self.optimizer_experts)
        self.lr_scheduler_experts.step()

        surprises_by_expert = self.interceptor.get_surprises()

        router_loss = torch.tensor(0.0, device=self.device)
        gating_acc = 0.0
        avg_surprise = 0.0

        if surprises_by_expert and student_outputs.router_logits:
            all_router_logits = torch.cat(
                [l.float() for l in student_outputs.router_logits], dim=0
            )
            all_selected_experts = torch.cat(
                [l for l in student_outputs.selected_experts], dim=0
            )

            num_tokens, num_experts = all_router_logits.shape

            per_token_surprise = torch.full(
                (num_tokens, num_experts), float("inf"), device=self.device
            )

            for expert_module, surprise_tensor in surprises_by_expert.items():
                expert_idx = self.interceptor.expert_to_id[expert_module]
                token_indices = surprise_tensor[:, 0].long()
                surprise_values = surprise_tensor[:, 1]
                per_token_surprise[token_indices, expert_idx] = surprise_values

            optimal_expert_indices = torch.argmin(per_token_surprise, dim=1)

            valid_surprise_mask = ~torch.isinf(per_token_surprise).all(dim=1)

            if valid_surprise_mask.any():
                router_loss = F.cross_entropy(
                    all_router_logits[valid_surprise_mask],
                    optimal_expert_indices[valid_surprise_mask],
                )

                gating_acc = (
                    (
                        all_selected_experts[valid_surprise_mask].squeeze(-1)
                        == optimal_expert_indices[valid_surprise_mask]
                    )
                    .float()
                    .mean()
                    .item()
                )

                finite_surprises = per_token_surprise[valid_surprise_mask]
                if finite_surprises.numel() > 0:
                    avg_surprise = (
                        finite_surprises[~torch.isinf(finite_surprises)].mean().item()
                    )

                self.expert_data_cache["selected_experts"].append(
                    np.vstack(
                        [
                            np.full(all_selected_experts.shape[0], self.global_step),
                            all_selected_experts.cpu().numpy().flatten(),
                        ]
                    ).T
                )
                self.expert_data_cache["optimal_experts"].append(
                    np.vstack(
                        [
                            np.full(optimal_expert_indices.shape[0], self.global_step),
                            optimal_expert_indices.cpu().numpy().flatten(),
                        ]
                    ).T
                )

        self.grad_scaler.scale(router_loss).backward()
        self.grad_scaler.step(self.optimizer_router)
        self.lr_scheduler_router.step()

        self.grad_scaler.update()
        self.interceptor.clear()

        with torch.no_grad():
            labels = inputs.get("labels")
            main_acc = (
                (student_outputs.logits.argmax(-1) == labels).float().mean().item()
                if labels is not None
                else 0.0
            )
            tau = (
                torch.distributions.Categorical(logits=student_outputs.logits.float())
                .entropy()
                .mean()
                .item()
            )
            pi_score = torch.exp(
                -1.0 * (main_loss.item() / tau + 0.1 * avg_surprise)
            ).item()

        metrics: dict[str, float] = {
            "main_loss": main_loss.item(),
            "router_loss": router_loss.item()
            if isinstance(router_loss, torch.Tensor)
            else router_loss,
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
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"

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

        # Rolling checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob("*.pt"), key=os.path.getmtime)
        if len(checkpoints) > self.config.logging.rolling_checkpoint_count:
            os.remove(checkpoints[0])

    def _load_checkpoint(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_experts.load_state_dict(
            checkpoint["optimizer_experts_state_dict"]
        )
        self.optimizer_router.load_state_dict(checkpoint["optimizer_router_state_dict"])
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

    def train(self):
        for epoch in range(self.start_epoch, self.config.training.num_train_epochs):
            self.model.train()
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            for batch in progress_bar:
                metrics = self._hyper_step(batch)
                progress_bar.set_postfix(metrics)

                if self.global_step % self.config.logging.log_interval == 0:
                    self.logger.log_metrics(metrics, self.global_step)

                if self.global_step % self.config.logging.eval_interval == 0:
                    self.visualizer.update_plots(
                        self.metrics_cache, self.expert_data_cache, self.global_step, self.model
                    )

                if self.global_step % self.config.logging.checkpoint_interval == 0:
                    self._save_checkpoint(epoch)

                self.global_step += 1

        self.logger.close()

    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(self.eval_dataloader)
        return {
            "eval_loss": avg_loss,
            "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
        }
