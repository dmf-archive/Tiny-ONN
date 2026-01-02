import os
import random
import shutil
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from transformers import logging as transformers_logging

from .config import TrainConfig
from .data import GridDeserializer, GridSerializer, InMemoryArcDataset
from .evaluation import EvaluationStep
from .model import ArcTransformer
from .observer import Observer
from .rmsuon import RMSuon
from .tokenizer import ArcColorTokenizerFast

# 过滤Fast tokenizer的警告，因为我们的使用场景是合理的
warnings.filterwarnings("ignore", message=".*ArcColorTokenizerFast tokenizer.*")
transformers_logging.set_verbosity_error()


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


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config, self.device = config, torch.device(config.device)
        torch.manual_seed(config.seed)
        self.console = Console()
        self.observer = Observer(self.console, config)
        self.tokenizer = ArcColorTokenizerFast()
        self.serializer = GridSerializer(self.tokenizer)
        self._setup_data()
        self._setup_model_and_optimizer()
        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.global_step, self.epoch, self.start_task_idx = 0, 0, 0
        self.total_tasks_processed = 0

        self.consecutive_high_pi_steps = 0
        self.early_stop_triggered = False

    def _setup_data(self):
        self.train_dataset = InMemoryArcDataset(
            data_path=self.config.data.data_path,
            tokenizer=self.tokenizer,
            split="training",
            max_len=self.config.model.max_position_embeddings,
        )
        self.eval_dataset = InMemoryArcDataset(
            data_path=self.config.data.data_path,
            tokenizer=self.tokenizer,
            split="evaluation",
            max_len=self.config.model.max_position_embeddings,
        )

        self.console.print(f"[bold green]Training dataset size: {len(self.train_dataset)}[/bold green]")
        self.console.print(f"[bold green]Evaluation dataset size: {len(self.eval_dataset)}[/bold green]")
        
        # 调试断言：确保数据集不为空且预处理逻辑正确
        assert len(self.train_dataset) > 0, "Training dataset is empty! Check data path or max_len filtering."
        # ARC-AGI-2 训练集有 1000 个样本，过滤后不应太少
        if len(self.train_dataset) < 100:
            self.console.print(f"[bold yellow]Warning: Training dataset size ({len(self.train_dataset)}) is suspiciously small.[/bold yellow]")

    def _setup_model_and_optimizer(self):
        self.config.model.vocab_size = self.tokenizer.vocab_size
        self.model = ArcTransformer(self.config.model, device=self.device).to(self.device)

        self.optimizer = RMSuon(
            self.model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
        )
        self.evaluator = EvaluationStep(self.model, self.serializer, GridDeserializer(self.tokenizer), self.observer, self.device, self.train_dataset, self.config)

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

        ids, labels, _, diff_mask = self.serializer.serialize_task(augmented_task)
        if len(ids) > max_len:
            return None

        valid_labels = [l for l in labels if l != -100]
        if not valid_labels:
            entropy = 0.0
        else:
            counts = torch.bincount(torch.tensor(valid_labels))
            probs = counts.float() / len(valid_labels)
            probs = probs[probs > 0]
            entropy = -torch.sum(probs * torch.log2(probs)).item()

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_entropy": torch.tensor(entropy, dtype=torch.float32),
            "diff_mask": torch.tensor(diff_mask, dtype=torch.bool),
        }

    def _train_step(self, batch: dict, epoch: int, task_idx: int | str) -> tuple | None:
        start_time = time.time()
        self.model.train()


        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            return_dict=True,
        )
        self.optimizer.zero_grad()
        main_loss = model_outputs.loss

        # --- START: Adaptive Auxiliary Loss Calculation ---
        logits = model_outputs.logits
        labels = batch["labels"]
        diff_mask = batch["diff_mask"]

        # Align logits and labels for loss calculation
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        shifted_diff_mask = diff_mask[..., 1:].contiguous()

        # Calculate diff loss only on differing pixels
        diff_labels = torch.where(shifted_diff_mask, shifted_labels, -100)
        
        loss_fct = nn.CrossEntropyLoss()
        diff_loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), diff_labels.view(-1))
        
        # If diff_loss is NaN (no differing pixels), set it to 0
        if torch.isnan(diff_loss):
            diff_loss = torch.tensor(0.0, device=main_loss.device)

        # Calculate adaptive weight
        num_diff_tokens = shifted_diff_mask.sum()
        num_total_label_tokens = (shifted_labels != -100).sum()
        
        num_identity_tokens = num_total_label_tokens - num_diff_tokens

        # Set a high penalty if there are no different tokens to encourage finding them
        if num_diff_tokens == 0:
            lambda_adaptive = torch.tensor(0.0, device=main_loss.device) # Set to 0 if no diff, no aux loss needed
        else:
            lambda_adaptive = (num_identity_tokens / num_diff_tokens).clamp(max=100.0)

        total_loss = main_loss + lambda_adaptive * diff_loss
        # --- END: Adaptive Auxiliary Loss Calculation ---

        if not torch.isfinite(total_loss):
            self.console.print(f"[bold red]NaN detected in total_loss at step {self.global_step}. Aborting step.[/bold red]")
            return None

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        model_outputs.labels = batch["labels"]
        model_outputs.sample_entropy = batch["sample_entropy"]

        signals = {"grad_norm": grad_norm.item(), "lambda_adaptive": lambda_adaptive.item()}
        metrics = self.observer.calculate_metrics(total_loss, model_outputs, signals, batch["input_ids"], self.model)

        self.observer.maybe_log_and_visualize(
            epoch,
            self.global_step,
            task_idx if isinstance(task_idx, int) else -1,
            metrics,
            time.time() - start_time,
            signals,
            self.evaluator,
            self.eval_dataset,
            self.total_tasks_processed,
            self._save_checkpoint,
        )

        self.global_step += 1

        if metrics.get('pi', 0.0) >= self.config.pi_early_stop_threshold:
            self.console.print(f"[bold green]Task PI={metrics['pi']:.3f} >= {self.config.pi_early_stop_threshold}, skipping remaining views[/bold green]")
            return None, signals

        return metrics, signals



    def _train_epoch(self, epoch: int):
        num_tasks_in_stage = len(self.train_dataset)

        # 收集所有样本并按长度排序（贪心背包算法）
        all_samples = []
        for task_idx in range(self.start_task_idx, num_tasks_in_stage):
            if self.early_stop_triggered:
                self.console.print(f"[bold green]Task loop early stopped at task {task_idx} due to global PI threshold[/bold green]")
                break
            task_data = self.train_dataset[task_idx]
            selected_views = random.sample(range(8), self.config.num_augmentation_views)
            
            for view_idx in selected_views:
                batch_cpu = self._prepare_batch(task_data, view_idx, self.config.model.max_position_embeddings)
                if not batch_cpu:
                    continue
                
                sample_tokens = len(batch_cpu["input_ids"])
                all_samples.append({
                    "batch": batch_cpu,
                    "task_idx": task_idx,
                    "tokens": sample_tokens
                })
        
        # 按长度降序排序（首次适应递减贪心算法）
        all_samples.sort(key=lambda x: x["tokens"], reverse=True)
        
        # 使用贪心背包算法打包
        max_tokens_per_batch = min(self.config.data.max_tokens_per_batch, 8192)
        current_batch_tasks = []
        current_batch_task_indices = []
        current_total_tokens = 0
        
        for sample in all_samples:
            sample_tokens = sample["tokens"]
            
            # 如果当前样本无法加入当前批次，处理当前批次并开始新批次
            if current_total_tokens + sample_tokens > max_tokens_per_batch and current_batch_tasks:
                self._process_dynamic_batch(current_batch_tasks, current_batch_task_indices, epoch)
                current_batch_tasks = []
                current_batch_task_indices = []
                current_total_tokens = 0
            
            # 尝试将样本加入当前批次
            if sample_tokens <= max_tokens_per_batch:
                current_batch_tasks.append(sample["batch"])
                current_batch_task_indices.append(sample["task_idx"])
                current_total_tokens += sample_tokens
        
        # 处理最后一个批次
        if current_batch_tasks:
            self._process_dynamic_batch(current_batch_tasks, current_batch_task_indices, epoch)

        self.start_task_idx = 0

    def _process_dynamic_batch(self, batch_tasks: list, batch_task_indices: list, epoch: int):
        """处理动态batch的训练"""
        if not batch_tasks:
            return

        # 使用新的Tokenizer进行批处理，自动处理padding和attention_mask
        raw_input_ids = [task["input_ids"] for task in batch_tasks]

        # 我们需要手动处理标签的填充，因为它们包含-100
        raw_labels = [task["labels"] for task in batch_tasks]
        padded_labels = nn.utils.rnn.pad_sequence(raw_labels, batch_first=True, padding_value=-100)

        raw_diff_masks = [task["diff_mask"] for task in batch_tasks]
        padded_diff_masks = nn.utils.rnn.pad_sequence(raw_diff_masks, batch_first=True, padding_value=False)

        # 由于raw_input_ids已经是token IDs，不能直接使用tokenizer.__call__方法
        # 使用tokenizer.pad方法来处理预编码的token IDs
        batch = self.tokenizer.pad(
            {"input_ids": raw_input_ids},
            padding=True,
            return_tensors="pt",
        )
        # 手动构造 attention_mask，避免模型 forward 时 KeyError
        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()

        batch["labels"] = padded_labels
        batch["diff_mask"] = padded_diff_masks
        batch["sample_entropy"] = torch.stack([task["sample_entropy"] for task in batch_tasks])

        # 将所有张量移动到正确的设备
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # 使用第一个任务索引作为标识
        first_task_idx = batch_task_indices[0] if batch_task_indices else 0


        result = self._train_step(batch, epoch, first_task_idx)
        torch.cuda.empty_cache()

        if result is not None:
            metrics, _ = result
            if metrics and metrics.get('pi', 0.0) >= self.config.pi_early_stop_threshold:
                self.consecutive_high_pi_steps += 1
                if self.consecutive_high_pi_steps >= self.config.global_early_stop_steps:
                    self.console.print(f"[bold green]Global early stop triggered: {self.consecutive_high_pi_steps} consecutive steps with PI >= {self.config.pi_early_stop_threshold}[/bold green]")
                    self.early_stop_triggered = True
                    return
            else:
                self.consecutive_high_pi_steps = 0

        self.total_tasks_processed += len(set(batch_task_indices))

    def train(self):
        self._load_checkpoint()
        self.console.print("[bold green]Starting Training...[/bold green]")
        for epoch in range(self.epoch, self.config.num_epochs):
            if self.early_stop_triggered:
                self.console.print(f"[bold green]Training early stopped at epoch {epoch} due to PI threshold[/bold green]")
                break
            self.epoch = epoch
            self._train_epoch(epoch)
        self.console.print("[bold green]Training Finished.[/bold green]")

    def _save_checkpoint(self, task_idx: int):
        checkpoint_dir = self.checkpoint_dir / f"checkpoint_{self.global_step}"
        checkpoint_dir.mkdir(exist_ok=True)

        model_path = checkpoint_dir / "model.safetensors"
        save_file(self.model.state_dict(), model_path)

        trainer_state = {
            "epoch": self.epoch,
            "step": self.global_step,
            "task_idx": self.total_tasks_processed,
            "total_tasks_processed": self.total_tasks_processed,
            "optimizer_state_dict": self.optimizer.state_dict(),
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
                    self.optimizer.load_state_dict(state["optimizer_state_dict"])
                    self.global_step, self.epoch, self.start_task_idx = (
                        state["step"],
                        state["epoch"],
                        state["task_idx"],
                    )
                    self.total_tasks_processed = state.get(
                        "total_tasks_processed", self.start_task_idx
                    )

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
