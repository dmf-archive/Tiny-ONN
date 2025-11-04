import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import ArcCollator, GridDeserializer, GridSerializer, InMemoryArcDataset
from .evaluation import EvaluationStep
from .F3EO import F3EO
from .model import ArcTransformer
from .observer import Observer
from .tokenizer import ArcColorTokenizer


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
        self.tokenizer, self.serializer = ArcColorTokenizer(), GridSerializer(ArcColorTokenizer())
        self._setup_data()
        self._setup_model_and_optimizer()
        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.global_step, self.epoch, self.start_task_idx = 0, 0, 0
        self.total_tasks_processed = 0

        # 内存监控阈值（9GB）
        self.memory_threshold_gb = 9.0
        self.memory_threshold_bytes = self.memory_threshold_gb * 1024 * 1024 * 1024
        
        self.consecutive_high_pi_steps = 0
        self.early_stop_triggered = False

    def _check_memory_usage(self) -> bool:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()

            if allocated > self.memory_threshold_bytes:
                self.console.print(
                    f"[bold red]Memory usage exceeded threshold: {allocated / 1024**3:.2f}GB > {self.memory_threshold_gb}GB. "
                    f"Reserved: {reserved / 1024**3:.2f}GB[/bold red]"
                )
                return False
        return True

    def _setup_data(self):
        collator = ArcCollator(self.tokenizer, max_len=self.config.model.max_position_embeddings)
        train_dataset = InMemoryArcDataset(
            data_path=self.config.data.data_path,
            tokenizer=self.tokenizer,
            split="training",
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

        self.optimizer = F3EO(
            self.model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
            orthogonalize=True
        )
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
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_entropy": torch.tensor(ArcCollator._calculate_sample_entropy(labels), dtype=torch.float32),
        }

    def _train_step(self, batch: dict, epoch: int, task_idx: int | str) -> tuple | None:
        start_time = time.time()
        self.model.train()


        model_outputs = self.model(
            batch["input_ids"],
            coords=None,
            return_dict=True,
        )
        self.optimizer.zero_grad()
        logits_fp32 = model_outputs["logits"][:, :-1, :].contiguous().view(-1, self.config.model.vocab_size)
        labels_long = batch["labels"][:, 1:].contiguous().view(-1)

        main_loss = F.cross_entropy(
            logits_fp32,
            labels_long,
            ignore_index=-100,
        )

        if not torch.isfinite(main_loss):
            self.console.print(f"[bold red]NaN detected in main_loss at step {self.global_step}. Aborting step.[/bold red]")
            return None

        main_loss.backward(create_graph=True)
        self.optimizer.step()

        model_outputs.update({"labels": batch["labels"], "sample_entropy": batch["sample_entropy"]})
        signals = {"grad_norm": self.optimizer.grad_norm}
        metrics = self.observer.calculate_metrics(main_loss, model_outputs, signals, batch["input_ids"], self.model)

        self.observer.maybe_log_and_visualize(
            epoch,
            self.global_step,
            task_idx if isinstance(task_idx, int) else -1,
            metrics,
            time.time() - start_time,
            signals,
            self.evaluator,
            self.eval_loader,
            self.total_tasks_processed,
            self._save_checkpoint,
            lambda: None,  # 空的课程推进函数
            1,  # 固定的课程阶段
        )

        self.global_step += 1
        
        if metrics.get('pi', 0.0) >= self.config.pi_early_stop_threshold:
            self.console.print(f"[bold green]Task PI={metrics['pi']:.3f} >= {self.config.pi_early_stop_threshold}, skipping remaining views[/bold green]")
            return None, signals
            
        return metrics, signals



    def _train_epoch(self, epoch: int):
        dataset = self.train_loader.dataset
        num_tasks_in_stage = len(dataset)

        for task_idx in range(self.start_task_idx, num_tasks_in_stage):
            if self.early_stop_triggered:
                self.console.print(f"[bold green]Task loop early stopped at task {task_idx} due to global PI threshold[/bold green]")
                break
            task_data = dataset[task_idx]
            selected_views = random.sample(range(8), self.config.num_augmentation_views)

            # 按view级别进行batch构建，不再累积整个任务的所有view
            max_tokens_per_batch = self.config.data.max_tokens_per_batch
            current_batch_tasks = []
            current_batch_task_indices = []  # 只用于跟踪训练进度
            current_total_tokens = 0

            for view_idx in selected_views:
                batch_cpu = self._prepare_batch(task_data, view_idx, self.config.model.max_position_embeddings)
                if not batch_cpu:
                    continue

                task_tokens = len(batch_cpu["input_ids"])

                # 检查当前view是否能加入当前batch
                if current_total_tokens + task_tokens > max_tokens_per_batch and current_batch_tasks:
                    # 处理当前batch，然后重新开始
                    self._process_dynamic_batch(current_batch_tasks, current_batch_task_indices, epoch)
                    current_batch_tasks = []
                    current_batch_task_indices = []
                    current_total_tokens = 0

                current_batch_tasks.append(batch_cpu)
                current_batch_task_indices.append(task_idx)
                current_total_tokens += task_tokens

            # 处理该任务剩余的view
            if current_batch_tasks:
                self._process_dynamic_batch(current_batch_tasks, current_batch_task_indices, epoch)

        self.start_task_idx = 0

    def _process_dynamic_batch(self, batch_tasks: list, batch_task_indices: list, epoch: int):
        """处理动态batch的训练"""
        if not batch_tasks:
            return

        # 填充序列
        mega_batch_inputs = [task["input_ids"] for task in batch_tasks]
        mega_batch_labels = [task["labels"] for task in batch_tasks]
        mega_batch_entropies = [task["sample_entropy"] for task in batch_tasks]

        padded_inputs = nn.utils.rnn.pad_sequence(mega_batch_inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels = nn.utils.rnn.pad_sequence(mega_batch_labels, batch_first=True, padding_value=-100)

        # 纯fp32模式
        batch = {
            "input_ids": padded_inputs.to(self.device, dtype=torch.long),
            "labels": padded_labels.to(self.device, dtype=torch.long),
            "sample_entropy": torch.stack(mega_batch_entropies).to(self.device, dtype=torch.float32),
        }

        # 使用第一个任务索引作为标识
        first_task_idx = batch_task_indices[0] if batch_task_indices else 0

        # 内存检查
        if not self._check_memory_usage():
            self.console.print("[bold red]Training aborted due to memory constraints.[/bold red]")
            return

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

    def _convert_state_dict_to_bf16(self, state_dict: dict) -> dict:
        """将检查点中的fp32权重自动转换为bf16"""
        converted_dict = {}
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32:
                # 将fp32权重转换为bf16
                converted_dict[key] = tensor.to(torch.bfloat16)
                self.console.print(f"[yellow]Converted {key} from fp32 to bf16[/yellow]")
            else:
                converted_dict[key] = tensor
        return converted_dict

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

                elif path.is_file():
                    ckpt = torch.load(path, map_location=self.device)
                    # 自动类型转换：fp32 -> bf16
                    model_state_dict = self._convert_state_dict_to_bf16(ckpt["model_state_dict"])
                    ckpt["model_state_dict"] = model_state_dict

                    self.model.load_state_dict(ckpt["model_state_dict"])
                    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    self.global_step, self.epoch, self.start_task_idx = (
                        ckpt["step"],
                        ckpt["epoch"],
                        ckpt["task_idx"],
                    )
                    self.total_tasks_processed = ckpt.get(
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
