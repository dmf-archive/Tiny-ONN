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
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import ArcCollator, GridDeserializer, GridSerializer, InMemoryArcDataset
from .evaluation import EvaluationStep
from .model import ArcTransformer
from .observer import Observer
from .tokenizer import ArcColorTokenizer
from .F3EO import F3EO


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

    def _train_step(self, batch: dict, epoch: int, task_idx: int | str, view_idx: int) -> tuple | None:
        start_time = time.time()
        self.model.train()

        with torch.autocast(
            device_type=self.config.device, dtype=torch.bfloat16
        ):
            model_outputs = self.model(
                batch["input_ids"],
                coords=None,
                return_dict=True,
            )
        self.optimizer.zero_grad()
        main_loss = F.cross_entropy(
            model_outputs["logits"][:, :-1, :].contiguous().view(-1, self.config.model.vocab_size),
            batch["labels"][:, 1:].contiguous().view(-1),
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
        return metrics, signals


    def advance_curriculum(self):
        if self.curriculum_stage == 1:
            self.curriculum_stage = 2
            self.train_loader.dataset.set_stage(2)
            self.console.print("[bold magenta]Curriculum stage advanced to 2. Using full dataset.[/bold magenta]")
            self.start_task_idx = 0
            self.epoch = 0

    def _train_epoch(self, epoch: int):
        dataset = self.train_loader.dataset
        num_tasks_in_stage = len(dataset)
        
        task_batch_size = self.config.task_batch_size
        
        for i in range(self.start_task_idx, num_tasks_in_stage, task_batch_size):
            task_batch_indices = range(i, min(i + task_batch_size, num_tasks_in_stage))
            
            mega_batch_inputs = []
            mega_batch_labels = []
            mega_batch_entropies = []

            for task_idx in task_batch_indices:
                task_data = dataset[task_idx]
                selected_views = random.sample(range(8), self.config.num_augmentation_views)

                for view_idx in selected_views:
                    batch_cpu = self._prepare_batch(task_data, view_idx, self.config.model.max_position_embeddings)
                    if not batch_cpu:
                        self.console.print(
                            f"[yellow]Skipping Task {task_idx} View {view_idx} due to excessive length.[/yellow]"
                        )
                        continue
                    
                    mega_batch_inputs.append(batch_cpu["input_ids"])
                    mega_batch_labels.append(batch_cpu["labels"])
                    mega_batch_entropies.append(batch_cpu["sample_entropy"])

            if not mega_batch_inputs:
                continue

            padded_inputs = nn.utils.rnn.pad_sequence(mega_batch_inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            padded_labels = nn.utils.rnn.pad_sequence(mega_batch_labels, batch_first=True, padding_value=-100)
            
            batch = {
                "input_ids": padded_inputs.to(self.device),
                "labels": padded_labels.to(self.device),
                "sample_entropy": torch.stack(mega_batch_entropies).to(self.device),
            }

            self._train_step(batch, epoch, i, -1)
            
            torch.cuda.empty_cache()
            self.total_tasks_processed = i + len(task_batch_indices)
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
            "epoch": self.epoch, "step": self.global_step, "task_idx": self.total_tasks_processed,
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
                    self.global_step, self.epoch, self.start_task_idx = (state["step"], state["epoch"], state["task_idx"])
                    self.total_tasks_processed = state.get("total_tasks_processed", self.start_task_idx)

                elif path.is_file():
                    ckpt = torch.load(path, map_location=self.device)
                    self.model.load_state_dict(ckpt["model_state_dict"])
                    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    self.global_step, self.epoch, self.start_task_idx = (ckpt["step"], ckpt["epoch"], ckpt["task_idx"])
                    self.total_tasks_processed = ckpt.get("total_tasks_processed", self.start_task_idx)

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
