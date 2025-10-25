import itertools
import random
from typing import Any

import torch
from rich.progress import Progress
from torch.utils.data import DataLoader, Subset

from .config import TrainConfig
from .data import GridDeserializer, GridSerializer
from .model import ArcTransformer
from .observer import Observer


class EvaluationStep:
    def __init__(self, model: ArcTransformer, serializer: GridSerializer, deserializer: GridDeserializer, observer: Observer, device: torch.device, train_dataset: Any, config: TrainConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.serializer = serializer
        self.deserializer = deserializer
        self.observer = observer
        self.device = device
        self.config = config

    def _run_eval_loop(self, loader: DataLoader, num_samples: int, title: str, global_step: int) -> tuple[int, int]:
        total_correct, evaluated_count = 0, 0
        visualized_this_loop = False

        with Progress(console=self.observer.console, transient=True) as progress:
            task = progress.add_task(f"[cyan]{title}...", total=num_samples)
            for item in itertools.islice(loader, num_samples):
                task_data = item if isinstance(item, dict) else item[0]

                if 'output' not in task_data["test"][0]:
                    progress.update(task, advance=1)
                    continue

                target_grid_raw = torch.tensor(task_data['test'][0]['output'], device=self.device)
                pred_grid, generated_tokens, probabilities = self.model.generate(self.serializer, self.deserializer, task_data)

                is_correct = 1 if torch.equal(pred_grid.to(self.device), target_grid_raw) else 0

                if not visualized_this_loop:
                    input_grid_raw = torch.tensor(task_data['test'][0]['input'])
                    decoded_tokens = self.serializer.tokenizer.decode(generated_tokens)
                    self.observer.visualize_evaluation_sample(input_grid_raw, target_grid_raw, pred_grid, decoded_tokens, probabilities, global_step)
                    visualized_this_loop = True

                total_correct += is_correct
                evaluated_count += 1
                progress.update(task, advance=1)

        return total_correct, evaluated_count

    def run(self, eval_loader: Any, total_tasks_processed: int, global_step: int, curriculum_stage: int, advance_curriculum_fn: callable, verbose: bool = False) -> dict[str, float]:
        self.model.eval()
        self.observer.console.print(f"\n[bold cyan]--- Running 3-Phase Evaluation @ Step {global_step} (Stage {curriculum_stage}) ---[/bold cyan]")

        num_historical = total_tasks_processed
        if num_historical == 0:
            self.observer.console.print("[yellow]Phase 1 skipped (no historical tasks to check for forgetting).[/yellow]")
        else:
            num_to_sample = min(5, num_historical)
            indices = random.sample(range(num_historical), k=num_to_sample)
            historical_subset = Subset(self.train_loader.dataset, indices)
            historical_loader = DataLoader(historical_subset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
            correct, total = self._run_eval_loop(historical_loader, num_to_sample, "Phase 1: Forgetting Check", global_step)
            if correct < total:
                self.observer.console.print(f"[bold red]Phase 1 FAILED: Forgetting detected! ({correct}/{total} passed). Aborting further evaluation.[/bold red]")
                self.model.train()
                return {"eval_grid_acc": correct / total if total > 0 else 0}
            self.observer.console.print(f"[bold green]Phase 1 PASSED: No forgetting detected ({correct}/{total}).[/bold green]")

            if curriculum_stage == 1:
                advance_curriculum_fn()
                self.model.train()
                return {"eval_grid_acc": 1.0}

        num_to_sample_quick = min(5, len(eval_loader.dataset))
        correct_quick, total_quick = self._run_eval_loop(eval_loader, num_to_sample_quick, "Phase 2: Quick Generalization", global_step)
        if correct_quick < total_quick:
            self.observer.console.print(f"[bold red]Phase 2 FAILED: Quick generalization check failed! ({correct_quick}/{total_quick} passed). Aborting full evaluation.[/bold red]")
            self.model.train()
            return {"eval_grid_acc": correct_quick / total_quick if total_quick > 0 else 0}
        self.observer.console.print(f"[bold green]Phase 2 PASSED: Quick generalization OK ({correct_quick}/{total_quick}). Proceeding to full evaluation.[/bold green]")

        num_full = len(eval_loader.dataset)
        correct_full, total_full = self._run_eval_loop(eval_loader, num_full, "Phase 3: Full Evaluation", global_step)

        final_acc = correct_full / total_full if total_full > 0 else 0
        metrics = {"eval_grid_acc": final_acc, "total_count": float(total_full)}
        self.observer.log_eval_summary(metrics, global_step)

        self.model.train()
        return metrics
