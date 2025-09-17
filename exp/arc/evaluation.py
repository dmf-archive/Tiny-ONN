import itertools
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from rich.progress import Progress
from torch.utils.data import DataLoader, Subset

from .config import GenerationConfig
from .consistency import ConsistencyTools
from .data import GridDeserializer, GridSerializer
from .model import ArcTransformer
from .observer import Observer


class SimpleEvaluator:
    """A minimal, hardcoded evaluator for ARC tasks. No beams, no candidates, just inference."""

    def __init__(self, model: ArcTransformer, serializer: GridSerializer, deserializer: GridDeserializer, observer: Observer, device: torch.device):
        self.model = model
        self.serializer = serializer
        self.deserializer = deserializer
        self.observer = observer
        self.device = device
        self.consistency_tools = ConsistencyTools()

    @torch.no_grad()
    def evaluate_single(self, mini_task: dict[str, Any], gate_temperature: float) -> tuple[torch.Tensor, list[int]]:
        """Evaluate a single mini-task (input-output pair) and return the predicted output grid."""
        input_grid = torch.tensor(mini_task['input'], device=self.device)
        h_in, w_in = input_grid.shape
        max_new_tokens = int(h_in * w_in * 9) + 50 # Heuristic for max tokens

        task_data_for_serializer = {'test': [mini_task]}
        prompt_ids = self.serializer.serialize_for_inference(task_data_for_serializer)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # Greedy generation is sufficient for evaluation
        generated_tokens = self._greedy_generate(prompt_tensor, max_new_tokens, gate_temperature)
        pred_grid = self.deserializer.deserialize(generated_tokens)

        return pred_grid, generated_tokens

    def _greedy_generate(self, input_ids: torch.Tensor, max_new_tokens: int, gate_temperature: float) -> list[int]:
        """Performs greedy generation without any logits warping or beam search."""
        tokens = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            model_input = tokens if past_key_values is None else tokens[:, -1:]
            logits, _, _, _, _, _, _, past_key_values = self.model(model_input, past_key_values=past_key_values, gate_temperature=gate_temperature)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            tokens = torch.cat([tokens, next_token], dim=-1)

            # Stop on EOS token
            if next_token.item() == self.serializer.tokenizer.eos_token_id:
                break

        prompt_len = input_ids.shape[1]
        return tokens[0, prompt_len:].tolist()


class EvaluationStep:
    """Orchestrates the 3-phase evaluation protocol."""

    def __init__(self, model: ArcTransformer, serializer: GridSerializer, deserializer: GridDeserializer, observer: Observer, device: torch.device, train_dataset: Any):
        self.model = model
        self.train_dataset = train_dataset
        self.serializer = serializer
        self.deserializer = deserializer
        self.observer = observer
        self.device = device
        self.evaluator = SimpleEvaluator(self.model, self.serializer, self.deserializer, self.observer, self.device)

    def _run_eval_loop(self, loader: DataLoader, num_samples: int, title: str, global_step: int) -> tuple[int, int]:
        """Core evaluation loop: run inference on a set of samples and count correct predictions."""
        total_correct, evaluated_count = 0, 0
        visualized_this_loop = False

        with Progress(console=self.observer.console, transient=True) as progress:
            task = progress.add_task(f"[cyan]{title}...", total=num_samples)
            for item in itertools.islice(loader, num_samples):
                # In the new data format, each item is a single mini-task (input-output pair)
                mini_task = item if isinstance(item, dict) else item[0]
                target_grid_raw = torch.tensor(mini_task['output'], device=self.device)

                # Use the same gate_temperature as training for consistency
                pred_grid, generated_tokens = self.evaluator.evaluate_single(mini_task, gate_temperature=self.observer.config.gate_temperature)

                is_correct = 0
                if torch.equal(pred_grid.to(self.device), target_grid_raw):
                    is_correct = 1

                if not visualized_this_loop:
                    input_grid_raw = torch.tensor(mini_task['input'])
                    self.observer.visualize_evaluation_sample(input_grid_raw, target_grid_raw, pred_grid, generated_tokens, global_step)
                    visualized_this_loop = True

                total_correct += is_correct
                evaluated_count += 1
                progress.update(task, advance=1)

        return total_correct, evaluated_count

    def run(self, eval_loader: Any, current_task_idx: int, global_step: int, verbose: bool = False) -> dict[str, float]:
        """Runs the full 3-phase evaluation protocol."""
        self.model.eval()
        self.observer.console.print(f"\n[bold cyan]--- Running 3-Phase Evaluation @ Step {global_step} ---[/bold cyan]")

        # Phase 1: Historical Task Sampling (Forgetting Check)
        num_historical = current_task_idx
        if num_historical == 0:
            self.observer.console.print("[yellow]Phase 1 skipped (no historical tasks to check for forgetting).[/yellow]")
        else:
            num_to_sample = min(5, num_historical)
            indices = random.sample(range(num_historical), k=num_to_sample)
            historical_subset = Subset(self.train_dataset, indices)
            historical_loader = DataLoader(historical_subset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
            correct, total = self._run_eval_loop(historical_loader, num_to_sample, "Phase 1: Forgetting Check", global_step)
            if correct < total:
                self.observer.console.print(f"[bold red]Phase 1 FAILED: Forgetting detected! ({correct}/{total} passed). Aborting further evaluation.[/bold red]")
                self.model.train()
                return {"eval_grid_acc": correct / total if total > 0 else 0}
            self.observer.console.print(f"[bold green]Phase 1 PASSED: No forgetting detected ({correct}/{total}).[/bold green]")
        
        # Phase 2: Quick Eval on Evaluation Set (Generalization Check 1)
        num_to_sample_quick = min(5, len(eval_loader.dataset))
        correct_quick, total_quick = self._run_eval_loop(eval_loader, num_to_sample_quick, "Phase 2: Quick Generalization", global_step)
        if correct_quick < total_quick:
            self.observer.console.print(f"[bold red]Phase 2 FAILED: Quick generalization check failed! ({correct_quick}/{total_quick} passed). Aborting full evaluation.[/bold red]")
            self.model.train()
            return {"eval_grid_acc": correct_quick / total_quick if total_quick > 0 else 0}
        self.observer.console.print(f"[bold green]Phase 2 PASSED: Quick generalization OK ({correct_quick}/{total_quick}). Proceeding to full evaluation.[/bold green]")

        # Phase 3: Full Evaluation (Final Generalization Check)
        num_full = len(eval_loader.dataset)
        correct_full, total_full = self._run_eval_loop(eval_loader, num_full, "Phase 3: Full Evaluation", global_step)
        
        final_acc = correct_full / total_full if total_full > 0 else 0
        metrics = {"eval_grid_acc": final_acc, "total_count": float(total_full)}
        self.observer.log_eval_summary(metrics, global_step)

        self.model.train()
        return metrics
