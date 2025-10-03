import itertools
import random
from typing import Any

import torch
from rich.progress import Progress
from torch.utils.data import DataLoader, Subset

from .config import GenerationConfig, TrainConfig
from .consistency import ConsistencyTools
from .data import GridDeserializer, GridSerializer
from .model import ArcTransformer
from .observer import Observer


class SimpleEvaluator:

    def __init__(self, model: ArcTransformer, serializer: GridSerializer, deserializer: GridDeserializer, observer: Observer, device: torch.device, generation_config: GenerationConfig):
        self.model = model
        self.serializer = serializer
        self.deserializer = deserializer
        self.observer = observer
        self.device = device
        self.consistency_tools = ConsistencyTools()
        self.generation_config = generation_config

    @torch.no_grad()
    def evaluate_single(self, task_data: dict[str, Any]) -> tuple[torch.Tensor, list[int]]:
        max_new_tokens = self.generation_config.max_new_tokens

        prompt_ids, prompt_coords = self.serializer.serialize_for_inference(task_data)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        prompt_coords_tensor = torch.tensor([prompt_coords], dtype=torch.long, device=self.device)

        generated_tokens = self._generate_greedy(prompt_tensor, prompt_coords_tensor, max_new_tokens)
        pred_grid = self.deserializer.deserialize(generated_tokens)

        return pred_grid, generated_tokens

    def _generate_greedy(self, input_ids: torch.Tensor, input_coords: torch.Tensor, max_new_tokens: int) -> list[int]:
        tokens = input_ids.clone()
        coords = input_coords.clone()
        past_key_values = None

        current_r, current_c = 0, -1

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            for _ in range(max_new_tokens):
                model_input = tokens if past_key_values is None else tokens[:, -1:]
                coords_input = coords if past_key_values is None else coords[:, -1:]
                outputs = self.model(model_input, coords=coords_input, past_key_values=past_key_values, return_dict=True)
                logits = outputs["logits"]
                past_key_values = outputs["past_key_values"]

                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_token], dim=-1)

                next_token_item = next_token.item()
                if self.serializer.tokenizer.token_id_to_color(next_token_item) is not None:
                    current_c += 1
                    next_coord_tuple = (current_r, current_c)
                elif next_token_item == self.serializer.tokenizer.row_sep_token_id:
                    current_r += 1
                    current_c = 0
                    next_coord_tuple = (current_r, 0)
                else:
                    next_coord_tuple = (-1, -1)

                next_coord = torch.tensor([[next_coord_tuple]], dtype=torch.long, device=self.device)
                coords = torch.cat([coords, next_coord], dim=1)

                if next_token_item == self.serializer.tokenizer.eos_token_id or next_token_item == self.serializer.tokenizer.vocab["<im_end>"]:
                    break

        prompt_len = input_ids.shape[1]
        return tokens[0, prompt_len:].tolist()


class EvaluationStep:

    def __init__(self, model: ArcTransformer, serializer: GridSerializer, deserializer: GridDeserializer, observer: Observer, device: torch.device, train_dataset: Any, config: TrainConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.serializer = serializer
        self.deserializer = deserializer
        self.observer = observer
        self.device = device
        self.config = config
        self.evaluator = SimpleEvaluator(self.model, self.serializer, self.deserializer, self.observer, self.device, self.config.generation)

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

                pred_grid, generated_tokens = self.evaluator.evaluate_single(task_data)

                is_correct = 0
                if torch.equal(pred_grid.to(self.device), target_grid_raw):
                    is_correct = 1

                if not visualized_this_loop:
                    input_grid_raw = torch.tensor(task_data['test'][0]['input'])
                    self.observer.visualize_evaluation_sample(input_grid_raw, target_grid_raw, pred_grid, generated_tokens, global_step)
                    visualized_this_loop = True

                total_correct += is_correct
                evaluated_count += 1
                progress.update(task, advance=1)

        return total_correct, evaluated_count

    def run(self, eval_loader: Any, current_task_idx: int, global_step: int, verbose: bool = False) -> dict[str, float]:
        self.model.eval()
        self.observer.console.print(f"\n[bold cyan]--- Running 3-Phase Evaluation @ Step {global_step} ---[/bold cyan]")

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
