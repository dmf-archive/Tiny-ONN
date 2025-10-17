import itertools
import random
import sys
from typing import Any

import numpy as np
import torch
from rich.progress import Progress
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Subset

from .config import GenerationConfig, TrainConfig
from .data import GridDeserializer, GridSerializer
from .model import ArcTransformer
from .observer import Observer


class ArcGenerator:

    def __init__(self, model: ArcTransformer, serializer: GridSerializer, deserializer: GridDeserializer, observer: Observer, device: torch.device, config: GenerationConfig):
        self.model = model
        self.serializer = serializer
        self.deserializer = deserializer
        self.observer = observer
        self.device = device
        self.config = config
        self.eos_token_id = self.serializer.tokenizer.eos_token_id

    @torch.no_grad()
    def generate(self, task_data: dict[str, Any]) -> tuple[torch.Tensor, list[int], list[float]]:
        output_grid = task_data["test"][0].get("output", [])
        num_rows = len(output_grid)
        num_pixels = sum(len(row) for row in output_grid)
        max_new_tokens = num_pixels + num_rows + 2

        prompt_ids, prompt_coords = self.serializer.serialize_for_inference(task_data)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        prompt_coords_tensor = torch.tensor([prompt_coords], dtype=torch.long, device=self.device)

        if self.config.use_dfs:
            results = self._dfs_search(prompt_tensor, prompt_coords_tensor, max_new_tokens)
            generated_tokens = results[0][0].tolist() if results else []
            probabilities = [-1.0] * len(generated_tokens)
        else:
            generated_tokens, probabilities = self._greedy_search(prompt_tensor, prompt_coords_tensor, max_new_tokens)

        pred_grid = self.deserializer.deserialize(generated_tokens)
        return pred_grid, generated_tokens, probabilities

    def _greedy_search(self, input_ids: torch.Tensor, input_coords: torch.Tensor, max_new_tokens: int) -> tuple[list[int], list[float]]:
        tokens = input_ids.clone()
        coords = input_coords.clone()
        past_key_values = None
        current_r, current_c = 0, -1
        probabilities = []

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION), torch.autocast(
            device_type=self.device.type, dtype=torch.float16
        ):
            for _ in range(max_new_tokens):
                model_input = tokens if past_key_values is None else tokens[:, -1:]
                outputs = self.model(
                    model_input, past_key_values=past_key_values, return_dict=True
                )
                logits, past_key_values = outputs["logits"], outputs["past_key_values"]

                next_token_logits = logits[:, -1, :]
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(next_token_probs, dim=-1, keepdim=True)

                prob = next_token_probs[0, next_token.item()].item()
                probabilities.append(prob)

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

                if next_token_item == self.eos_token_id or next_token_item == self.serializer.tokenizer.vocab["<im_end>"]:
                    break

        return tokens[0, input_ids.shape[1] :].tolist(), probabilities

    def _dfs_search(self, input_ids: torch.Tensor, input_coords: torch.Tensor, max_new_tokens: int) -> list[tuple[np.ndarray, float]]:
        sys.setrecursionlimit(1000 + max_new_tokens)
        input_ids_squeezed = input_ids.squeeze(0)
        pos = len(input_ids_squeezed)

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION), torch.autocast(
            device_type=self.device.type, dtype=torch.float16
        ):
            outputs = self.model(input_ids, return_dict=True)
            logits, cache = outputs["logits"], [outputs["past_key_values"]]

        logits_sliced = logits[0, pos - 1 :]
        result = self._explore(
            logits_sliced, [], max_new_tokens, -np.log(self.config.min_prob), pos, cache, input_coords
        )
        return sorted([(np.array(suffix[::-1]), score_val) for suffix, score_val in result], key=lambda x: x[1])

    def _explore(self, logits: torch.Tensor, path: list[int], max_new_tokens: int, max_score: float, pos: int, cache: list, coords: torch.Tensor, score: float = 0.0) -> list[tuple[list[int], float]]:
        first_token_logits, remaining_logits = logits[0], (logits[1:] if len(logits) > 1 else None)
        nll = -first_token_logits.detach().float().log_softmax(-1).cpu()
        softmax = list(enumerate(nll))

        if path:
            softmax[0], softmax[path[0]], path = softmax[path[0]], softmax[0], path[1:]

        return_suffixes = []
        current_r, current_c = coords[0, -1, 0].item(), coords[0, -1, 1].item()

        for i, s in softmax:
            next_score = score + s.item()
            if next_score < max_score:
                if i == self.eos_token_id:
                    suffixes = [([], next_score)]
                elif max_new_tokens > 1:
                    if remaining_logits is None:
                        if pos < cache[0][0][0].shape[2]:
                            cache[0] = tuple(tuple(c[:, :, :pos] for c in l) for l in cache[0])

                        next_token_tensor = torch.tensor([[i]], device=self.device)
                        if self.serializer.tokenizer.token_id_to_color(i) is not None:
                            current_c += 1
                            next_coord_tuple = (current_r, current_c)
                        elif i == self.serializer.tokenizer.row_sep_token_id:
                            current_r += 1
                            current_c = 0
                            next_coord_tuple = (current_r, 0)
                        else:
                            next_coord_tuple = (-1, -1)
                        next_coord = torch.tensor([[next_coord_tuple]], dtype=torch.long, device=self.device)

                        outputs = self.model(
                            next_token_tensor, past_key_values=cache[0], return_dict=True
                        )
                        new_logits, cache[0] = outputs["logits"], outputs["past_key_values"]
                        new_logits = new_logits[0]
                    else:
                        new_logits, next_coord = remaining_logits, coords

                    suffixes = self._explore(new_logits, path, max_new_tokens - 1, max_score, pos + 1, cache, next_coord, next_score)
                else:
                    suffixes = []

                for suffix in suffixes:
                    suffix[0].append(i)
                return_suffixes.extend(suffixes)
            remaining_logits = None
        return return_suffixes


class EvaluationStep:

    def __init__(self, model: ArcTransformer, serializer: GridSerializer, deserializer: GridDeserializer, observer: Observer, device: torch.device, train_dataset: Any, config: TrainConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.serializer = serializer
        self.deserializer = deserializer
        self.observer = observer
        self.device = device
        self.config = config
        self.generator = ArcGenerator(self.model, self.serializer, self.deserializer, self.observer, self.device, self.config.generation)

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
                pred_grid, generated_tokens, probabilities = self.generator.generate(task_data)

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

    def run(self, eval_loader: Any, current_task_idx: int, global_step: int, curriculum_stage: int, advance_curriculum_fn: callable, verbose: bool = False) -> dict[str, float]:
        self.model.eval()
        self.observer.console.print(f"\n[bold cyan]--- Running 3-Phase Evaluation @ Step {global_step} (Stage {curriculum_stage}) ---[/bold cyan]")

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
