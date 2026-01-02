import itertools
import random
import sys
from typing import Any

import numpy as np
import torch
from rich.progress import Progress
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
    def generate(self, task_data: dict[str, Any]) -> tuple[torch.Tensor, list[int], list[float], float]:
        test_data = task_data.get("test", [])
        if test_data and len(test_data) > 0 and "output" in test_data[0]:
            output_grid = test_data[0]["output"]
        else:
            output_grid = []

        num_rows = len(output_grid) if output_grid else 0
        num_pixels = sum(len(row) for row in output_grid) if output_grid else 0
        max_new_tokens = num_pixels + num_rows + 2

        prompt_ids, _ = self.serializer.serialize_for_inference(task_data)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        if self.config.use_dfs:
            results = self._dfs_search(prompt_tensor, max_new_tokens)
            generated_tokens = results[0][0].tolist() if results else []
            probabilities = [-1.0] * len(generated_tokens)
            perplexity = -1.0
        else:
            outputs = self.model.model.generate(
                prompt_tensor,
                max_new_tokens=max_new_tokens,
                eos_token_id=[self.eos_token_id, self.serializer.tokenizer.vocab["<im_end>"]],
                pad_token_id=self.serializer.tokenizer.pad_token_id if self.serializer.tokenizer.pad_token_id is not None else self.eos_token_id,
                attention_mask=torch.ones_like(prompt_tensor),
                do_sample=False, # greedy
                output_scores=True,
                return_dict_in_generate=True,
            )
            generated_tokens = outputs.sequences[0, prompt_tensor.shape[1]:].tolist()

            # Extract probabilities for the generated tokens
            probs = torch.stack(outputs.scores, dim=1).softmax(-1)
            gen_sequences = outputs.sequences[:, prompt_tensor.shape[1]:]
            gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
            probabilities = gen_probs[0].tolist()
            
            # Calculate perplexity
            log_probs = torch.log(gen_probs[0] + 1e-10)
            perplexity = torch.exp(-torch.mean(log_probs)).item()

        pred_grid = self.deserializer.deserialize(generated_tokens)
        return pred_grid, generated_tokens, probabilities, perplexity

    def _dfs_search(self, input_ids: torch.Tensor, max_new_tokens: int) -> list[tuple[np.ndarray, float]]:
        sys.setrecursionlimit(1000 + max_new_tokens)
        pos = input_ids.shape[1]
        position_ids = torch.arange(pos, device=self.device).unsqueeze(0)

        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16
        ):
            outputs = self.model(input_ids, coords=None, position_ids=position_ids, return_dict=True)
            logits, cache = outputs["logits"], [outputs["past_key_values"]]

        logits_sliced = logits[0, pos - 1 :]

        result = self._explore(
            logits_sliced, [], max_new_tokens, -np.log(self.config.min_prob), pos, cache, position_ids, 0, -1
        )
        return sorted([(np.array(suffix[::-1]), score_val) for suffix, score_val in result], key=lambda x: x[1])

    def _explore(self, logits: torch.Tensor, path: list[int], max_new_tokens: int, max_score: float, pos: int, cache: list, position_ids: torch.Tensor, current_r: int, current_c: int, score: float = 0.0) -> list[tuple[list[int], float]]:
        first_token_logits, remaining_logits = logits[0], (logits[1:] if len(logits) > 1 else None)
        nll = -first_token_logits.detach().float().log_softmax(-1).cpu()
        softmax = list(enumerate(nll))

        if path:
            softmax[0], softmax[path[0]], path = softmax[path[0]], softmax[0], path[1:]

        return_suffixes = []

        for i, s in softmax:
            next_score = score + s.item()
            if next_score < max_score:
                if i == self.eos_token_id:
                    suffixes = [([], next_score)]
                elif max_new_tokens > 1:
                    next_position_ids = torch.tensor([[position_ids[0, -1] + 1]], device=self.device)

                    if remaining_logits is None:
                        if pos < cache[0][0][0].shape[2]:
                            cache[0] = tuple(tuple(c[:, :, :pos] for c in l) for l in cache[0])

                        next_token_tensor = torch.tensor([[i]], device=self.device)

                        outputs = self.model(
                            next_token_tensor, coords=None, position_ids=next_position_ids, past_key_values=cache[0], return_dict=True
                        )
                        new_logits, cache[0] = outputs["logits"], outputs["past_key_values"]
                        new_logits = new_logits[0]
                    else:
                        new_logits = remaining_logits

                    suffixes = self._explore(new_logits, path, max_new_tokens - 1, max_score, pos + 1, cache, next_position_ids, 0, -1, next_score)
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

                if "test" not in task_data or not task_data["test"] or len(task_data["test"]) == 0:
                    progress.update(task, advance=1)
                    continue

                test_item = task_data["test"][0]
                if not isinstance(test_item, dict) or "output" not in test_item:
                    progress.update(task, advance=1)
                    continue

                target_grid_raw = torch.tensor(test_item['output'], device=self.device)
                pred_grid, generated_tokens, probabilities, perplexity = self.generator.generate(task_data)

                is_correct = 1 if torch.equal(pred_grid.to(self.device), target_grid_raw) else 0

                if not visualized_this_loop and "input" in test_item:
                    input_grid_raw = torch.tensor(test_item["input"])
                    decoded_tokens = self.serializer.tokenizer.decode(generated_tokens)
                    self.observer.visualize_evaluation_sample(
                        input_grid_raw,
                        target_grid_raw,
                        pred_grid,
                        decoded_tokens,
                        probabilities,
                        perplexity,
                        global_step,
                    )
                    visualized_this_loop = True

                total_correct += is_correct
                evaluated_count += 1
                progress.update(task, advance=1)

        return total_correct, evaluated_count

    def run(self, eval_loader: Any, total_tasks_processed: int, global_step: int, verbose: bool = False) -> dict[str, float]:
        self.model.eval()
        self.observer.console.print(f"\n[bold cyan]--- Running Evaluation @ Step {global_step} ---[/bold cyan]")

        # 第一阶段：抗遗忘检查
        num_training_tasks = len(self.train_dataset)
        if num_training_tasks > 0:
            num_to_sample = min(5, num_training_tasks)
            indices = random.sample(range(num_training_tasks), k=num_to_sample)
            historical_subset = Subset(self.train_dataset, indices)
            historical_loader = DataLoader(historical_subset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
            correct, total = self._run_eval_loop(historical_loader, num_to_sample, "Forgetting Check", global_step)
            if correct < total:
                self.observer.console.print(f"[bold red]Forgetting detected! ({correct}/{total} passed).[/bold red]")
                # 如果抗遗忘检查失败，跳过验证集评估，提前返回
                self.model.train()
                return {"eval_grid_acc": 0.0, "total_count": 0.0, "forgetting_detected": True}
            else:
                self.observer.console.print(f"[bold green]No forgetting detected ({correct}/{total}).[/bold green]")

        # 第二阶段：验证集评估（只有在抗遗忘通过后才进行）
        num_full = len(eval_loader)
        if num_full > 0:
            num_to_eval = min(15, num_full)  # 限制评估样本数量
            correct_full, total_full = self._run_eval_loop(eval_loader, num_to_eval, "Validation Evaluation", global_step)
            final_acc = correct_full / total_full
            metrics = {"eval_grid_acc": final_acc, "total_count": float(total_full), "forgetting_detected": False}
            self.observer.log_eval_summary(metrics, global_step)
        else:
            metrics = {"eval_grid_acc": 0.0, "total_count": 0.0, "forgetting_detected": False}

        self.model.train()
        return metrics
