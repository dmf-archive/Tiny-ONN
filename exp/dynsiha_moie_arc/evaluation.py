import itertools
from typing import Any

import torch

from .data import GridDeserializer, GridSerializer
from .model import ArcTransformer
from .observer import Observer


class EvaluationStep:
    def __init__(
        self,
        model: ArcTransformer,
        serializer: GridSerializer,
        deserializer: GridDeserializer,
        observer: Observer,
        device: torch.device,
    ):
        self.model = model
        self.serializer = serializer
        self.deserializer = deserializer
        self.observer = observer
        self.device = device

    @torch.no_grad()
    def run(self, eval_loader: Any, global_step: int, quick_eval: bool = True) -> dict[str, float]:
        self.model.eval()
        num_samples_to_eval = 10 if quick_eval and len(eval_loader) > 10 else len(eval_loader)
        eval_title = "Quick Eval" if quick_eval else "Full Eval"
        self.observer.console.print(f"\n[bold cyan]--- Running {eval_title} ({num_samples_to_eval} samples) ---[/bold cyan]")

        total_grid_acc, evaluated_count = 0, 0

        for i, batch in enumerate(itertools.islice(eval_loader, num_samples_to_eval)):
            task_data = batch["task_data"]
            input_grid_raw = torch.tensor(task_data['test'][0]['input'])
            target_grid_raw = torch.tensor(task_data['test'][0]['output'])

            problem_ids_list = self.serializer.serialize_for_inference(task_data)
            problem_ids = torch.tensor(problem_ids_list, dtype=torch.long).unsqueeze(0).to(self.device)

            # Estimate target length for generation
            output_grid = task_data['test'][0]['output']
            target_len = len(output_grid) * len(output_grid[0]) + len(output_grid) * 3 + 20 # Add buffer

            # DFS Generate returns a list of (score, tensor) tuples, sorted by score
            solutions = self.model.dfs_generate(
                input_ids=problem_ids,
                max_new_tokens=target_len,
                eos_token_id=self.serializer.tokenizer.eos_token_id,
                threshold=-5.0 # Heuristic threshold
            )

            pred_grid_1, pred_grid_2 = None, None
            is_correct = 0

            if solutions:
                prompt_len = problem_ids.shape[1]
                # Attempt 1 (best score)
                pred_tokens_1 = solutions[0][1][0, prompt_len:].tolist()
                pred_grid_1 = self.deserializer.deserialize(pred_tokens_1)
                
                if torch.equal(pred_grid_1, target_grid_raw):
                    is_correct = 1
                else:
                    # Attempt 2 (second best, if available)
                    if len(solutions) > 1:
                        pred_tokens_2 = solutions[1][1][0, prompt_len:].tolist()
                        pred_grid_2 = self.deserializer.deserialize(pred_tokens_2)
                        if torch.equal(pred_grid_2, target_grid_raw):
                            is_correct = 1
                    else:
                        pred_grid_2 = pred_grid_1 # Repeat attempt 1 if no other solution

            if i == 0:
                self.observer.visualize_evaluation_sample(input_grid_raw, target_grid_raw, pred_grid_1, global_step)
            
            total_grid_acc += is_correct
            evaluated_count += 1

        avg_grid_acc = total_grid_acc / evaluated_count if evaluated_count > 0 else 0
        metrics = {"grid_acc": avg_grid_acc, "total_count": float(evaluated_count)}
        self.observer.log_eval_summary(metrics, global_step)

        self.model.train()
        return metrics
