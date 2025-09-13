import itertools
from typing import Any

import torch
import torch.nn.functional as F

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
    def dfs_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int,
        threshold: float
    ) -> list[tuple[float, torch.Tensor]]:
        self.model.eval()

        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.model.num_layers
        valid_sequences: list[tuple[float, torch.Tensor]] = []
        max_len = input_ids.shape[1] + max_new_tokens

        vocab_size = self.model.vocab_size
        top_k = min(5, vocab_size)

        def _explore(tokens: torch.Tensor, score: float):
            nonlocal past_key_values
            if tokens.shape[1] >= max_len or tokens[0, -1] == eos_token_id:
                if tokens[0, -1] == eos_token_id:
                    valid_sequences.append((score, tokens.clone()))
                return

            len_before_recursion = tokens.shape[1]
            model_input = tokens if past_key_values[0] is None else tokens[:, -1:]

            logits, _, _, _, _, _, past_key_values = self.model(
                model_input, prior_std=1.0, kl_epsilon=1e-9, past_key_values=past_key_values
            )

            next_token_logits = logits[:, -1, :]
            next_token_log_prob = F.log_softmax(next_token_logits, dim=-1)

            top_k_log_probs, top_k_indices = torch.topk(next_token_log_prob, k=top_k, dim=-1)

            for i in range(top_k_indices.shape[1]):
                token_id = top_k_indices[0, i]
                log_prob = top_k_log_probs[0, i].item()
                next_score = score + log_prob

                if next_score >= threshold:
                    next_tokens = torch.cat([tokens, token_id.view(1, 1)], dim=1)
                    _explore(next_tokens, next_score)

            if past_key_values[0] is not None:
                trimmed_pkv: list[tuple[torch.Tensor, torch.Tensor] | None] = []
                for k_v_pair in past_key_values:
                    if k_v_pair is not None:
                        k, v = k_v_pair
                        k = k[:, :len_before_recursion, :]
                        v = v[:, :len_before_recursion, :]
                        trimmed_pkv.append((k, v))
                    else:
                        trimmed_pkv.append(None)
                past_key_values = trimmed_pkv

        _explore(input_ids.clone(), 0.0)
        return sorted(valid_sequences, key=lambda x: x[0], reverse=True)

    @torch.no_grad()
    def run(self, eval_loader: Any, global_step: int, quick_eval: bool = True, verbose: bool = False) -> dict[str, float]:
        self.model.eval()
        num_samples_to_eval = 10 if quick_eval and len(eval_loader) > 10 else len(eval_loader)
        eval_title = "Quick Eval" if quick_eval else "Full Eval"
        self.observer.console.print(f"\n[bold cyan]--- Running {eval_title} ({num_samples_to_eval} samples) ---[/bold cyan]")

        total_grid_acc, evaluated_count = 0, 0
        visualized = False

        for batch in itertools.islice(eval_loader, num_samples_to_eval):
            if not batch: continue

            task_data = batch['task_data'][0]
            input_grid_raw = torch.tensor(task_data['test'][0]['input'])
            target_grid_raw = torch.tensor(task_data['test'][0]['output'])

            prompt_ids = self.serializer.serialize_for_inference(task_data)
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)

            h, w = target_grid_raw.shape
            max_new_tokens = int(h * w * 1.5) + 30

            solutions = self.dfs_generate(
                input_ids=prompt_tensor,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.serializer.tokenizer.eos_token_id,
                threshold=-5.0
            )

            pred_grid_1, pred_grid_2 = None, None
            is_correct = 0

            if solutions:
                prompt_len = prompt_tensor.shape[1]
                pred_tokens_1 = solutions[0][1][0, prompt_len:].tolist()
                pred_grid_1 = self.deserializer.deserialize(pred_tokens_1)

                if torch.equal(pred_grid_1, target_grid_raw):
                    is_correct = 1
                elif len(solutions) > 1:
                    pred_tokens_2 = solutions[1][1][0, prompt_len:].tolist()
                    pred_grid_2 = self.deserializer.deserialize(pred_tokens_2)
                    if torch.equal(pred_grid_2, target_grid_raw):
                        is_correct = 1

            if not visualized:
                self.observer.visualize_evaluation_sample(input_grid_raw, target_grid_raw, pred_grid_1, global_step)
                if verbose and solutions:
                    self.observer.console.print(f"[bold]Top solution decoded:[/bold] {self.serializer.tokenizer.decode(pred_tokens_1)}")
                visualized = True

            total_grid_acc += is_correct
            evaluated_count += 1

        avg_grid_acc = total_grid_acc / evaluated_count if evaluated_count > 0 else 0
        metrics = {"eval_grid_acc": avg_grid_acc, "total_count": float(evaluated_count)}
        self.observer.log_eval_summary(metrics, global_step)

        self.model.train()
        return metrics
