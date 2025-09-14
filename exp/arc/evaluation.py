import itertools
import sys
from rich.progress import Progress
from typing import Any, Optional, List, Dict, Tuple

import torch
import torch.nn.functional as F

from .data import GridDeserializer, GridSerializer
from .consistency import ConsistencyTools
from .model import ArcTransformer
from .observer import Observer

from .config import GenerationConfig


# === Generation Framework (as per 'transformers' philosophy) ===

class LogitsProcessor:
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

class TopPLogitsWarper(LogitsProcessor):
    def __init__(self, top_p: float):
        self.top_p = top_p

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        scores[:, indices_to_remove] = -float("inf")
        return scores

class StoppingCriteria:
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        raise NotImplementedError

class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids.shape[-1] >= self.max_length

class ArcGenerator:
    def __init__(self, model: ArcTransformer):
        self.model = model

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        **kwargs,
    ):
        logits_processor = TopPLogitsWarper(config.top_p) if config.top_p < 1.0 and config.do_sample else None
        stopping_criteria = MaxLengthCriteria(config.max_new_tokens + input_ids.shape[-1])

        if config.num_beams > 1:
            return self.beam_search(input_ids, config, logits_processor, stopping_criteria, **kwargs)
        if config.do_sample:
            return self.sample(input_ids, config, logits_processor, stopping_criteria, **kwargs)
        return self.greedy_search(input_ids, config, logits_processor, stopping_criteria, **kwargs)

    def sample(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        logits_processor: Optional[LogitsProcessor] = None,
        stopping_criteria: Optional[StoppingCriteria] = None,
        **kwargs,
    ) -> List[torch.Tensor]:
        final_sequences = []
        for _ in range(config.num_return_sequences):
            tokens = input_ids.clone()
            past_key_values = None

            while not stopping_criteria(tokens, None):
                model_input = tokens if past_key_values is None else tokens[:, -1:]
                
                logits, _, _, _, _, _, past_key_values = self.model(
                    model_input, past_key_values=past_key_values
                )

                next_token_logits = logits[:, -1, :]

                if logits_processor:
                    next_token_logits = logits_processor(tokens, next_token_logits)

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                tokens = torch.cat([tokens, next_token], dim=-1)

                if config.eos_token_id is not None and next_token.item() == config.eos_token_id:
                    break
            
            final_sequences.append(tokens)

        return final_sequences

    def beam_search(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        logits_processor: Optional[LogitsProcessor] = None,
        stopping_criteria: Optional[StoppingCriteria] = None,
        **kwargs,
    ) -> List[torch.Tensor]:
        
        beam_scorer = BeamSearchScorer(
            batch_size=input_ids.shape[0],
            num_beams=config.num_beams,
            device=self.model.device,
            eos_token_id=config.eos_token_id,
            num_return_sequences=config.num_return_sequences,
        )

        input_ids = input_ids.repeat_interleave(config.num_beams, dim=0)
        
        batch_beam_size, cur_len = input_ids.shape
        beam_scores = torch.zeros((input_ids.shape[0],), dtype=torch.float, device=input_ids.device)
        
        past_key_values = None

        while not stopping_criteria(input_ids, None):
            model_input = input_ids if past_key_values is None else input_ids[:, -1:]
            
            logits, _, _, _, _, _, past_key_values = self.model(
                model_input, past_key_values=past_key_values
            )
            
            next_token_logits = logits[:, -1, :]
            
            if logits_processor:
                next_token_logits = logits_processor(input_ids, next_token_logits)

            scores = F.log_softmax(next_token_logits, dim=-1)
            next_token_scores = scores + beam_scores[:, None].expand_as(scores)
            
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(input_ids.shape[0] // config.num_beams, -1)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * config.num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size
            
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
            )
            
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            
            if past_key_values:
                past_key_values = [
                    (k[beam_idx], v[beam_idx]) for k, v in past_key_values
                ]

            if beam_scorer.is_done:
                break
        
        return beam_scorer.finalize(input_ids, beam_scores)

    def greedy_search(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        logits_processor: Optional[LogitsProcessor] = None,
        stopping_criteria: Optional[StoppingCriteria] = None,
        **kwargs,
    ) -> List[torch.Tensor]:
        tokens = input_ids.clone()
        past_key_values = None

        while not stopping_criteria(tokens, None):
            model_input = tokens if past_key_values is None else tokens[:, -1:]
            
            logits, _, _, _, _, _, past_key_values = self.model(
                model_input, past_key_values=past_key_values
            )

            next_token_logits = logits[:, -1, :]

            if logits_processor:
                next_token_logits = logits_processor(tokens, next_token_logits)

            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            tokens = torch.cat([tokens, next_token], dim=-1)

            if config.eos_token_id is not None and next_token.item() == config.eos_token_id:
                break
        
        return [tokens]


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
        self.consistency_tools = ConsistencyTools()
        self.scorer = CandidateScorer(self.model, self.serializer, self.consistency_tools, self.device)
        self.generator = ArcGenerator(self.model)

    @torch.no_grad()
    def run(self, eval_loader: Any, global_step: int, quick_eval: bool = True, verbose: bool = False) -> dict[str, float]:
        self.model.eval()
        num_samples_to_eval = 10 if quick_eval and len(eval_loader) > 10 else len(eval_loader)
        eval_title = "Quick Eval" if quick_eval else "Full Eval"
        self.observer.console.print(f"\n[bold cyan]--- Running {eval_title} ({num_samples_to_eval} samples) ---[/bold cyan]")

        total_grid_acc, evaluated_count = 0, 0
        visualized = False

        with Progress(console=self.observer.console, transient=True) as progress:
            task = progress.add_task(f"[cyan]Running {eval_title}...", total=num_samples_to_eval)

            for batch in itertools.islice(eval_loader, num_samples_to_eval):
                if not batch:
                    progress.update(task, advance=1)
                    continue

                task_data = batch['task_data'][0]
                input_grid_raw = torch.tensor(task_data['test'][0]['input'])

                target_grid_raw = torch.tensor(task_data['test'][0]['output'])
                h_in, w_in = input_grid_raw.shape
                max_new_tokens = int(h_in * w_in * 9) + 50

                prompt_ids = self.serializer.serialize_for_inference(task_data)
                prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)

                generation_config = GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    num_beams=2,
                    do_sample=False,
                    eos_token_id=self.serializer.tokenizer.eos_token_id,
                    num_return_sequences=2
                )

                generated_sequences = self.generator.generate(
                    input_ids=prompt_tensor,
                    config=generation_config
                )
                
                all_candidates: Dict[Tuple, torch.Tensor] = {}
                prompt_len = prompt_tensor.shape[1]
                for seq in generated_sequences:
                    pred_tokens = seq[prompt_len:].tolist()
                    pred_grid = self.deserializer.deserialize(pred_tokens)
                    grid_tuple = tuple(map(tuple, pred_grid.tolist()))
                    if grid_tuple:
                        all_candidates[grid_tuple] = pred_grid

                scored_candidates = []
                for grid_tuple, candidate_grid in all_candidates.items():
                    final_score = self.scorer.score_candidate(task_data, candidate_grid.to(self.device))
                    scored_candidates.append((final_score, candidate_grid))

                sorted_candidates = sorted(scored_candidates, key=lambda x: x[0], reverse=True)
                
                is_correct = 0
                pred_grid_1 = None
                if sorted_candidates:
                    pred_grid_1 = sorted_candidates[0][1].to(self.device)
                    if torch.equal(pred_grid_1, target_grid_raw.to(self.device)):
                        is_correct = 1
                
                if not visualized:
                    self.observer.visualize_evaluation_sample(input_grid_raw, target_grid_raw, pred_grid_1, global_step)
                    visualized = True

                total_grid_acc += is_correct
                evaluated_count += 1
                progress.update(task, advance=1)

        avg_grid_acc = total_grid_acc / evaluated_count if evaluated_count > 0 else 0
        metrics = {"eval_grid_acc": avg_grid_acc, "total_count": float(evaluated_count)}
        self.observer.log_eval_summary(metrics, global_step)

        self.model.train()
        return metrics

class BeamSearchScorer:
    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: float = 1.0,
        do_early_stopping: bool = False,
        eos_token_id: Optional[int] = None,
        num_return_sequences: int = 1,
    ):
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.eos_token_id = eos_token_id
        self.num_return_sequences = num_return_sequences

        self.beam_hypos = [[] for _ in range(batch_size)]
        self._is_done = False

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
    ) -> Dict[str, torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = self.batch_size
        num_beams = self.num_beams

        next_beam_scores = torch.zeros((batch_size, num_beams), dtype=next_scores.dtype, device=self.device)
        next_beam_tokens = torch.zeros((batch_size, num_beams), dtype=next_tokens.dtype, device=self.device)
        next_beam_indices = torch.zeros((batch_size, num_beams), dtype=next_indices.dtype, device=self.device)

        for batch_idx in range(batch_size):
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                if beam_idx >= num_beams:
                    break
                
                beam_id = next_index.item()
                
                if self.eos_token_id is not None and next_token.item() == self.eos_token_id:
                    self.beam_hypos[batch_idx].append((next_score.item(), input_ids[batch_idx * num_beams + beam_id]))
                else:
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_idx * num_beams + beam_id
                    beam_idx += 1
        
        self._is_done = all(len(b) >= self.num_beams for b in self.beam_hypos)

        return {
            "next_beam_scores": next_beam_scores.view(-1),
            "next_beam_tokens": next_beam_tokens.view(-1),
            "next_beam_indices": next_beam_indices.view(-1),
        }

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
    ) -> List[torch.LongTensor]:
        
        for batch_idx in range(self.batch_size):
            for beam_idx in range(self.num_beams):
                final_score = final_beam_scores[batch_idx * self.num_beams + beam_idx].item()
                final_tokens = input_ids[batch_idx * self.num_beams + beam_idx]
                self.beam_hypos[batch_idx].append((final_score, final_tokens))

        sorted_hypos = []
        for batch_hypos in self.beam_hypos:
            sorted_batch = sorted(batch_hypos, key=lambda x: x[0], reverse=True)
            sorted_hypos.extend(s for _, s in sorted_batch)

        return sorted_hypos[:self.num_return_sequences]

    @property
    def is_done(self) -> bool:
        return self._is_done

class CandidateScorer:
    def __init__(self, model: ArcTransformer, serializer: GridSerializer, consistency_tools: Any, device: torch.device):
        self.model = model
        self.serializer = serializer
        self.consistency_tools = consistency_tools
        self.device = device

    @torch.no_grad()
    def score_candidate(self, task_data: dict[str, Any], solution_grid: torch.Tensor) -> float:
        total_log_prob = 0.0
        transforms = self.consistency_tools.get_transforms()
        input_grid_raw = torch.tensor(task_data['test'][0]['input'])

        for transform in transforms:
            input_grid_aug = transform(input_grid_raw).to(self.device)
            solution_grid_aug = transform(solution_grid).to(self.device)

            view_task_data = {
                'train': task_data['train'],
                'test': [{'input': input_grid_aug.tolist(), 'output': solution_grid_aug.tolist()}]
            }
            
            full_ids, labels = self.serializer.serialize_task_with_context(view_task_data)
            
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)
            labels = torch.tensor([labels], dtype=torch.long, device=self.device)

            logits, _, _, _, _, _, _ = self.model(input_ids)
            
            shifted_logits = logits[:, :-1, :]
            shifted_labels = labels[:, 1:]
            
            log_probs = F.log_softmax(shifted_logits, dim=-1)

            mask = (shifted_labels != -100)
            
            active_labels = shifted_labels[mask]
            active_log_probs = log_probs[mask]
            
            if active_labels.numel() > 0:
                target_log_probs = active_log_probs.gather(1, active_labels.unsqueeze(1)).squeeze(1)
                total_log_prob += target_log_probs.sum().item()

        return total_log_prob
