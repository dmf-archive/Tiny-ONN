import torch
import torch.nn.functional as F
from typing import Literal

from exp.tiny_onn_arc.utils.batch_transforms import _apply_single_augmentation
from exp.tiny_onn_arc.model import DynONNForArc


def _sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def _sample_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probabilities.view(-1, logits.shape[-1]), num_samples=1).view(logits.shape[:-1])


def _sample_top_p(logits: torch.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., :min_tokens_to_keep] = False

    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    logits_to_remove = torch.full_like(logits, -float("Inf"))
    filtered_logits = torch.where(indices_to_remove, logits_to_remove, logits)

    probabilities = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probabilities.view(-1, logits.shape[-1]), num_samples=1).view(logits.shape[:-1])


def generate_candidates(
    model: DynONNForArc,
    input_grid: torch.Tensor,
    target_h: int,
    target_w: int,
    num_candidates: int = 8,
    strategy: Literal["greedy", "temperature", "top-p"] = "greedy",
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[torch.Tensor]:
    model.eval()
    candidates = []

    with torch.no_grad():
        encoder_output, _ = model.encoder(input_grid.unsqueeze(0))
        logits, _ = model.decoder(encoder_output, target_h, target_w)
        logits = logits.squeeze(0)

        for _ in range(num_candidates):
            if strategy == "greedy":
                sample = _sample_greedy(logits)
            elif strategy == "temperature":
                sample = _sample_temperature(logits, temperature)
            elif strategy == "top-p":
                sample = _sample_top_p(logits, top_p)
            else:
                raise ValueError(f"Unknown sampling strategy: {strategy}")
            candidates.append(sample)

    model.train()
    return candidates


def calculate_augmented_score(
    model: DynONNForArc,
    input_grid: torch.Tensor,
    candidate_grid: torch.Tensor,
    num_augmentations: int = 8
) -> float:
    total_log_prob = 0.0
    device = next(model.parameters()).device

    for _ in range(num_augmentations):
        aug_input = _apply_single_augmentation(input_grid)
        aug_candidate = _apply_single_augmentation(candidate_grid)

        if aug_input.shape != input_grid.shape and (aug_candidate.shape[0] != aug_input.shape[0] or aug_candidate.shape[1] != aug_input.shape[1]):
            aug_candidate = F.interpolate(aug_candidate.unsqueeze(0).unsqueeze(0).float(), size=aug_input.shape, mode='nearest').squeeze(0).squeeze(0).long()

        with torch.no_grad():
            target_h, target_w = aug_candidate.shape[0], aug_candidate.shape[1]
            logits, _ = model(input_grid=aug_input.unsqueeze(0).to(device), target_h=target_h, target_w=target_w)
            log_probs = F.log_softmax(logits, dim=3)
            target_log_probs = log_probs.gather(3, aug_candidate.unsqueeze(0).unsqueeze(3).to(device)).squeeze(3)
            total_log_prob += target_log_probs.sum().item()

    return total_log_prob
