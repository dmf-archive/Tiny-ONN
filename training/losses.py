import torch
import torch.nn.functional as F


def compute_smk_weighted_loss(
    surprise_matrix: torch.Tensor,
    activated_scores: torch.Tensor,
) -> torch.Tensor:
    routing_mask = activated_scores > 0
    masked_scores = torch.where(
        routing_mask,
        activated_scores,
        torch.tensor(
            float("-inf"), device=activated_scores.device, dtype=activated_scores.dtype
        ),
    )
    log_probs = F.log_softmax(masked_scores, dim=-1)

    surprise_clean = surprise_matrix
    surprise_clean[torch.isinf(surprise_clean)] = 1e9

    smk_loss = torch.sum(log_probs.exp() * surprise_clean, dim=-1).mean()
    return smk_loss


def compute_load_balancing_loss(
    routing_weights: torch.Tensor, activated_scores: torch.Tensor
) -> torch.Tensor:
    num_experts = routing_weights.shape[-1]

    # Sum across tokens and layers for each expert, works for both 2D and 3D
    tokens_per_expert = routing_weights.sum(dim=list(range(routing_weights.dim() - 1)))
    load_loss = (
        num_experts
        * torch.sum(tokens_per_expert**2)
        / (torch.sum(tokens_per_expert) ** 2)
    )

    importance_per_expert = activated_scores.sum(dim=list(range(activated_scores.dim() - 1)))
    imp_loss = (
        num_experts
        * torch.sum(importance_per_expert**2)
        / (torch.sum(importance_per_expert) ** 2)
    )

    return (load_loss + imp_loss) / 2.0
