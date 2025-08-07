from typing import Any

import torch
import torch.nn.functional as F


def get_nsa_auxiliary_loss(
    normalized_entropy: torch.Tensor,
    k_ratio: torch.Tensor,
    w_entropy: float = 1.0,
    w_sparse: float = 1.0
) -> torch.Tensor:
    entropy_loss = F.mse_loss(k_ratio, normalized_entropy.detach())
    sparsity_loss = k_ratio.mean()
    return w_entropy * entropy_loss + w_sparse * sparsity_loss

def compute_all_auxiliary_losses(
    aux_outputs: list[dict[str, Any]],
    main_loss: torch.Tensor,
    config: Any
) -> tuple[torch.Tensor, dict[str, Any], dict[int, torch.Tensor]]:
    total_aux_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
    layer_stats = {}
    surprise_matrices = {}

    all_expert_outputs = [
        layer["moe"]["full_expert_outputs"] for layer in aux_outputs
    ]

    grad_matrices = torch.autograd.grad(
        outputs=main_loss,
        inputs=all_expert_outputs,
        retain_graph=True,
        allow_unused=True
    )

    for layer_idx, layer_aux in enumerate(aux_outputs):
        attn_aux = layer_aux["attn"]
        moe_aux = layer_aux["moe"]

        nsa_loss = get_nsa_auxiliary_loss(
            attn_aux["normalized_entropy"],
            attn_aux["k_ratio"],
            config.w_entropy,
            config.w_sparse
        )
        total_aux_loss += nsa_loss

        grad_matrix = grad_matrices[layer_idx]

        if grad_matrix is not None:
            surprise_matrix = grad_matrix.float().mean(dim=-1)
            pre_act_logits = moe_aux["pre_act_logits"]

            with torch.no_grad():
                target_indices = torch.argmin(surprise_matrix, dim=-1)
            ce_loss = F.cross_entropy(pre_act_logits, target_indices)

            log_target_dist = F.log_softmax(-surprise_matrix, dim=-1)
            log_gate_dist = F.log_softmax(pre_act_logits, dim=-1)
            kl_loss = F.kl_div(log_gate_dist, log_target_dist, reduction='batchmean', log_target=True)

            moe_gating_loss = config.w_ce * ce_loss + config.w_kl * kl_loss
        else:
            moe_gating_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
            surprise_matrix = torch.zeros_like(moe_aux["pre_act_logits"])

        total_aux_loss += moe_gating_loss
        surprise_matrices[layer_idx] = surprise_matrix.detach()

        layer_stats[f"layer_{layer_idx}"] = {
            "nsa_loss": nsa_loss.item(),
            "moe_gating_loss": moe_gating_loss.item(),
            "avg_k": attn_aux["avg_k"].item(),
            "avg_surprise": surprise_matrix.mean().item(),
            "activation_rate": (moe_aux["activation_mask"].sum() / moe_aux["activation_mask"].numel()).item()
        }

    return total_aux_loss, layer_stats, surprise_matrices

def apply_expert_gradient_filtering(experts: torch.nn.ModuleList, threshold_sigma: float = 2.0) -> list[int]:
    clipped_counts_per_expert = []
    for expert in experts:
        all_grads_list = []
        for param in expert.parameters():
            if param.grad is not None:
                all_grads_list.append(param.grad.view(-1))

        if not all_grads_list:
            clipped_counts_per_expert.append(0)
            continue

        all_grads_tensor = torch.cat(all_grads_list)
        mean = all_grads_tensor.mean()
        std = all_grads_tensor.std()

        upper_threshold = mean + threshold_sigma * std
        lower_threshold = mean - threshold_sigma * std

        num_clipped_in_expert = 0
        for param in expert.parameters():
            if param.grad is not None:
                clipped_mask = (param.grad.data > upper_threshold) | (param.grad.data < lower_threshold)
                num_clipped_in_expert += int(clipped_mask.sum().item())
                param.grad.data[clipped_mask] = 0.0
        clipped_counts_per_expert.append(num_clipped_in_expert)

    return clipped_counts_per_expert
