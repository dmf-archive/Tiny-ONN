import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Any, Optional

class RoutingShaper:
    def __init__(self, w_meta: float = 0.5, cost_alpha: float = 0.5):
        self.w_meta = w_meta
        self.cost_alpha = cost_alpha
        self.performance_stats = {}

    def calculate_meta_loss(
        self,
        routing_info: List[Dict[str, torch.Tensor]],
        model: nn.Module,
        optimizer: Any
    ) -> torch.Tensor:
        """
        FARS: Fisher-Aware Routing Shaping (Module-level).
        Uses optimizer's second moment (v_t) as a proxy for Fisher Information.
        """
        start_time = time.perf_counter()
        meta_losses = []
        
        # Pre-calculate expert-level Fisher costs
        module_costs = {}
        for name, module in model.named_modules():
            if any(target in name for target in ["experts", "expert_library"]):
                # Check if it's a VectorizedExpertMLP which has (num_experts, ...) parameters
                expert_costs = None
                for p_name, p in module.named_parameters():
                    state = optimizer.state.get(p)
                    if state and 'exp_avg_sq' in state:
                        # Cost_FARS ≈ ||sqrt(v_t)|| per expert
                        # exp_avg_sq shape: [num_experts, d_in, d_out] or [num_experts, d_out, d_in]
                        v_t_sqrt = state['exp_avg_sq'].sqrt()
                        # Average over all dims except the first (expert dim)
                        reduce_dims = list(range(1, v_t_sqrt.ndim))
                        p_expert_costs = v_t_sqrt.mean(dim=reduce_dims)
                        
                        if expert_costs is None:
                            expert_costs = p_expert_costs
                        else:
                            expert_costs = expert_costs + p_expert_costs
                
                if expert_costs is not None:
                    module_costs[name] = expert_costs
        
        # Debug: Verify mapping once
        if not hasattr(self, "_mapping_verified"):
            if module_costs:
                print(f"\n[FARS Debug] Found {len(module_costs)} expert modules with per-expert costs.")
                self._mapping_verified = True

        for i, layer_info in enumerate(routing_info):
            for key, logits in layer_info.items():
                if "logits" in key:
                    # 1. SARS: Surprise-Aware (Entropy of the prior)
                    probs = F.softmax(logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
                    
                    # 2. FARS: Fisher-Aware (Expert-level cost)
                    prefix = f"layers.{i}."
                    if "q_logits" in key: module_name = prefix + "attn.q_experts"
                    elif "k_logits" in key: module_name = prefix + "attn.k_experts"
                    elif "v_logits" in key: module_name = prefix + "attn.v_experts"
                    elif "mlp_logits" in key: module_name = prefix + "mlp.experts"
                    else: module_name = None
                    
                    if module_name and module_name not in module_costs:
                        for m_name in module_costs:
                            if m_name.endswith(module_name.split('.')[-1]) and prefix in m_name:
                                module_name = m_name
                                break
                    
                    expert_costs = module_costs.get(module_name)
                    if expert_costs is not None:
                        # expert_costs shape: [num_experts]
                        # probs shape: [B, T, num_heads, num_experts] or [B, T, num_experts]
                        # We want to penalize high-cost experts weighted by their selection probability
                        cost_fars = (probs * expert_costs).sum(dim=-1).mean()
                    else:
                        cost_fars = 0.0
                    
                    meta_losses.append(entropy + self.cost_alpha * cost_fars)
        
        if not meta_losses:
            res = torch.tensor(0.0, device=next(model.parameters()).device)
        else:
            res = torch.stack(meta_losses).mean() * self.w_meta
            
        self.performance_stats['shaper_calc_time_ms'] = (time.perf_counter() - start_time) * 1000
        return res

    def get_routing_diagnostics(self, routing_info: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        diagnostics = {}
        for i, layer_info in enumerate(routing_info):
            for key, logits in layer_info.items():
                if "logits" in key:
                    probs = F.softmax(logits, dim=-1)
                    max_prob = probs.max(dim=-1).values.mean().item()
                    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
                    diagnostics[f"L{i}_{key}_conf"] = max_prob
                    diagnostics[f"L{i}_{key}_ent"] = entropy
        return diagnostics
