import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Any, Optional

class RoutingShaper:
    def __init__(self, w_meta: float = 0.1, cost_alpha: float = 0.1):
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
        
        # Pre-calculate module-level Fisher costs
        module_costs = {}
        for name, module in model.named_modules():
            # Target expert modules
            if any(target in name for target in ["experts", "expert_library"]):
                v_t_norm_sum = 0.0
                p_count = 0
                for p in module.parameters():
                    state = optimizer.state.get(p)
                    if state and 'exp_avg_sq' in state:
                        # Cost_FARS ≈ ||sqrt(v_t)||
                        v_t_norm_sum += state['exp_avg_sq'].sqrt().mean().item()
                        p_count += 1
                if p_count > 0:
                    module_costs[name] = v_t_norm_sum / p_count

        for i, layer_info in enumerate(routing_info):
            for key, logits in layer_info.items():
                if "logits" in key:
                    # 1. SARS: Surprise-Aware (Entropy of the prior)
                    probs = F.softmax(logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
                    
                    # 2. FARS: Fisher-Aware (Module-level cost)
                    # Map routing key to module name
                    prefix = f"layers.{i}."
                    if "q_logits" in key: module_name = prefix + "attn.q_experts"
                    elif "k_logits" in key: module_name = prefix + "attn.k_experts"
                    elif "v_logits" in key: module_name = prefix + "attn.v_experts"
                    elif "mlp_logits" in key: module_name = prefix + "mlp.experts"
                    else: module_name = None
                    
                    cost_fars = module_costs.get(module_name, 0.0) if module_name else 0.0
                    
                    # Shaping: Penalize experts with high Fisher cost (high curvature/uncertainty)
                    # This encourages routing to "stable" experts.
                    # We use a simple linear combination for now.
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
