from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

@dataclass
class MoELayerMetrics:
    activations: torch.Tensor       # Dense tensor [N, E]
    surprises: torch.Tensor         # Sparse COO tensor [N, E]

class TinyExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))

class MoELayer(nn.Module):
    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([TinyExpert(hidden_size, intermediate_size) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, C)
        num_tokens = flat_hidden_states.shape[0]

        router_logits = self.gate(flat_hidden_states)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float).to(DTYPE)

        full_expert_outputs = torch.zeros(num_tokens, self.num_experts, C, device=DEVICE, dtype=DTYPE)

        for i in range(num_tokens):
            for k in range(self.top_k):
                expert_idx = selected_experts[i, k].item()
                expert_input = flat_hidden_states[i:i+1]
                full_expert_outputs[i, expert_idx, :] = self.experts[expert_idx](expert_input)

        selected_expert_outputs = torch.gather(full_expert_outputs, 1, selected_experts.unsqueeze(-1).expand(-1, -1, C))
        final_output = torch.bmm(routing_weights.unsqueeze(1), selected_expert_outputs).squeeze(1)

        return final_output.view(B, T, C), full_expert_outputs, router_logits, selected_experts

class DoubleMoEModel(nn.Module):
    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int, top_k: int):
        super().__init__()
        self.moe1 = MoELayer(num_experts, hidden_size, intermediate_size, top_k)
        self.ln = nn.LayerNorm(hidden_size)
        self.moe2 = MoELayer(num_experts, hidden_size, intermediate_size, top_k)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        h1, out1, logits1, experts1 = self.moe1(hidden_states)
        h_res = hidden_states + h1

        h2, out2, logits2, experts2 = self.moe2(self.ln(h_res))

        return h_res + h2, [out1, out2], [logits1, logits2], [experts1, experts2]

def get_moe_metrics(loss: torch.Tensor, full_expert_outputs_list: list[torch.Tensor], router_logits_list: list[torch.Tensor]) -> list[MoELayerMetrics]:
    grad_matrices = torch.autograd.grad(loss, full_expert_outputs_list, allow_unused=False)

    metrics_list = []
    for grads, logits in zip(grad_matrices, router_logits_list, strict=False):
        surprise_matrix_dense = torch.linalg.norm(grads.float(), dim=-1)
        activation_matrix = torch.softmax(logits, dim=-1)

        sparse_indices = torch.nonzero(surprise_matrix_dense).t()
        sparse_values = surprise_matrix_dense[sparse_indices[0], sparse_indices[1]]

        surprise_sparse_coo = torch.sparse_coo_tensor(
            sparse_indices, sparse_values, surprise_matrix_dense.size()
        )

        metrics_list.append(MoELayerMetrics(
            activations=activation_matrix,
            surprises=surprise_sparse_coo.coalesce()
        ))

    return metrics_list

def main():
    print("--- Final PoC: Verifying Sparse Tensor Metrics Structure ---")

    NUM_EXPERTS = 8; HIDDEN_SIZE = 16; INTERMEDIATE_SIZE = 32; BATCH_SIZE = 4; SEQ_LEN = 10; TOP_K = 2
    TOTAL_TOKENS = BATCH_SIZE * SEQ_LEN

    model = DoubleMoEModel(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, TOP_K).to(DEVICE, dtype=DTYPE)
    hidden_states = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    labels = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)

    final_output, full_expert_outputs_list, router_logits_list, selected_experts_list = model(hidden_states)
    loss = F.mse_loss(final_output, labels)

    print("\n1. Capturing MoE Metrics...")
    moe_metrics_list = get_moe_metrics(loss, full_expert_outputs_list, router_logits_list)
    print("   Done.")

    print("\n2. Verifying all metrics...")
    for i, (metrics, selected_experts) in enumerate(zip(moe_metrics_list, selected_experts_list, strict=False)):
        print(f"\n--- Verifying Layer {i+1} ---")

        # Check shapes
        print(f"   Activations shape: {metrics.activations.shape} (Expected: [{TOTAL_TOKENS}, {NUM_EXPERTS}])")
        assert metrics.activations.shape == (TOTAL_TOKENS, NUM_EXPERTS)
        print(f"   Sparse Surprises shape: {metrics.surprises.shape} (Expected: [{TOTAL_TOKENS}, {NUM_EXPERTS}])")
        assert metrics.surprises.shape == (TOTAL_TOKENS, NUM_EXPERTS)

        # Verify correspondence
        surprise_indices = metrics.surprises.indices()
        num_non_zero = surprise_indices.shape[1]

        print(f"   Non-zero surprise values: {num_non_zero} (Expected: {TOTAL_TOKENS * TOP_K})")
        assert num_non_zero == TOTAL_TOKENS * TOP_K

        # Check a random token's indices
        random_token_idx = torch.randint(0, TOTAL_TOKENS, (1,)).item()
        expected_experts = selected_experts[random_token_idx].sort().values

        actual_experts_mask = surprise_indices[0, :] == random_token_idx
        actual_experts = surprise_indices[1, actual_experts_mask].sort().values

        correspondence_ok = torch.equal(expected_experts, actual_experts)
        print(f"   Token {random_token_idx} correspondence check: {'OK' if correspondence_ok else 'FAIL'}")

        if correspondence_ok:
            print(f"   ✅ SUCCESS: Layer {i+1} sparse metrics are correct and verifiable.")
        else:
            print(f"   ❌ FAILURE: Layer {i+1} has a mismatch in sparse indices.")

if __name__ == "__main__":
    main()
