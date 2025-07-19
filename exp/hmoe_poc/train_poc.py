from collections import defaultdict

import torch
import torch.nn as nn

from tiny_onn.hmoe_poc.model import PocHMoEModel


class GradientInterceptor:
    def __init__(self):
        self.surprises = defaultdict(list)
        self.handles = []

    def forward_hook(self, module, input_tuple, output):
        if torch.is_grad_enabled():
            module._saved_input = input_tuple[0].detach()

    def backward_hook(self, module, grad_input, grad_output):
        if not hasattr(module, "_saved_input"):
            return

        input_tensor = module._saved_input
        grad_output_tensor = grad_output[0]

        per_token_grad = torch.einsum("bi,bo->boi", input_tensor, grad_output_tensor)
        surprise = torch.linalg.vector_norm(per_token_grad, dim=(1, 2))

        # This mapping is simplified for PoC. In a real scenario,
        # a more robust way to identify the expert globally would be needed.
        group_id = self.expert_to_group_map[module]
        expert_id_in_group = self.expert_to_local_id_map[module]
        global_expert_name = f"Group_{group_id}-Expert_{expert_id_in_group}"

        self.surprises[global_expert_name].append(surprise.cpu().numpy())

        del module._saved_input

    def attach(self, model):
        self.expert_to_group_map = {}
        self.expert_to_local_id_map = {}
        for i, group in enumerate(model.hmoe.groups):
            for j, expert in enumerate(group.experts):
                self.expert_to_group_map[expert] = i
                self.expert_to_local_id_map[expert] = j
                handle = expert.register_full_backward_hook(self.backward_hook)
                self.handles.append(handle)
                expert.register_forward_hook(self.forward_hook)

    def clear(self):
        self.surprises.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def run_hmoe_poc():
    vocab_size = 100
    d_model = 32
    n_groups = 2
    n_experts_per_group = 2
    top_k_l1 = 1
    top_k_l2 = 1
    batch_size = 6
    seq_len = 8

    model = PocHMoEModel(
        vocab_size, d_model, n_groups, n_experts_per_group, top_k_l1, top_k_l2
    )
    interceptor = GradientInterceptor()
    interceptor.attach(model)

    input_data = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_data = torch.randint(0, vocab_size, (batch_size, seq_len))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("--- Running Hierarchical MoE Proof-of-Concept ---")

    optimizer.zero_grad()
    output_logits = model(input_data)
    loss = loss_fn(output_logits.view(-1, vocab_size), target_data.view(-1))
    loss.backward()
    optimizer.step()

    print("\n--- HMoE Gradient Interception Results ---")
    if not interceptor.surprises:
        print("No gradients were intercepted.")
    else:
        for expert_name, surprises_list in sorted(interceptor.surprises.items()):
            print(f"\n{expert_name}:")
            for i, surprise_batch in enumerate(surprises_list):
                print(f"  - Surprise values (batch size: {len(surprise_batch)}):")
                for token_idx, value in enumerate(surprise_batch):
                    print(f"    Token {token_idx}: {value:.4f}")

    interceptor.clear()
    print("\n--- HMoE PoC Finished ---")


if __name__ == "__main__":
    run_hmoe_poc()
