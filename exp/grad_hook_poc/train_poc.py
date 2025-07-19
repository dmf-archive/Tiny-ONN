from collections import defaultdict

import torch
import torch.nn as nn

from tiny_onn.grad_hook.model import PocMoEModel


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

        expert_id = self.handles.index(module.handle)
        self.surprises[expert_id].append(surprise.cpu().numpy())

        del module._saved_input

    def attach(self, module, expert_id):
        handle = module.register_full_backward_hook(self.backward_hook)
        module.handle = handle
        self.handles.append(handle)
        module.register_forward_hook(self.forward_hook)

    def clear(self):
        self.surprises.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def run_poc():
    vocab_size = 100
    d_model = 32
    n_experts = 4
    top_k = 2
    batch_size = 5
    seq_len = 10

    model = PocMoEModel(vocab_size, d_model, n_experts, top_k)
    interceptor = GradientInterceptor()

    for i, expert in enumerate(model.moe.experts):
        interceptor.attach(expert, i)

    input_data = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_data = torch.randint(0, vocab_size, (batch_size, seq_len))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("--- Running Proof-of-Concept ---")

    optimizer.zero_grad()

    output_logits = model(input_data)

    loss = loss_fn(output_logits.view(-1, vocab_size), target_data.view(-1))

    loss.backward()

    optimizer.step()

    print("\n--- Gradient Interception Results ---")
    if not interceptor.surprises:
        print("No gradients were intercepted. Check model and hook logic.")
    else:
        for expert_id, surprises_list in sorted(interceptor.surprises.items()):
            print(f"\nExpert {expert_id}:")
            for i, surprise_batch in enumerate(surprises_list):
                print(
                    f"  - Surprise values from backward pass {i + 1} (batch size: {len(surprise_batch)}):"
                )
                for token_idx, value in enumerate(surprise_batch):
                    print(f"    Token {token_idx}: {value:.4f}")

    interceptor.clear()
    print("\n--- PoC Finished ---")


if __name__ == "__main__":
    run_poc()
