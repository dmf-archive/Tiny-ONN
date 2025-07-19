from collections import defaultdict

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from exp.integration_poc.model import SimpleMoE


class GradientInterceptor:
    def __init__(self):
        self.surprises = defaultdict(list)
        self.handles = []
        self.expert_map = {}

    def forward_hook(self, module, input_tuple, output):
        if torch.is_grad_enabled():
            module._saved_input = input_tuple[0].detach()

    def backward_hook(self, module, grad_input, grad_output):
        if not hasattr(module, "_saved_input"):
            return

        input_tensor = module._saved_input
        grad_output_tensor = grad_output[0]

        # This einsum is for nn.Linear's weight gradient
        # w2 layer in Expert: (intermediate, hidden) -> grad: (hidden, intermediate)
        # input: (tokens, intermediate), grad_output: (tokens, hidden)
        # grad = grad_output.T @ input -> (hidden, intermediate)
        # per-token grad = einsum('ti,th->tih', input, grad_output)

        # We will intercept the gradient of w2 layer in the Expert MLP
        w2_grad = torch.einsum("ti,th->tih", input_tensor, grad_output_tensor)
        surprise = torch.linalg.vector_norm(w2_grad, dim=(1, 2))

        expert_name = self.expert_map.get(module, "UnknownExpert")
        self.surprises[expert_name].append(surprise.cpu().numpy())
        del module._saved_input

    def attach(self, moe_module: SimpleMoE):
        for i, expert in enumerate(moe_module.experts):
            # Attach hook to the last linear layer (w2) of the expert
            target_module = expert.w2
            self.expert_map[target_module] = f"Expert_{i}"
            handle = target_module.register_full_backward_hook(self.backward_hook)
            self.handles.append(handle)
            # The forward hook needs to be on the module that receives the grad_output
            # which is the expert itself. But we need the input to w2.
            # A simpler way for this PoC is to attach forward hook to w2 as well.
            target_module.register_forward_hook(self.forward_hook)

    def clear(self):
        self.surprises.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.expert_map.clear()


def run_integration_poc():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    config = AutoConfig.from_pretrained(model_name)

    # Override config for a minimal PoC model
    config.hidden_size = 64
    config.intermediate_size = 256
    config.num_hidden_layers = 4
    config.num_attention_heads = 4
    config.num_key_value_heads = 4

    print("--- Initializing a minimal model from modified config ---")
    model = AutoModelForCausalLM.from_config(config).to(torch.float32)

    print("\n--- Original Model Architecture (Layer 0 MLP) ---")
    print(model.model.layers[0].mlp)

    print("\n--- Performing MLP -> SimpleMoE Swap ---")
    moe_layer = SimpleMoE(config, n_experts=4, top_k=2)
    model.model.layers[0].mlp = moe_layer

    print("\n--- Modified Model Architecture (Layer 0 MLP) ---")
    print(model.model.layers[0].mlp)

    interceptor = GradientInterceptor()
    interceptor.attach(model.model.layers[0].mlp)

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print("\n--- Running single forward/backward pass ---")

    # No need for optimizer or loss function for this verification
    output = model(input_ids).logits
    # Create a dummy loss for backward pass
    loss = output.sum()
    loss.backward()

    print("\n--- Integration PoC Gradient Interception Results ---")
    if not interceptor.surprises:
        print("No gradients were intercepted. Check model and hook logic.")
    else:
        for expert_name, surprises_list in sorted(interceptor.surprises.items()):
            print(f"\n{expert_name}:")
            for i, surprise_batch in enumerate(surprises_list):
                print(f"  - Surprise values (batch size: {len(surprise_batch)}):")
                for token_idx, value in enumerate(surprise_batch):
                    print(f"    Token {token_idx}: {value:.4f}")

    interceptor.clear()
    print("\n--- Integration PoC Finished ---")


if __name__ == "__main__":
    run_integration_poc()
