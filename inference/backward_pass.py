import torch


def run_backward_pass(model, full_sequence_ids):
    """
    Runs a single backward pass for the full sequence to capture gradients.
    Returns a dictionary of parameter gradients.
    """
    final_gradients = {}
    model.train()
    model.zero_grad()
    
    outputs = model(input_ids=full_sequence_ids, attention_mask=torch.ones_like(full_sequence_ids), labels=full_sequence_ids)
    loss = outputs.loss
    if loss is not None and loss.requires_grad:
        loss.backward()
        print("Backward pass completed.")

    for name, param in model.named_parameters():
        if param.grad is not None:
            final_gradients[name] = param.grad.norm().item()
    
    model.zero_grad()
    print(f"Collected {len(final_gradients)} final gradients.")
    return final_gradients
