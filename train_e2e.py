import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

from tiny_onn.modular import TinyOnnForCausalLM
from training.config import load_config
from training.losses import (
    apply_expert_gradient_filtering,
    compute_all_auxiliary_losses,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/E2E_test_train.yaml")
    args = parser.parse_args()

    full_config = load_config(Path(args.config))
    model_config = full_config.model
    train_config = full_config.train
    observer_config = full_config.observer

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.bfloat16

    model = TinyOnnForCausalLM(model_config).to(DEVICE, dtype=DTYPE)
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", cache_dir="./weights", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model_config.vocab_size = len(tokenizer)
    model.lm_head = torch.nn.Linear(model_config.hidden_size, model_config.vocab_size, bias=False).to(DEVICE, dtype=DTYPE)
    model.model.embed_tokens = torch.nn.Embedding(model_config.vocab_size, model_config.hidden_size, model.model.padding_idx).to(DEVICE, dtype=DTYPE)

    with open(full_config.data.train_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    global_step = 0

    for epoch in range(train_config.num_epochs):
        for item in data:
            start_time = time.time()
            optimizer.zero_grad()

            prompt = f"user: {item['messages'][0]['content']}\nassistant: "
            full_text = f"{prompt}{item['messages'][1]['content']}{tokenizer.eos_token}"
            inputs = tokenizer(full_text, return_tensors="pt", max_length=model_config.max_position_embeddings, truncation=True, padding="max_length")

            input_ids = inputs.input_ids.to(DEVICE)
            labels = input_ids.clone()
            labels[:, :len(tokenizer.encode(prompt))] = -100
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(input_ids=input_ids)

            main_loss = F.cross_entropy(outputs.logits.view(-1, model_config.vocab_size), labels.view(-1), ignore_index=-100)

            if torch.isnan(main_loss):
                continue

            aux_loss, layer_stats, surprise_matrices = compute_all_auxiliary_losses(outputs.aux_outputs, main_loss, model_config)

            total_loss = main_loss + aux_loss
            total_loss.backward()

            surprise_pi = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    surprise_pi += param_norm.item() ** 2
            surprise_pi = surprise_pi ** 0.5

            clipped_grads_by_layer = []
            for layer in model.model.layers:
                clipped_grads_by_layer.append(apply_expert_gradient_filtering(layer.mlp.experts, train_config.surprise_threshold_sigma))

            optimizer.step()

            if global_step % train_config.log_interval == 0:
                with torch.no_grad():
                    main_acc = (outputs.logits.argmax(-1) == labels).float().mean().item()

                    nsa_loss = sum(stats['nsa_loss'] for stats in layer_stats.values())
                    moe_loss = sum(stats['moe_gating_loss'] for stats in layer_stats.values())

                    avg_k_nsa = sum(stats['avg_k'] for stats in layer_stats.values()) / len(layer_stats)
                    avg_act_rate = sum(stats['activation_rate'] for stats in layer_stats.values()) / len(layer_stats)

                    tau = torch.distributions.Categorical(logits=outputs.logits.detach()).entropy().mean().item()
                    cost = (1 - observer_config.pi_gamma) * (main_loss.item() / (tau + 1e-9)) + observer_config.pi_gamma * surprise_pi
                    pi_score = torch.exp(torch.tensor(-observer_config.pi_alpha * cost)).item()

                    gating_acc = 0
                    gating_kld = 0
                    for i, layer_aux in enumerate(outputs.aux_outputs):
                        pre_act_logits = layer_aux["moe"]["pre_act_logits"]
                        surprise_matrix = surprise_matrices[i]
                        gating_acc += (pre_act_logits.argmax(dim=-1) == surprise_matrix.argmin(dim=-1)).float().mean().item()

                        log_target_dist = F.log_softmax(-surprise_matrix, dim=-1)
                        log_gate_dist = F.log_softmax(pre_act_logits, dim=-1)
                        gating_kld += F.kl_div(log_gate_dist, log_target_dist, reduction='batchmean', log_target=True).item()

                    gating_acc /= len(outputs.aux_outputs)
                    gating_kld /= len(outputs.aux_outputs)

                    surprise_stats = ""
                    for i, sm in surprise_matrices.items():
                        mean, std = sm.mean().item(), sm.std().item()
                        surprise_stats += f"L{i}: {mean:.2e}±{std:.2e} | "

                    step_time = time.time() - start_time

                    stabilizer_stats = ""
                    for i, layer_clips in enumerate(clipped_grads_by_layer):
                        num_affected_experts = sum(1 for count in layer_clips if count > 0)
                        total_clips = sum(layer_clips)
                        avg_clips_per_expert = total_clips / len(layer_clips) if len(layer_clips) > 0 else 0
                        stabilizer_stats += f"L{i}: {num_affected_experts}/{len(layer_clips)} affected, {avg_clips_per_expert:.1f} avg_clips/expert | "

                    print("-" * 80)
                    print(f"Step: {global_step} | Epoch: {epoch+1} | Time: {step_time:.2f}s")
                    print(f"  Main   | Loss: {main_loss.item():.3f} | Acc: {main_acc:.3f}")
                    print(f"  DynNSA | Loss: {nsa_loss:.3f} | Avg K: {avg_k_nsa:.2f}")
                    print(f"  DynMoE | Loss: {moe_loss:.3f} | Gate Acc: {gating_acc:.3f} | KLD: {gating_kld:.3f} | Avg K: {avg_act_rate*model_config.num_experts_per_layer:.2f}")
                    print(f"  PI     | Score: {pi_score:.3f} | Tau: {tau:.3f} | Global Surprise: {surprise_pi:.3f}")
                    print(f"  Stabilizer | {stabilizer_stats}")
                    print(f"  Surprise Stats | {surprise_stats}")
                    print("-" * 80)

            global_step += 1

if __name__ == "__main__":
    main()
