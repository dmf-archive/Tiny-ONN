import json
import random
import time

import torch
import torch.nn.functional as F
from einops import rearrange
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.distributions import Categorical
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from .config import DEVICE, UnifiedConfig
from .model import TinyOnnModel


class ChatDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int, data: list):
        self.tokenizer = tokenizer
        self.max_length = max_length

        prompts = [f"user: {next((msg['content'] for msg in item['messages'] if msg['role'] == 'user'), '')}\nassistant: " for item in data]
        responses = [f"{next((msg['content'] for msg in item['messages'] if msg['role'] == 'assistant'), '')}{tokenizer.eos_token or ''}" for item in data]
        full_texts = [p + r for p, r in zip(prompts, responses, strict=False)]

        self.inputs = self.tokenizer(full_texts, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        prompt_token_lengths = [len(self.tokenizer(p, add_special_tokens=False).input_ids) for p in prompts]

        self.labels = self.inputs.input_ids.clone()
        for i, prompt_len in enumerate(prompt_token_lengths):
            self.labels[i, :prompt_len] = -100

        if self.tokenizer.pad_token_id is not None:
            self.labels[self.labels == self.tokenizer.pad_token_id] = -100

    def __len__(self) -> int:
        return len(self.inputs.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs.input_ids[idx], self.labels[idx]


def generate_text(model, tokenizer, prompt, max_new_tokens=30):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        prompt_len = input_ids.shape[1]

        for _ in range(max_new_tokens):
            outputs, _ = model(input_ids)
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            if tokenizer.eos_token_id is not None and next_token_id.item() == tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    model.train()
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)


def main():
    config = UnifiedConfig()

    with open("data/dummy_chat_data.jsonl", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    get_training_corpus = lambda: (f"user: {next((msg['content'] for msg in item['messages'] if msg['role'] == 'user'), '')}\nassistant: {next((msg['content'] for msg in item['messages'] if msg['role'] == 'assistant'), '')}" for item in data)

    base_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    base_tokenizer.pre_tokenizer = Whitespace()
    base_tokenizer.train_from_iterator(get_training_corpus(), BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[EOS]"], vocab_size=config.vocab_size))

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=base_tokenizer, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
    config.vocab_size = base_tokenizer.get_vocab_size()

    model = TinyOnnModel(config).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    dataset = ChatDataset(tokenizer, config.max_seq_len, data)

    global_step = 0
    start_time = time.time()
    for epoch in range(config.epochs):
        print(f"\n--- Starting Epoch {epoch+1}/{config.epochs} ---")

        expert_activation_counts: dict[tuple[int, int, int], int] = {}
        expert_true_activation_counts: dict[tuple[int, int, int], int] = {}

        for layer_idx in range(config.num_hidden_layers):
            for expert_type in [1, 2]:
                num_experts = config.max_attention_experts if expert_type == 1 else config.max_moe_experts
                for expert_sub_idx in range(num_experts):
                    expert_id = (expert_type, expert_sub_idx, layer_idx)
                    expert_activation_counts[expert_id] = 0
                    expert_true_activation_counts[expert_id] = 0


        for i in range(len(dataset)):
            step_start_time = time.time()
            input_ids, labels = dataset[i]
            input_ids, labels = input_ids.unsqueeze(0).to(DEVICE), labels.unsqueeze(0).to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            final_logits, forward_cache = model(input_ids)

            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            main_loss = F.cross_entropy(rearrange(shift_logits, 'b t d -> (b t) d'), rearrange(shift_labels, 'b t -> (b t)'), ignore_index=-100)
            if torch.isnan(main_loss) or main_loss.isinf(): continue

            total_gating_loss = torch.tensor(0.0, device=DEVICE)
            metrics = { "smha_surprise": [], "moe_surprise": [], "smha_gate_acc": [], "moe_gate_acc": [], "smha_avg_k": [], "moe_avg_k": [] }

            outputs_to_grad = [cache['final_output'] for cache in forward_cache.values()]
            if outputs_to_grad:
                grads_for_surprise = torch.autograd.grad(main_loss, outputs_to_grad, create_graph=False, retain_graph=True)

                grad_idx = 0
                for expert_id, layer_cache in forward_cache.items():
                    expert_type, _, layer_index = expert_id
                    if expert_type == 1:
                        layer_type_str ="smha"
                    elif expert_type == 2:
                        layer_type_str = "moe"
                    else:
                        continue

                    B, T = layer_cache['B'], layer_cache['T']
                    grad_norm_shape = (B * T, -1)
                    grad_norm = torch.linalg.norm(grads_for_surprise[grad_idx].view(grad_norm_shape), dim=-1)

                    routing_weights = layer_cache['routing_weights']
                    surprise_matrix_shape = (B * T, -1)
                    surprise_matrix = grad_norm.unsqueeze(-1) * routing_weights.view(surprise_matrix_shape)

                    logits = layer_cache['gate_cache']['logits'].view(surprise_matrix_shape)

                    with torch.no_grad():
                        targets = torch.argmin(surprise_matrix, dim=-1).detach()
                        acc = (logits.argmax(dim=-1) == targets).float().mean()

                    log_targ = F.log_softmax(-surprise_matrix.detach(), dim=-1)
                    log_gate = F.log_softmax(logits, dim=-1)

                    if expert_type == 1:
                        w_ce, w_kl, w_aux = config.w_ce_smha, config.w_kl_smha, config.w_aux_smha
                    elif expert_type == 2:
                        w_ce, w_kl, w_aux = config.w_ce_moe, config.w_kl_moe, config.w_aux_moe
                    else:
                        continue
                    
                    gating_loss = w_ce * F.cross_entropy(logits, targets.view(-1)) + w_kl * F.kl_div(log_gate, log_targ.detach(), reduction='batchmean', log_target=True)
                    total_gating_loss += w_aux * gating_loss
                    
                    activation_mask_flat = layer_cache['gate_cache']['activation_mask'].view(-1, logits.size(-1)) > 0
                    active_surprise = surprise_matrix.view(-1, logits.size(-1))[activation_mask_flat]
                    metrics[f"{layer_type_str}_surprise"].append(active_surprise.mean().item() if active_surprise.numel() > 0 else 0)
                    metrics[f"{layer_type_str}_gate_acc"].append(acc.item())
                    metrics[f"{layer_type_str}_avg_k"].append(layer_cache["num_active_tokens"] / (layer_cache['B'] * layer_cache['T']))
                    
                    # Update expert activation counts
                    activation_mask_flat = layer_cache['gate_cache']['activation_mask'].view(-1, logits.size(-1))
                    active_experts_in_batch = torch.where(activation_mask_flat.sum(dim=0) > 0)[0].tolist()
                    for expert_sub_idx in active_experts_in_batch:
                        expert_activation_counts[(expert_type, expert_sub_idx, layer_index)] += 1
                    
                    gated_logits_flat = layer_cache['gate_cache']['gated_logits'].view(-1, logits.size(-1))
                    true_active_experts_in_batch = torch.where((gated_logits_flat > 0).sum(dim=0) > 0)[0].tolist()
                    for expert_sub_idx in true_active_experts_in_batch:
                        expert_true_activation_counts[(expert_type, expert_sub_idx, layer_index)] += 1

                    grad_idx += 1

            total_loss = main_loss + total_gating_loss
            if not (torch.isnan(total_loss) or total_loss.isinf()):
                total_loss.backward()

            optimizer.step()

            global_step += 1
            if global_step % 20 == 0:
                with torch.no_grad():
                    mask = (shift_labels != -100)
                    main_acc = ((shift_logits.argmax(-1) == shift_labels) & mask).float().sum() / mask.sum() if mask.sum() > 0 else 0.0

                    tau = Categorical(logits=final_logits).entropy().mean().item()
                    all_surprises = metrics["smha_surprise"] + metrics["moe_surprise"]
                    surprise = sum(all_surprises) / len(all_surprises) if all_surprises else 0
                    pi_score = torch.exp(-torch.tensor(config.pi_alpha * ((1 - config.pi_gamma) * (main_loss.item() / (tau + 1e-6)) + config.pi_gamma * surprise))).item()

                    num_smha_layers = config.num_hidden_layers
                    num_moe_layers = config.num_hidden_layers
                    
                    avg_smha_loss = total_gating_loss.item() / num_smha_layers if num_smha_layers > 0 and metrics["smha_surprise"] else 0
                    avg_moe_loss = total_gating_loss.item() / num_moe_layers if num_moe_layers > 0 and metrics["moe_surprise"] else 0
                    
                    avg_smha_acc = sum(metrics["smha_gate_acc"]) / len(metrics["smha_gate_acc"]) if metrics["smha_gate_acc"] else 0
                    avg_moe_acc = sum(metrics["moe_gate_acc"]) / len(metrics["moe_gate_acc"]) if metrics["moe_gate_acc"] else 0

                    avg_smha_k = sum(metrics["smha_avg_k"]) / len(metrics["smha_avg_k"]) if metrics["smha_avg_k"] else 0
                    avg_moe_k = sum(metrics["moe_avg_k"]) / len(metrics["moe_avg_k"]) if metrics["moe_avg_k"] else 0

                    it_per_sec = 1.0 / (time.time() - step_start_time)
                    log_str = (
                        f"  Step {global_step}: PI: {pi_score:.2f} | "
                        f"Loss(M/S/M): {main_loss.item():.2f}/{avg_smha_loss:.2f}/{avg_moe_loss:.2f} | "
                        f"Acc(M/S/M): {main_acc.item():.2f}/{avg_smha_acc:.2f}/{avg_moe_acc:.2f} | "
                        f"K(S/M): {avg_smha_k:.2f}/{avg_moe_k:.2f} | "
                        f"Speed: {it_per_sec:.2f} it/s"
                    )
                    print(log_str)

        dead_experts_count = sum(1 for count in expert_activation_counts.values() if count == 0)
        true_dead_experts_count = sum(1 for count in expert_true_activation_counts.values() if count == 0)
        
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Total Dead Experts (Post-Fallback): {dead_experts_count}")
        print(f"Total Dead Experts (Pre-Fallback): {true_dead_experts_count}")

        dead_expert_ids = [eid for eid, count in expert_true_activation_counts.items() if count == 0]
        
        experts_to_revive = dead_expert_ids
        if config.k_reborn_experts != -1 and len(dead_expert_ids) > config.k_reborn_experts:
            experts_to_revive = random.sample(dead_expert_ids, config.k_reborn_experts)

        if experts_to_revive:
            with torch.no_grad():
                for expert_type, expert_sub_idx, layer_idx in experts_to_revive:
                    if expert_type == 1: # SMHA
                        gating_network = model.layers[layer_idx].smha_layer.gating_network
                    else: # MoE
                        gating_network = model.layers[layer_idx].moe_layer.gating_network

                    new_weights = torch.randn_like(gating_network.sim_matrix[:, expert_sub_idx])
                    gating_network.sim_matrix[:, expert_sub_idx] = F.normalize(new_weights, dim=0)
                    gating_network.gates.data[expert_sub_idx] = 0.0
            print(f"Reborn {len(experts_to_revive)} experts.")


        sample_item = random.choice(data)
        prompt = f"user: {sample_item['messages'][0]['content']}\nassistant: "
        generated_text = generate_text(model, tokenizer, prompt)
        print(f"--- Sample Generation (Epoch {epoch+1}) ---\nuser: {sample_item['messages'][0]['content']}\nassistant:{generated_text}\n-------------------------")


if __name__ == "__main__":
    main()
