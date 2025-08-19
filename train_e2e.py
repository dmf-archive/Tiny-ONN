import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from torch.distributions import Categorical
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from tiny_onn.config import TinyOnnConfig
from tiny_onn.modular import TinyOnnForCausalLM


class ChatDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, data: list, max_length: int):
        self.tokenizer = tokenizer
        self.messages = [item['messages'] for item in data]
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        messages = self.messages[idx]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).squeeze(0)

        labels = input_ids.clone()
        
        assistant_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        
        assistant_indices = (input_ids == assistant_token_id).nonzero(as_tuple=True)[0]
        if assistant_indices.numel() > 0:
            assistant_start_index = assistant_indices[-1]
            labels[:assistant_start_index] = -100
        
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, labels

def generate_text(model: TinyOnnForCausalLM, tokenizer: PreTrainedTokenizerFast, messages: list[dict[str, str]], device: torch.device, max_new_tokens: int = 30) -> str:
    model.eval()
    prompt_input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(prompt_input_ids, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)

    model.train()
    return tokenizer.decode(generated_ids[0][prompt_input_ids.shape[1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/E2E_test_train.yaml")
    args = parser.parse_args()

    with open(args.config_path) as f:
        yaml_config = yaml.safe_load(f)

    model_config_dict = yaml_config['model']
    train_config_dict = yaml_config['train']
    data_config_dict = yaml_config['data']

    tokenizer_path = model_config_dict.pop("tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir="./weights", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model_config_dict['vocab_size'] = len(tokenizer)
    config = TinyOnnConfig(**model_config_dict)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyOnnForCausalLM(config).to(device=DEVICE, dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))
    
    optimizer = AdamW(model.parameters(), lr=float(train_config_dict['learning_rate']))

    with open(data_config_dict['train_path'], encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    dataset = ChatDataset(tokenizer, data, config.max_position_embeddings)

    global_step = 0
    for epoch in range(train_config_dict['epochs']):
        print(f"\n--- Starting Epoch {epoch+1}/{train_config_dict['epochs']} ---")

        expert_activation_counts = {}
        for layer_idx in range(config.num_hidden_layers):
            for expert_type in ["smha", "moe"]:
                num_experts = config.max_attention_experts if expert_type == "smha" else config.max_moe_experts
                for expert_sub_idx in range(num_experts):
                    expert_id = (expert_type, layer_idx, expert_sub_idx)
                    expert_activation_counts[expert_id] = 0

        for i in range(len(dataset)):
            step_start_time = time.time()
            input_ids, labels = dataset[i]
            input_ids, labels = input_ids.unsqueeze(0).to(DEVICE), labels.unsqueeze(0).to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(input_ids=input_ids)
            final_logits = outputs.logits
            forward_cache = outputs.aux_outputs

            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            main_loss = F.cross_entropy(rearrange(shift_logits, 'b t d -> (b t) d'), rearrange(shift_labels, 'b t -> (b t)'), ignore_index=-100)
            if torch.isnan(main_loss) or main_loss.isinf(): continue

            total_gating_loss = torch.tensor(0.0, device=DEVICE)
            metrics = { "smha_surprise": [], "moe_surprise": [], "smha_gate_acc": [], "moe_gate_acc": [], "smha_avg_k": [], "moe_avg_k": [] }

            if forward_cache:
                grad_outputs = [cache['final_output'] for cache in forward_cache.values()]
                if grad_outputs:
                    grads_for_surprise = torch.autograd.grad(main_loss, grad_outputs, create_graph=False, retain_graph=True)

                    for grad_idx, (expert_id_tuple, layer_cache) in enumerate(forward_cache.items()):
                        layer_type_str, layer_index, _ = expert_id_tuple

                        B, T = layer_cache['B'], layer_cache['T']
                        grad_norm = torch.linalg.norm(grads_for_surprise[grad_idx].view(B * T, -1), dim=-1)
                        routing_weights = layer_cache['routing_weights']
                        surprise_matrix = grad_norm.unsqueeze(-1) * routing_weights
                        logits = layer_cache['gate_cache']['logits']

                        with torch.no_grad():
                            targets = torch.argmin(surprise_matrix, dim=-1)
                            acc = (logits.argmax(dim=-1) == targets).float().mean()

                        log_targ = F.log_softmax(-surprise_matrix.detach(), dim=-1)
                        log_gate = F.log_softmax(logits, dim=-1)

                        w_ce = config.w_ce_smha if layer_type_str == "smha" else config.w_ce_moe
                        w_kl = config.w_kl_smha if layer_type_str == "smha" else config.w_kl_moe
                        w_aux = config.w_aux_smha if layer_type_str == "smha" else config.w_aux_moe

                        gating_loss = w_ce * F.cross_entropy(logits, targets) + w_kl * F.kl_div(log_gate, log_targ, reduction='batchmean', log_target=True)
                        total_gating_loss += w_aux * gating_loss

                        activation_mask = layer_cache['gate_cache']['activation_mask'] > 0
                        active_surprise = surprise_matrix[activation_mask]
                        metrics[f"{layer_type_str}_surprise"].append(active_surprise.mean().item() if active_surprise.numel() > 0 else 0)
                        metrics[f"{layer_type_str}_gate_acc"].append(acc.item())
                        metrics[f"{layer_type_str}_avg_k"].append(layer_cache["num_active_tokens"] / (B * T))

                        active_experts_in_batch = torch.where(activation_mask.sum(dim=0) > 0)[0].tolist()
                        for expert_sub_idx in active_experts_in_batch:
                            expert_activation_counts[(layer_type_str, layer_index, expert_sub_idx)] += 1

            total_loss = main_loss + total_gating_loss
            if not (torch.isnan(total_loss) or total_loss.isinf()):
                total_loss.backward()

            optimizer.step()
            global_step += 1

            if global_step % 20 == 0:
                with torch.no_grad():
                    mask = (shift_labels != -100)
                    main_acc = ((shift_logits.argmax(-1) == shift_labels) & mask).float().sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0)
                    tau = Categorical(logits=final_logits).entropy().mean()
                    
                    all_surprises = metrics["smha_surprise"] + metrics["moe_surprise"]
                    surprise = sum(all_surprises) / len(all_surprises) if all_surprises else 0
                    
                    pi_score = torch.exp(-config.pi_alpha * ((1 - config.pi_gamma) * (main_loss / (tau + 1e-9)) + config.pi_gamma * surprise))

                    num_smha_layers = config.num_hidden_layers if metrics["smha_surprise"] else 0
                    num_moe_layers = config.num_hidden_layers if metrics["moe_surprise"] else 0

                    avg_gating_loss = total_gating_loss / (num_smha_layers + num_moe_layers) if (num_smha_layers + num_moe_layers > 0) else torch.tensor(0.0)
                    avg_smha_acc = sum(metrics["smha_gate_acc"]) / len(metrics["smha_gate_acc"]) if metrics["smha_gate_acc"] else 0
                    avg_moe_acc = sum(metrics["moe_gate_acc"]) / len(metrics["moe_gate_acc"]) if metrics["moe_gate_acc"] else 0
                    avg_smha_k = sum(metrics["smha_avg_k"]) / len(metrics["smha_avg_k"]) if metrics["smha_avg_k"] else 0
                    avg_moe_k = sum(metrics["moe_avg_k"]) / len(metrics["moe_avg_k"]) if metrics["moe_avg_k"] else 0
                    it_per_sec = 1.0 / (time.time() - step_start_time)
                    print(
                        f"  Step {global_step}: PI: {pi_score.item():.2f} | "
                        f"Loss(M/G): {main_loss.item():.2f}/{avg_gating_loss.item():.2f} | "
                        f"Acc(M/S/M): {main_acc.item():.2f}/{avg_smha_acc:.2f}/{avg_moe_acc:.2f} | "
                        f"K(S/M): {avg_smha_k:.2f}/{avg_moe_k:.2f} | "
                        f"Speed: {it_per_sec:.2f} it/s"
                    )

        dead_experts = [eid for eid, count in expert_activation_counts.items() if count == 0]
        if dead_experts:
            print(f"--- Epoch {epoch+1} Summary: Found {len(dead_experts)} dead experts. Reviving... ---")
            with torch.no_grad():
                for expert_type, layer_idx, expert_sub_idx in dead_experts:
                    layer = model.model.layers[layer_idx]
                    gating_network = layer.smha_layer.gating_network if expert_type == "smha" else layer.moe_layer.gating_network
                    new_weights = torch.randn_like(gating_network.sim_matrix[:, expert_sub_idx])
                    gating_network.sim_matrix[:, expert_sub_idx] = F.normalize(new_weights, dim=0)
                    gating_network.gates.data[expert_sub_idx] = 0.0

        sample_item = random.choice(data)
        user_message = {"role": "user", "content": next((msg['content'] for msg in sample_item['messages'] if msg['role'] == 'user'), '')}
        messages = [user_message]
        prompt = tokenizer.apply_chat_template([user_message], tokenize=False, add_generation_prompt=True)
        generated_output = generate_text(model, tokenizer, prompt, device=DEVICE)
        print(f"--- Sample Generation (Epoch {epoch+1}) ---\nuser: {user_message['content']}\nassistant:{generated_output}\n-------------------------")

if __name__ == "__main__":
    main()
