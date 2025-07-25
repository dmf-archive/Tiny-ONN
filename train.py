import argparse
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from bitsandbytes.optim import AdamW8bit
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from tiny_onn.modular import TinyOnnForCausalLM
from training.config import FullConfig, load_config
from training.data import get_dataloaders
from training.observer import UnifiedObserver


def main(config_path: str):
    config: FullConfig = load_config(Path(config_path))
    device = torch.device("cuda")
    output_dir = Path(config.observer.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = TinyOnnForCausalLM.from_pretrained(
        config.model.model_path, trust_remote_code=True
    ).to(device, dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_path,
        model_max_length=config.data.max_seq_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rotary_emb = Qwen3RotaryEmbedding(cast(Qwen3Config, model.config)).to(device)

    train_dataloader, _ = get_dataloaders(
        data_config=config.data,
        tokenizer=tokenizer,
        batch_size=config.train.per_device_train_batch_size,
        num_workers=0,
    )

    expert_params = [p for n, p in model.named_parameters() if "gate" not in n and p.requires_grad]
    selection_params = [p for n, p in model.named_parameters() if "gate" in n and p.requires_grad]

    optimizer_expert = AdamW8bit(
        expert_params,
        lr=config.train.expert_learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        eps=config.train.adam_epsilon,
        weight_decay=config.train.weight_decay,
    )
    optimizer_gate = AdamW8bit(
        selection_params,
        lr=config.train.gate_learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        eps=config.train.adam_epsilon,
        weight_decay=config.train.weight_decay,
    )

    num_training_steps = len(train_dataloader) * config.train.num_train_epochs
    lr_scheduler_expert = get_scheduler(
        name="linear",
        optimizer=optimizer_expert,
        num_warmup_steps=config.train.lr_scheduler_warmup_steps,
        num_training_steps=num_training_steps,
    )
    lr_scheduler_gate = get_scheduler(
        name="linear",
        optimizer=optimizer_gate,
        num_warmup_steps=config.train.lr_scheduler_warmup_steps,
        num_training_steps=num_training_steps,
    )

    observer = UnifiedObserver(output_dir, device)
    global_step = 0

    model.train()
    for epoch in range(config.train.num_train_epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.pop("labels")
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")

            if attention_mask is not None:
                attention_mask = attention_mask.to(torch.bool)
                inputs["attention_mask"] = attention_mask

            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            position_embeddings = rotary_emb(torch.zeros_like(input_ids, dtype=model.dtype).unsqueeze(-1), position_ids)

            surprise_cache: list[dict[str, Any]] = []
            logits_cache: list[torch.Tensor] = []
            inputs["surprise_cache"] = surprise_cache
            inputs["logits_cache"] = logits_cache
            inputs["position_embeddings"] = position_embeddings

            optimizer_expert.zero_grad()
            optimizer_gate.zero_grad()

            outputs = model(**inputs)
            main_loss = F.cross_entropy(outputs.logits.view(-1, model.config.vocab_size), labels.view(-1))
            main_loss.backward()
            optimizer_expert.step()
            lr_scheduler_expert.step()

            num_layers = model.config.num_hidden_layers
            num_tokens = batch_size * seq_len
            num_experts = model.config.num_experts_per_layer
            surprise_matrix = torch.full((num_layers, num_tokens, num_experts), float("inf"), device=device, dtype=torch.bfloat16)
            for entry in surprise_cache:
                surprise_matrix[entry["layer_idx"], entry["token_idx"], entry["expert_idx"]] = entry["norm"]

            optimal_indices_per_layer = torch.argmin(surprise_matrix, dim=2)
            optimal_indices = optimal_indices_per_layer.view(-1)

            all_router_logits_detached = torch.cat(logits_cache, dim=0)
            smk_loss = F.cross_entropy(all_router_logits_detached, optimal_indices)

            detached_inputs = {
                k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
                if k not in ["surprise_cache", "logits_cache"]
            }
            gate_only_logits = model.forward_gate_only(**detached_inputs)
            smk_loss_for_grad = F.cross_entropy(gate_only_logits, optimal_indices)
            (smk_loss_for_grad * config.train.smk_loss_weight).backward()
            optimizer_gate.step()
            lr_scheduler_gate.step()

            with torch.no_grad():
                main_loss_item = main_loss.item()
                smk_loss_item = smk_loss.item()
                main_acc = (outputs.logits.argmax(-1) == labels).float().mean().item()
                gating_acc = (torch.argmax(all_router_logits_detached, dim=1) == optimal_indices).float().mean().item()

                valid_surprise = surprise_matrix[surprise_matrix != float("inf")]
                surprise_val = valid_surprise.mean().item() if valid_surprise.numel() > 0 else 0.0

                tau = torch.distributions.Categorical(logits=outputs.logits.float()).entropy().mean().item()
                cost = (1 - config.observer.pi_gamma) * (main_loss_item / tau) + config.observer.pi_gamma * surprise_val
                pi_score = torch.exp(torch.tensor(-config.observer.pi_alpha * cost)).item()

                k_per_token_list = outputs.aux_outputs["k_per_token"]
                avg_k_float = torch.cat([k.float().view(-1) for k in k_per_token_list]).mean().item() if k_per_token_list else 0.0
                global_avg_k_float = torch.stack([k.float() for k in k_per_token_list]).sum(dim=0).mean().item() if k_per_token_list else 0.0


                metrics = {
                    "type": "train", "step": global_step, "main_loss": main_loss_item, "smk_loss": smk_loss_item,
                    "main_acc": main_acc, "gating_acc": gating_acc,
                    "surprise": surprise_val, "tau": float(tau), "pi_score": pi_score, "avg_k": avg_k_float,
                    "global_avg_k": global_avg_k_float,
                }
                progress_bar.set_postfix(metrics)
                observer.log_metrics(metrics, global_step)

                observer.update_expert_data(
                    optimal_indices=optimal_indices,
                    all_router_logits=logits_cache,
                    model=model,
                    global_step=global_step,
                    routing_mask_list=outputs.aux_outputs["routing_mask"],
                )
                observer.plot_dashboards(global_step, model.config)

            global_step += 1

    observer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args.config)
