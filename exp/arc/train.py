import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from torch.utils.data import DataLoader

from .config import TrainConfig
from .consistency import ConsistencyTools
from .data import ArcCollator, GridDeserializer, GridSerializer, InMemoryArcDataset
from .evaluation import EvaluationStep
from .model import ArcTransformer, SparseProtoLinear
from .observer import Observer
from .tokenizer import ArcColorTokenizer

torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)

        self.console = Console()
        self.observer = Observer(self.console, config)
        self.tokenizer = ArcColorTokenizer()
        self.serializer = GridSerializer(self.tokenizer)
        self.deserializer = GridDeserializer(self.tokenizer)

        train_dataset = InMemoryArcDataset(data_path=config.data.data_path, split="training")
        eval_dataset = InMemoryArcDataset(data_path=config.data.data_path, split="evaluation")

        train_collator = ArcCollator(self.tokenizer, max_len=config.model.max_position_embeddings)
        eval_collator = ArcCollator(self.tokenizer, max_len=config.model.max_position_embeddings)

        self.train_loader = DataLoader(
            train_dataset, batch_size=config.data.batch_size, collate_fn=train_collator,
            num_workers=config.data.num_workers, shuffle=False
        )
        self.eval_loader = DataLoader(
            eval_dataset, batch_size=1, collate_fn=eval_collator, num_workers=config.data.num_workers, shuffle=False
        )

        self.config.model.vocab_size = self.tokenizer.vocab_size
        self.model = ArcTransformer(self.config.model, device=self.device).to(self.device)

        main_params = [p for name, p in self.model.named_parameters() if 'gate_param' not in name]
        meta_params = [p for name, p in self.model.named_parameters() if 'gate_param' in name]

        self.optimizer_main = torch.optim.AdamW(main_params, lr=self.config.lr_main, weight_decay=0.0)
        self.optimizer_meta = torch.optim.AdamW(meta_params, lr=self.config.lr_meta, weight_decay=0.0)

        self.consistency_tools = ConsistencyTools()
        self.evaluator = EvaluationStep(self.model, self.serializer, self.deserializer, self.observer, self.device, train_dataset, self.config)

        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.global_step = 0
        self.epoch = 0
        self.start_task_idx = 0
        self.start_view_idx = 0
        self.ema_acc = 0.0

    def _save_checkpoint(self, task_idx: int, view_idx: int):
        state = {
            'epoch': self.epoch,
            'step': self.global_step,
            'task_idx': task_idx,
            'view_idx': view_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_main_state_dict': self.optimizer_main.state_dict(),
            'optimizer_meta_state_dict': self.optimizer_meta.state_dict(),
            'ema_acc': self.ema_acc,
        }
        filepath = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        torch.save(state, filepath)
        checkpoints = sorted(self.checkpoint_dir.glob("*.pt"), key=os.path.getmtime)
        if len(checkpoints) > self.config.max_checkpoints: os.remove(checkpoints[0])

    def _load_checkpoint(self):
        checkpoints = sorted(self.checkpoint_dir.glob("*.pt"), key=os.path.getmtime, reverse=True)
        if not checkpoints:
            self.console.print("[bold yellow]No checkpoint found, starting from scratch.[/bold yellow]")
            return
        for checkpoint_path in checkpoints:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer_main.load_state_dict(checkpoint['optimizer_main_state_dict'])
                self.optimizer_meta.load_state_dict(checkpoint['optimizer_meta_state_dict'])
                self.global_step = checkpoint['step']
                self.epoch = checkpoint.get('epoch', 0)
                self.start_task_idx = checkpoint.get('task_idx', 0)
                self.start_view_idx = checkpoint.get('view_idx', 0)
                self.ema_acc = checkpoint.get('ema_acc', 0.0)
                self.console.print(f"[bold green]Successfully loaded checkpoint from {checkpoint_path} at step {self.global_step}. Resuming from task {self.start_task_idx}, view {self.start_view_idx}.[/bold green]")
                return
            except (RuntimeError, KeyError, EOFError) as e:
                self.console.print(f"[bold red]Checkpoint {checkpoint_path} appears to be corrupted or incomplete ({e}). Deleting and trying the next one.[/bold red]")
                os.remove(checkpoint_path)
        self.console.print("[bold yellow]No valid checkpoint found. Starting training from scratch.[/bold yellow]")

    def _get_spl_layers(self) -> list[SparseProtoLinear]:
        return [m for m in self.model.modules() if isinstance(m, SparseProtoLinear)]

    def _calculate_saps_loss(
        self,
        proto_weights: list[torch.Tensor],
        sbl_inputs: list[torch.Tensor],
        raw_weights: list[torch.Tensor],
        mu_surprises: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        total_proto_loss = torch.tensor(0.0, device=self.device, dtype=torch.bfloat16)
        all_saps_masks = []
        for p, x, rw, mu_s in zip(proto_weights, sbl_inputs, raw_weights, mu_surprises):
            p = p.contiguous()
            x = x.contiguous()
            rw = rw.contiguous()
            mu_s = mu_s.contiguous()
            
            N, _ = p.shape
            saps_mask = torch.zeros(N, device=self.device, dtype=torch.int8)

            if x.numel() == 0 or rw.numel() == 0 or mu_s.numel() == 0:
                all_saps_masks.append(saps_mask)
                continue

            B, S, D = x.shape
            x_flat = x.view(B * S, D)
            rw_flat = rw.view(B * S, N)
            activated_mask = rw_flat > 0

            if not activated_mask.any():
                all_saps_masks.append(saps_mask)
                continue

            activation_rate = activated_mask.float().mean()
            dynamic_factor = activation_rate
            
            mu_s_float = mu_s.float()
            surprise_q_low = torch.quantile(mu_s_float, dynamic_factor)
            surprise_q_high = torch.quantile(mu_s_float, 1.0 - dynamic_factor)

            is_good = mu_s <= surprise_q_low
            is_bad = mu_s >= surprise_q_high
            
            saps_mask[is_good] = 1
            saps_mask[is_bad] = 2
            all_saps_masks.append(saps_mask)

            loss_mask = is_good | is_bad
            if not loss_mask.any():
                continue

            signs = torch.ones(N, device=self.device)
            signs[is_good] = -1.0
            signs[is_bad] = 1.0

            # Vectorized calculation of local anchors
            # Sum of tokens for each expert, divided by count of tokens for that expert
            expert_token_counts = activated_mask.sum(dim=0).clamp(min=1)  # [N]
            expert_token_sums = torch.matmul(activated_mask.to(dtype=x_flat.dtype).t(), x_flat)  # [N, D]
            anchors = expert_token_sums / expert_token_counts.unsqueeze(1)  # [N, D]
            anchors = F.normalize(anchors, p=2, dim=-1)

            p_norm = F.normalize(p, p=2, dim=-1)
            similarities = F.cosine_similarity(p_norm, anchors, dim=-1)

            adaptive_strength = mu_s[loss_mask]
            proto_loss_per_expert = signs[loss_mask] * (adaptive_strength * (1 - similarities[loss_mask]))
            total_proto_loss += proto_loss_per_expert.mean()

        final_loss = total_proto_loss / len(proto_weights) if proto_weights else torch.tensor(0.0)
        return final_loss, all_saps_masks

    def _calculate_gate_loss(
        self,
        predicted_costs: list[torch.Tensor],
        mu_surprises: list[torch.Tensor],
        proto_surprises: list[torch.Tensor]
    ) -> torch.Tensor:
        total_gate_loss = torch.tensor(0.0, device=self.device, dtype=torch.bfloat16)
        num_losses = 0
        for pc, mu_s, proto_s in zip(predicted_costs, mu_surprises, proto_surprises):
            pc = pc.contiguous()
            mu_s = mu_s.contiguous()
            proto_s = proto_s.contiguous()
            if pc is None or mu_s is None or proto_s is None:
                continue
            target_surprise = (mu_s + proto_s).detach()
            total_gate_loss += F.mse_loss(pc.mean(dim=(0, 1)), target_surprise)
            num_losses += 1
        return total_gate_loss / num_losses if num_losses > 0 else torch.tensor(0.0)

    def _calculate_adaptive_l2_penalty(self, spl_layers: list[SparseProtoLinear]) -> torch.Tensor:
        all_weights = []
        for layer in spl_layers:
            all_weights.append(layer.mu_weight)
            all_weights.append(layer.proto_weight)
        
        if not all_weights:
            return torch.tensor(0.0, device=self.device)

        avg_norm = torch.stack([w.norm(p=2) for w in all_weights]).mean()
        lambda_l2 = avg_norm * 1e-4
        
        penalty = sum(w.pow(2).sum() for w in all_weights)
        return lambda_l2 * penalty

    def _calculate_dynamic_protection_mask(self, spl_layers: list[SparseProtoLinear]) -> list[torch.Tensor]:
        """Returns a list of protection masks for proto_weight.grad, one for each layer."""
        if not spl_layers:
            return []

        all_gate_norms = torch.cat([layer.gate_param.norm(p=2, dim=-1) for layer in spl_layers])
        temperature = max(1.0 - self.ema_acc, 0.01)

        with torch.no_grad():
            # CORRECTED: Protect proto_weights with LOW gate_param norms.
            # Low norm -> High quality score -> High protection probability.
            protection_probs = F.softmax(all_gate_norms / temperature, dim=0)

        protection_masks = []
        start_idx = 0
        for layer in spl_layers:
            num_experts = layer.gate_param.shape[0]
            end_idx = start_idx + num_experts
            protection_mask = 1.0 - protection_probs[start_idx:end_idx]
            protection_masks.append(protection_mask.unsqueeze(1)) # Add a dimension for broadcasting
            start_idx = end_idx
        return protection_masks

    def _apply_mu_gradient_clamp(self, mu_weights: list[torch.Tensor], mu_surprises: list[torch.Tensor]):

        for mu_w, mu_s in zip(mu_weights, mu_surprises):
            if mu_w.grad is None or mu_s.numel() == 0:
                continue

            min_norm = 0.1
            max_norms = torch.clamp(1.0 - mu_s, min=min_norm, max=1.0)
            
            torch.nn.utils.clip_grad_norm_(mu_w, max_norms)

    def _run_step(self, mini_task: dict[str, Any], view_idx: int, epoch: int, mini_task_idx: int):
        start_time = time.time()
        
        input_grid = torch.tensor(mini_task['input'], device=self.device)
        output_grid = torch.tensor(mini_task['output'], device=self.device)
        transform = self.consistency_tools.get_transforms()[view_idx]
        augmented_mini_task = {'input': transform(input_grid).cpu().tolist(), 'output': transform(output_grid).cpu().tolist()}
        
        input_ids_list, labels_list = self.serializer.serialize_mini_task(augmented_mini_task)
        if len(input_ids_list) > self.config.model.max_position_embeddings: return None
        
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)
        labels = torch.tensor([labels_list], dtype=torch.long, device=self.device)

        self.model.train()
        self.optimizer_main.zero_grad()
        self.optimizer_meta.zero_grad()

        logits, _, _, _, _, all_sbl_inputs, all_raw_weights, all_predicted_costs, _ = self.model(input_ids)
        main_loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, self.config.model.vocab_size), labels[:, 1:].contiguous().view(-1), ignore_index=-100)
        if not torch.isfinite(main_loss): return None

        spl_layers = self._get_spl_layers()
        mu_weights = [p for name, p in self.model.named_parameters() if 'mu_weight' in name and p.requires_grad]
        proto_weights = [p for name, p in self.model.named_parameters() if 'proto_weight' in name and p.requires_grad]
        
        if not mu_weights:
            return None

        # Phase 1: Calculate all losses and their gradients w.r.t. their primary parameters.
        # This phase must complete all autograd.grad calls before any parameter is modified.
        # CRITICAL: Detach the surprise values to avoid second-order derivatives and inplace errors.

        # 1B: Calculate mu_surprise from main_loss
        mu_grads = torch.autograd.grad(main_loss, mu_weights, retain_graph=True, allow_unused=True)
        mu_surprises = [g.norm(p=2, dim=-1).detach() if g is not None else torch.zeros(p.shape[0], device=self.device, dtype=p.dtype) for g, p in zip(mu_grads, mu_weights)]

        # Calculate proto_loss using mu_surprise
        proto_loss, saps_masks = self._calculate_saps_loss(proto_weights, all_sbl_inputs, all_raw_weights, mu_surprises)
        
        # 2B: Calculate proto_surprise from proto_loss
        proto_grads = torch.autograd.grad(proto_loss, proto_weights, retain_graph=True, allow_unused=True)
        proto_surprises = [g.norm(p=2, dim=-1).detach() if g is not None else torch.zeros(p.shape[0], device=self.device, dtype=p.dtype) for g, p in zip(proto_grads, proto_weights)]

        # Calculate gate_loss using mu_surprise and proto_surprise
        gate_loss = self._calculate_gate_loss(all_predicted_costs, mu_surprises, proto_surprises)

        # Phase 2: Accumulate gradients for main optimizer
        adaptive_l2_penalty = self._calculate_adaptive_l2_penalty(spl_layers)
        total_loss = main_loss + proto_loss + adaptive_l2_penalty
        
        self.optimizer_main.zero_grad()
        if torch.isfinite(total_loss):
            total_loss.backward(retain_graph=True) # Accumulates gradients for mu_weight and proto_weight

        # Phase 3: Update meta parameters (gate_param)
        self.optimizer_meta.zero_grad()
        if torch.isfinite(gate_loss):
            (self.config.w_gate * gate_loss).backward() # Calculates gradients for gate_param

        # Phase 4: Update parameters
        self.optimizer_main.step()
        self.optimizer_meta.step()

        with torch.no_grad():
            # Calculate accuracy only on the generated output tokens, excluding the input prompt and pad tokens.
            logits_for_acc = logits[:, :-1, :]
            labels_for_acc = labels[:, 1:]
            active_mask = labels_for_acc != -100
            if active_mask.any():
                active_logits = logits_for_acc[active_mask]
                active_labels = labels_for_acc[active_mask]
                token_acc = (torch.argmax(active_logits, dim=-1) == active_labels).float().mean().item()
            else:
                token_acc = 0.0
            self.ema_acc = self.config.ema_alpha_acc * self.ema_acc + (1 - self.config.ema_alpha_acc) * token_acc
            
            # Calculate missing metrics for observer
            # Display predicted_cost statistics instead of raw gate_param norms
            if all_predicted_costs:
                all_pc_flat = torch.cat([pc.detach().float().view(-1) for pc in all_predicted_costs])
                raw_top10_mean = all_pc_flat[all_pc_flat >= torch.quantile(all_pc_flat, 0.9)].mean().item() if all_pc_flat.numel() > 0 else 0.0
                raw_avg_gate_val = all_pc_flat.mean().item() if all_pc_flat.numel() > 0 else 0.0
                raw_max_gate_val = all_pc_flat.max().item() if all_pc_flat.numel() > 0 else 0.0
            else:
                raw_top10_mean, raw_avg_gate_val, raw_max_gate_val = 0.0, 0.0, 0.0

            num_spl_per_block = 4
            num_layers = self.config.model.num_layers
            act_rate_l0, act_rate_l_mid, act_rate_ln = 0.0, 0.0, 0.0
            if all_raw_weights:
                num_weights = len(all_raw_weights)
                l0_weights = all_raw_weights[:num_spl_per_block]
                if l0_weights:
                    act_rate_l0 = torch.cat([rw.view(-1) for rw in l0_weights if rw.numel() > 0]).gt(0).float().mean().item() if any(rw.numel() > 0 for rw in l0_weights) else 0.0

                mid_block_idx = num_layers // 2
                mid_start = mid_block_idx * num_spl_per_block
                mid_end = mid_start + num_spl_per_block
                if num_weights >= mid_end:
                    mid_weights = all_raw_weights[mid_start:mid_end]
                    act_rate_l_mid = torch.cat([rw.view(-1) for rw in mid_weights if rw.numel() > 0]).gt(0).float().mean().item() if any(rw.numel() > 0 for rw in mid_weights) else 0.0

                last_block_start = (num_layers - 1) * num_spl_per_block
                if num_weights > last_block_start:
                    ln_weights = all_raw_weights[last_block_start:]
                    act_rate_ln = torch.cat([rw.view(-1) for rw in ln_weights if rw.numel() > 0]).gt(0).float().mean().item() if any(rw.numel() > 0 for rw in ln_weights) else 0.0

            metrics = {
                "main_loss": main_loss.item(),
                "proto_loss": proto_loss.item(),
                "gate_loss": gate_loss.item(),
                "token_acc": token_acc,
                "ema_acc": self.ema_acc,
                "pi_score": torch.exp(-main_loss).item(),
                "tau": -torch.sum(F.softmax(active_logits, dim=-1) * F.log_softmax(active_logits, dim=-1), dim=-1).mean().item() if active_mask.any() else 0.0,
                "seq_len": float(input_ids.shape[1]),
                "activation_rate_avg": torch.cat([rw.view(-1) for rw in all_raw_weights]).gt(0).float().mean().item() if all_raw_weights else 0.0,
                "activation_rate_l0": act_rate_l0,
                "activation_rate_l_mid": act_rate_l_mid,
                "activation_rate_ln": act_rate_ln,
                "raw_top10_gate": raw_top10_mean,
                "raw_avg_gate": raw_avg_gate_val,
                "raw_max_gate": raw_max_gate_val,
            }
            
        elapsed_time = time.time() - start_time
        if self.global_step % self.config.log_interval == 0:
            self.observer.log_step(epoch, self.global_step, mini_task_idx, metrics, elapsed_time)
            
            # Prepare data for visualization in the new format
            spl_module_names = ["attn_qkv", "attn_o", "ffn_sbl1", "ffn_sbl2"]
            saps_data_per_block = []
            
            # Group spl_layers and saps_masks by block
            num_spl_per_block = len(spl_module_names)
            num_blocks = len(spl_layers) // num_spl_per_block
            
            status_map = {0: "neutral", 1: "good", 2: "bad"}
            
            for block_idx in range(num_blocks):
                block_data = {}
                start_idx = block_idx * num_spl_per_block
                for i, name in enumerate(spl_module_names):
                    layer_idx = start_idx + i
                    if layer_idx < len(spl_layers) and i < len(saps_masks):
                        protos = spl_layers[layer_idx].proto_weight.detach().cpu()
                        statuses = [status_map[code.item()] for code in saps_masks[i]]
                        block_data[name] = {"protos": protos, "statuses": statuses}
                saps_data_per_block.append(block_data)

            self.observer.visualize_saps_clusters(saps_data_per_block, self.global_step)
            
            self._save_checkpoint(mini_task_idx, view_idx)

        if self.global_step > 0 and self.global_step % self.config.eval_interval == 0:
            self.evaluator.run(self.eval_loader, mini_task_idx, self.global_step)

        self.global_step += 1
        torch.cuda.empty_cache()
        return metrics

    def _train_epoch(self, epoch: int):
        self.model.train()
        num_mini_tasks = len(self.train_loader.dataset)
        for mini_task_idx in range(self.start_task_idx, num_mini_tasks):
            mini_task = self.train_loader.dataset[mini_task_idx]
            start_view = self.start_view_idx if mini_task_idx == self.start_task_idx else 0
            for view_idx in range(start_view, 8):
                inner_step = 0
                MAX_INNER_STEPS = 500
                while inner_step < MAX_INNER_STEPS:
                    metrics = self._run_step(mini_task, view_idx, epoch, mini_task_idx)
                    if metrics is None:
                        self.console.print(f"[yellow]Skipping mini-task {mini_task_idx} view {view_idx} (too long or NaN).[/yellow]")
                        break
                    
                    if metrics["token_acc"] >= 1.0 and metrics["tau"] <= 0.01:
                        self.console.print(f"Mini-task {mini_task_idx} view {view_idx} CONVERGED in {inner_step + 1} steps.")
                        break
                    inner_step += 1
                if inner_step == MAX_INNER_STEPS:
                    self.console.print(f"[red]Mini-task {mini_task_idx} view {view_idx} hit MAX_INNER_STEPS.[/red]")
            if mini_task_idx == self.start_task_idx:
                self.start_view_idx = 0
        self.start_task_idx = 0
        self.console.print(f"[bold yellow]End of Epoch {epoch}.[/bold yellow]")

    def train(self):
        self._load_checkpoint()
        self.console.print("[bold green]Starting Training...[/bold green]")
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            self._train_epoch(epoch)
        self.console.print("[bold green]Training Finished.[/bold green]")

def main():
    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
