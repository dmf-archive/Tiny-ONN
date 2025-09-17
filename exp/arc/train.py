import os
import time
from pathlib import Path
from typing import Any, List, Tuple

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
import torch.nn as nn
from rich.console import Console
from torch.utils.data import DataLoader

from .config import TrainConfig
from .consistency import ConsistencyTools
from .data import ArcCollator, InMemoryArcDataset, GridDeserializer, GridSerializer
from .evaluation import EvaluationStep
from .model import ArcTransformer
from .observer import Observer
from .tokenizer import ArcColorTokenizer


import math

class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.dead_proto_threshold = 0.01
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
        
        # JIT compile entire computation graph for maximum performance
        self._jit_compile_modules()

        main_params = []
        meta_params = []
        for name, p in self.model.named_parameters():
            if 'proto_weight' in name or 'gate_param' in name:
                meta_params.append(p)
            else:
                main_params.append(p)

        self.optimizer_main = torch.optim.AdamW(main_params, lr=self.config.lr_main, weight_decay=0.0) # Now uses adaptive decay
        self.optimizer_meta = torch.optim.AdamW(meta_params, lr=self.config.lr_meta, weight_decay=0.0) # Also uses adaptive decay

        self.consistency_tools = ConsistencyTools()
        self.evaluator = EvaluationStep(self.model, self.serializer, self.deserializer, self.observer, self.device, train_dataset)

        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.global_step = 0
        self.epoch = 0
        self.start_task_idx = 0
        self.start_view_idx = 0
        self.last_task_activation_profile: torch.Tensor | None = None

    def _jit_compile_modules(self):
        """Strategic JIT compilation focusing on compute-intensive paths"""
        from .model import SparseProtoLinear
        
        compiled_count = 0
        
        # Only compile SPL modules - these are the compute bottlenecks
        spl_modules = [m for m in self.model.modules() if isinstance(m, SparseProtoLinear)]
        for i, module in enumerate(spl_modules):
            try:
                scripted_module = torch.jit.script(module)
                parent, name = self._find_module_parent(self.model, module)
                if parent and name:
                    setattr(parent, name, scripted_module)
                    compiled_count += 1
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to JIT compile SPL module {i}: {e}[/yellow]")
        
        self.console.print(f"[green]Successfully JIT compiled {compiled_count} SPL modules[/green]")

    def _get_spl_module_attributes(self, module):
        """Safely access SPL module attributes with TorchScript compatibility"""
        if hasattr(module, 'mu_weight'):
            return module.mu_weight, module.proto_weight, module.mu_bias, module.gate_param
        else:
            # For scripted modules, access parameters through named_parameters
            params = dict(module.named_parameters())
            return params['mu_weight'], params['proto_weight'], params['mu_bias'], params['gate_param']

    def _find_module_parent(self, model, target_module):
        """Find parent module and attribute name for a given module"""
        for name, module in model.named_modules():
            for child_name, child in module.named_children():
                if child is target_module:
                    return module, child_name
        return None, None

    def _save_checkpoint(self, task_idx: int, view_idx: int):
        state = {
            'epoch': self.epoch,
            'step': self.global_step,
            'task_idx': task_idx,
            'view_idx': view_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_main_state_dict': self.optimizer_main.state_dict(),
            'optimizer_meta_state_dict': self.optimizer_meta.state_dict(),
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
                self.console.print(f"[bold green]Successfully loaded checkpoint from {checkpoint_path} at step {self.global_step}. Resuming from task {self.start_task_idx}, view {self.start_view_idx}.[/bold green]")
                return # Successfully loaded, exit the function
            except (RuntimeError, KeyError, EOFError) as e:
                self.console.print(f"[bold red]Checkpoint {checkpoint_path} appears to be corrupted or incomplete ({e}). Deleting and trying the next one.[/bold red]")
                os.remove(checkpoint_path)
        
        # If no valid checkpoint is found after the loop
        self.console.print("[bold yellow]No valid checkpoint found. Starting training from scratch.[/bold yellow]")
        self.global_step = 0
        self.epoch = 0
        self.start_task_idx = 0
        self.start_view_idx = 0

    def _reinitialize_dead_prototypes(self) -> int:
        reinitialized_count = 0

        spl_module_type = type(self.model.blocks[0].attn.sbl_qkv)
        
        # Collect all SPL layers from all blocks
        all_spl_layers = []
        for block in self.model.blocks:
            all_spl_layers.extend([
                block.attn.sbl_qkv, block.attn.sbl_o,
                block.ffn.sbl1, block.ffn.sbl2
            ])

        with torch.no_grad():
            for module in all_spl_layers:
                mu_weight, proto_weight, _, gate_param = self._get_spl_module_attributes(module)
                
                proto_norms = torch.norm(proto_weight, p=2, dim=-1)
                dead_mask = proto_norms < self.dead_proto_threshold
                num_dead = torch.sum(dead_mask).item()

                if num_dead > 0:
                    reinitialized_count += num_dead
                    
                    in_features = proto_weight.shape[1]
                    device = proto_weight.device
                    dtype = proto_weight.dtype

                    new_protos = torch.empty(num_dead, in_features, device=device, dtype=dtype)
                    nn.init.kaiming_uniform_(new_protos, a=math.sqrt(5))
                    proto_weight.data[dead_mask] = new_protos

                    new_mu = torch.empty_like(mu_weight.data[dead_mask])
                    nn.init.kaiming_uniform_(new_mu, a=math.sqrt(5))
                    mu_weight.data[dead_mask] = new_mu

                    gate_param.data[dead_mask] = 0.0
        
        return reinitialized_count


    @torch.jit.script
    def _calculate_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # JIT compiled version of token accuracy calculation
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        masked_labels = labels.contiguous().view(-1)
        active_mask = masked_labels != -100
        if not active_mask.any():
            return torch.tensor(0.0)
        active_logits = logits.contiguous().view(-1, logits.size(-1))[active_mask]
        active_labels = masked_labels[active_mask]
        preds = torch.argmax(active_logits, dim=-1)
        return (preds == active_labels).float().mean()

    @torch.jit.script
    def _calculate_activation_profile(raw_weights: List[torch.Tensor], seq_len: int) -> torch.Tensor:
        # JIT compiled version of activation profile calculation
        if not raw_weights or seq_len == 0:
            return torch.empty(0)
            
        activation_rates: List[torch.Tensor] = []
        for rw in raw_weights:
            activated = rw.gt(0)
            rate = activated.sum(dim=1, dtype=torch.float32) / seq_len
            activation_rates.append(rate.view(-1))
        
        return torch.cat(activation_rates, dim=0)

    def _calculate_meta_loss(
        self,
        proto_weights: List[torch.Tensor],
        gate_params: List[torch.Tensor],
        mu_grads: List[torch.Tensor],
        sbl_inputs: List[torch.Tensor],
        raw_weights: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate meta loss for proto and gate parameters, directly inlined from optimizer.py."""
        if not proto_weights:
            return torch.tensor(0.0), torch.tensor(0.0)

        device = proto_weights[0].device
        dtype = proto_weights[0].dtype

        total_proto_loss = torch.tensor(0.0, device=device, dtype=dtype)
        total_gate_loss = torch.tensor(0.0, device=device, dtype=dtype)

        for pw, gp, mg, si, rw in zip(proto_weights, gate_params, mu_grads, sbl_inputs, raw_weights):
            if mg is None: # Skip layers without gradients
                continue

            surprise_norm = torch.linalg.vector_norm(mg.detach(), ord=2, dim=-1).float()
            
            scaled_surprise = surprise_norm / self.config.gate_sigmoid_temperature
            target_gate = torch.sigmoid(-scaled_surprise).to(dtype=gp.dtype)
            
            gate_loss = F.mse_loss(gp, target_gate.detach())
            total_gate_loss += gate_loss

            # Proto loss: SAPS dynamics without plasticity weighting in loss calc.
            low_threshold = torch.quantile(surprise_norm, 0.3)
            high_threshold = torch.quantile(surprise_norm, 0.7)
            is_good = surprise_norm < low_threshold
            is_bad = surprise_norm > high_threshold
            is_active = is_good | is_bad

            if torch.sum(is_good) > 0 and torch.sum(is_bad) > 0:
                active_protos = pw[is_active]
                active_protos_normalized = F.normalize(active_protos, p=2.0, dim=-1)

                sbl_input_flat = si.detach().view(-1, si.shape[-1])
                raw_weight_flat = rw.detach().view(-1, rw.shape[-1])
                activation_mask = (raw_weight_flat[:, is_active] > 0).to(sbl_input_flat.dtype)
                token_counts = activation_mask.sum(dim=0)

                similarities = torch.zeros(active_protos_normalized.shape[0], device=device, dtype=dtype)
                
                if (token_counts > 0).all():
                    weighted_sums = torch.matmul(activation_mask.t(), sbl_input_flat)
                    local_anchors = weighted_sums / token_counts.unsqueeze(1)
                    similarities = torch.einsum("ph,ph->p", active_protos_normalized, local_anchors)
                else:
                    continue # Skip if no active prototypes have tokens

                signs = torch.zeros_like(similarities)
                signs[is_good[is_active]] = -1.0 # Pull good prototypes
                signs[is_bad[is_active]] = 1.0  # Push bad prototypes

                differential_loss = torch.sum(signs * similarities)
                
                active_norms = torch.linalg.vector_norm(active_protos, ord=2, dim=-1)
                l2_penalty = torch.sum(active_norms)

                total_proto_loss += (differential_loss + l2_penalty)

        return total_proto_loss, total_gate_loss

    def _run_step(self, mini_task: dict[str, Any], view_idx: int, epoch: int, mini_task_idx: int):
        
        def _calculate_adaptive_decay(weights: List[torch.Tensor], gate_params: List[torch.Tensor]) -> torch.Tensor:
            normalized_gates = [torch.tanh(gp.detach()) for gp in gate_params]
            w_plasticity = [F.relu(1.0 - ngp) for ngp in normalized_gates]
            l2_penalty = torch.stack([(self.config.base_decay * w_p.unsqueeze(1) * (w**2)).sum() for w_p, w in zip(w_plasticity, weights)]).sum()
            return l2_penalty

        start_time = time.time()
        
        # NOTE: Augmentations are now applied directly to the mini_task grids
        input_grid = torch.tensor(mini_task['input'], device=self.device)
        output_grid = torch.tensor(mini_task['output'], device=self.device)

        # Apply geometric and color augmentations
        transform = self.consistency_tools.get_transforms()[view_idx]
        input_grid_aug = transform(input_grid)
        output_grid_aug = transform(output_grid)
        
        # color_map = torch.randperm(10, device=self.device)
        # input_grid_aug = self.consistency_tools._jit_apply_color_transform(input_grid_aug, color_map)
        # output_grid_aug = self.consistency_tools._jit_apply_color_transform(output_grid_aug, color_map)

        augmented_mini_task = {'input': input_grid_aug.cpu().tolist(), 'output': output_grid_aug.cpu().tolist()}
        input_ids_list, labels_list = self.serializer.serialize_mini_task(augmented_mini_task)

        if len(input_ids_list) > self.config.model.max_position_embeddings: return None, None, None, None

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)
        labels = torch.tensor([labels_list], dtype=torch.long, device=self.device)

        self.model.train()
        self.optimizer_main.zero_grad()
        self.optimizer_meta.zero_grad()

        logits, tok_emb, _, _, all_effective_protos, all_sbl_inputs, all_block_raw_weights, _ = self.model(input_ids, gate_temperature=self.config.gate_sigmoid_temperature)
        main_loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, self.config.model.vocab_size), labels[:, 1:].contiguous().view(-1), ignore_index=-100)
        
        if not torch.isfinite(main_loss): return None, None, None, None

        # --- Adaptive Weight Decay for mu_weight ---
        spl_module_type = type(self.model.blocks[0].attn.sbl_qkv)
        spl_layers = [m for m in self.model.modules() if isinstance(m, spl_module_type)]
        
        # Safely extract attributes with TorchScript compatibility
        mu_weights, gate_params_for_mu = [], []
        for layer in spl_layers:
            mu_weight, _, _, gate_param = self._get_spl_module_attributes(layer)
            mu_weights.append(mu_weight)
            gate_params_for_mu.append(gate_param)

        adaptive_decay_loss_main = _calculate_adaptive_decay(mu_weights, gate_params_for_mu)
        main_loss += adaptive_decay_loss_main

        main_loss.backward(retain_graph=True)
        
        proto_weights, gate_params, mu_grads, sbl_inputs = [], [], [], []
        
        # We need a full list of grads (including None) to correctly map to raw_weights
        mu_grads_all = []
        for layer in spl_layers:
            mu_weight, _, _, _ = self._get_spl_module_attributes(layer)
            mu_grads_all.append(mu_weight.grad)

        for i, layer in enumerate(spl_layers):
            if mu_grads_all[i] is not None:
                _, proto_weight, _, gate_param = self._get_spl_module_attributes(layer)
                proto_weights.append(proto_weight)
                gate_params.append(gate_param)
                mu_grads.append(mu_grads_all[i].detach())
                sbl_inputs.append(all_sbl_inputs[i])

        if mu_grads:
            normalized_gate_params = [torch.tanh(gp.detach()) for gp in gate_params]
            
            # Collect raw_weights for active SPL layers
            active_raw_weights = []
            mu_grad_indices = [i for i, grad in enumerate(mu_grads_all) if grad is not None]
            
            # This logic assumes a 1-to-1 correspondence which is correct.
            # `all_block_raw_weights` is a flat list of all raw_weights from all SPLs in execution order.
            # `mu_grads_all` is a list of gradients (or None) for all SPLs in the same order.
            active_raw_weights = [all_block_raw_weights[i] for i in mu_grad_indices]

            proto_loss_total, gate_loss_total = self._calculate_meta_loss(
                proto_weights,
                gate_params,
                mu_grads,
                sbl_inputs,
                active_raw_weights
            )

            adaptive_decay_loss_meta = _calculate_adaptive_decay(proto_weights, gate_params)

            meta_loss = self.config.w_proto * proto_loss_total + self.config.w_gate * gate_loss_total + adaptive_decay_loss_meta
            
            if torch.isfinite(meta_loss):
                meta_loss.backward()
        else:
            proto_loss_total, gate_loss_total, adaptive_decay_loss_meta = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        # --- SGM (Sparse Gradient Masking) Implementation ---
        # Apply unified plasticity weight to mu_weight and mu_bias gradients
        
        spl_module_type = type(self.model.blocks[0].attn.sbl_qkv)
        spl_layers = [m for m in self.model.modules() if isinstance(m, spl_module_type)]
        
        for layer in spl_layers:
            mu_weight, proto_weight, mu_bias, gate_param = self._get_spl_module_attributes(layer)
            
            if mu_weight.grad is not None:
                normalized_gate = torch.tanh(gate_param.detach())
                w_plasticity = F.relu(1.0 - normalized_gate)
                
                mu_weight.grad = mu_weight.grad * w_plasticity.unsqueeze(1)
                if mu_bias.grad is not None:
                    mu_bias.grad = mu_bias.grad * w_plasticity

        torch.nn.utils.clip_grad_norm_([p for p in self.optimizer_main.param_groups[0]['params'] if p.grad is not None], 1.0)
        torch.nn.utils.clip_grad_norm_([p for p in self.optimizer_meta.param_groups[0]['params'] if p.grad is not None], 1.0)
        
        self.optimizer_main.step()
        self.optimizer_meta.step()

        with torch.no_grad():
            token_acc = self._calculate_token_accuracy(logits, labels).item()
            pi_score = torch.exp(-main_loss).item()
            
            # Calculate tau only for active tokens, consistent with main_loss and token_acc
            logits_for_tau = logits[:, :-1, :]
            labels_for_tau = labels[:, 1:]
            
            masked_labels_for_tau = labels_for_tau.contiguous().view(-1)
            active_mask_for_tau = masked_labels_for_tau != -100
            
            if active_mask_for_tau.any():
                active_logits_for_tau = logits_for_tau.contiguous().view(-1, logits_for_tau.size(-1))[active_mask_for_tau]
                probs_for_tau = F.softmax(active_logits_for_tau, dim=-1)
                tau = -torch.sum(probs_for_tau * torch.log(probs_for_tau + 1e-9), dim=-1).mean().item()
            else:
                tau = 0.0
            
            seq_len = input_ids.shape[1]
            token_counts = torch.bincount(input_ids.flatten(), minlength=self.tokenizer.vocab_size)
            token_probs = token_counts / seq_len if seq_len > 0 else torch.zeros_like(token_counts, dtype=torch.float)
            token_probs = token_probs[token_probs > 0]
            seq_entropy = -torch.sum(token_probs * torch.log2(token_probs)).item() if token_probs.numel() > 0 else 0.0

            activation_rate_avg = torch.cat([rw.view(-1) for rw in all_block_raw_weights]).gt(0).float().mean().item() if all_block_raw_weights else 0.0
            
            num_spl_per_block = 4
            act_rate_l0, act_rate_l_mid, act_rate_ln = 0.0, 0.0, 0.0
            num_layers = self.config.model.num_layers
            if all_block_raw_weights and len(all_block_raw_weights) == num_layers * num_spl_per_block:
                first_block_weights = all_block_raw_weights[:num_spl_per_block]
                act_rate_l0 = torch.cat([rw.view(-1) for rw in first_block_weights]).gt(0).float().mean().item()
                
                mid_layer_start_idx = (num_layers // 2 - 1) * num_spl_per_block
                mid_layer_end_idx = (num_layers // 2 + 1) * num_spl_per_block
                mid_block_weights = all_block_raw_weights[mid_layer_start_idx:mid_layer_end_idx]
                act_rate_l_mid = torch.cat([rw.view(-1) for rw in mid_block_weights]).gt(0).float().mean().item()

                last_block_weights = all_block_raw_weights[-num_spl_per_block:]
                act_rate_ln = torch.cat([rw.view(-1) for rw in last_block_weights]).gt(0).float().mean().item()

            all_proto_tensors = [p for block_protos in all_effective_protos for p in block_protos.values()]
            avg_proto_norm = torch.stack([torch.norm(p, p=2) for p in all_proto_tensors]).mean().item() if all_proto_tensors else 0.0
            
            all_gate_params = [p for name, p in self.model.named_parameters() if 'gate_param' in name]
            if all_gate_params:
                raw_gates_flat = torch.cat([p.detach().view(-1) for p in all_gate_params]).float()
                normalized_gates_flat = raw_gates_flat
                
                top10_threshold_norm = torch.quantile(normalized_gates_flat, 0.9)
                top10_mask_norm = normalized_gates_flat >= top10_threshold_norm
                top10_mean = normalized_gates_flat[top10_mask_norm].mean().item() if top10_mask_norm.any() else 0.0
                avg_gate_val = normalized_gates_flat.mean().item()
            else:
                top10_mean, avg_gate_val = 0.0, 0.0

            metrics = {
                "main_loss": main_loss.item(),
                "proto_loss": proto_loss_total.item(),
                "gate_loss": gate_loss_total.item(),
                "decay_loss": (adaptive_decay_loss_main + adaptive_decay_loss_meta).item(),
                "token_acc": token_acc, "pi_score": pi_score,
                "activation_rate_avg": activation_rate_avg,
                "activation_rate_l0": act_rate_l0,
                "activation_rate_l_mid": act_rate_l_mid,
                "activation_rate_ln": act_rate_ln,
                "seq_len": float(seq_len), 
                "tau": tau, 
                "seq_entropy": seq_entropy,
                "avg_proto_norm": avg_proto_norm,
                "avg_gate_val": avg_gate_val,
                "top10_gate_mean": top10_mean,
            }
        
        elapsed_time = time.time() - start_time
        if self.global_step % self.config.log_interval == 0:
            self.observer.log_step(epoch, self.global_step, mini_task_idx, metrics, elapsed_time)
            self._save_checkpoint(mini_task_idx, view_idx)

        if self.global_step > 0 and self.global_step % self.config.eval_interval == 0:
            self.evaluator.run(self.eval_loader, mini_task_idx, self.global_step)

        self.global_step += 1
        torch.cuda.empty_cache()
        return main_loss.item(), token_acc, tau, all_block_raw_weights

    def _train_epoch(self, epoch: int):
        self.model.train()
        num_mini_tasks = len(self.train_loader.dataset)
        
        for mini_task_idx in range(self.start_task_idx, num_mini_tasks):
            mini_task = self.train_loader.dataset[mini_task_idx]
            last_view_activation_pattern: torch.Tensor | None = None
            mini_task_final_raw_weights = None
            mini_task_final_seq_len = 0
            
            start_view = self.start_view_idx if mini_task_idx == self.start_task_idx else 0
            for view_idx in range(start_view, 8):
                inner_step = 0
                MAX_INNER_STEPS = 500
                view_last_raw_weights = None
                
                while inner_step < MAX_INNER_STEPS:
                    loss, acc, tau, raw_weights = self._run_step(mini_task, view_idx, epoch, mini_task_idx)
                    if raw_weights: view_last_raw_weights = raw_weights
                    
                    if loss is None:
                        if inner_step == 0:
                            self.console.print(f"[yellow]Skipping mini-task {mini_task_idx} view {view_idx} (too long).[/yellow]")
                            break
                        continue

                    if loss is not None and acc is not None and tau is not None and loss <= 0.1 and acc >= 1.0 and tau <= 0.1:
                        if view_last_raw_weights:
                            flat_pattern = torch.cat([rw.view(-1) for rw in view_last_raw_weights]).gt(0)
                            overlap_info = " | Overlap: N/A"
                            if last_view_activation_pattern is not None and last_view_activation_pattern.shape == flat_pattern.shape:
                                intersection = (last_view_activation_pattern & flat_pattern).sum().item()
                                union = (last_view_activation_pattern | flat_pattern).sum().item()
                                jaccard = intersection / union if union > 0 else 0
                                overlap_info = f" | Overlap: {jaccard:.2%}"
                            self.console.print(f"Mini-task {mini_task_idx} view {view_idx} CONVERGED in {inner_step + 1} steps.{overlap_info}")
                            last_view_activation_pattern = flat_pattern

                        if view_last_raw_weights:
                            mini_task_final_raw_weights = view_last_raw_weights
                            mini_task_final_seq_len = view_last_raw_weights[0].shape[1]
                        break
                        
                    inner_step += 1

                if inner_step == MAX_INNER_STEPS:
                    self.console.print(f"[red]Mini-task {mini_task_idx} view {view_idx} hit MAX_INNER_STEPS.[/red]")
                
                # --- Expert Reinitialization (after view completion) ---
                # Final check for dead prototypes after each view completes
                reinit_count = self._reinitialize_dead_prototypes()
                if reinit_count > 0:
                    self.console.print(f"[bold yellow]Reinitialized {reinit_count} dead prototypes after completing view {view_idx}.[/bold yellow]")
            
            # --- Inter-Mini-Task Profile Logic ---
            if mini_task_final_raw_weights:
                current_profile = self._calculate_activation_profile(mini_task_final_raw_weights, mini_task_final_seq_len)
                if self.last_task_activation_profile is not None and self.last_task_activation_profile.shape == current_profile.shape:
                    cos_sim = F.cosine_similarity(self.last_task_activation_profile.unsqueeze(0), current_profile.unsqueeze(0)).item()
                    l2_dist = torch.norm(self.last_task_activation_profile - current_profile, p=2).item()
                    self.console.print(f"[bold cyan]Mini-Task Profile Diff | Cosine Sim: {cos_sim:.2%} | L2 Dist: {l2_dist:.2f}[/bold cyan]")
                self.last_task_activation_profile = current_profile

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
