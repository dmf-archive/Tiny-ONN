import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.nn as nn
from rich.console import Console
from torch.utils.data import DataLoader

from .config import TrainConfig
from .consistency import ConsistencyTools
from .data import ArcCollator, ArcDataset, GridDeserializer, GridSerializer
from .evaluation import EvaluationStep
from .model import ArcTransformer
from .observer import Observer
from .tokenizer import ArcColorTokenizer


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

        train_dataset = ArcDataset(data_path=config.data.data_path, split="training")
        eval_dataset = ArcDataset(data_path=config.data.data_path, split="evaluation")

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

        main_params = []
        meta_params = []
        for name, p in self.model.named_parameters():
            if 'proto_weight' in name or 'gate_param' in name:
                meta_params.append(p)
            else:
                main_params.append(p)

        self.optimizer_main = torch.optim.AdamW(main_params, lr=self.config.lr_main, weight_decay=self.config.weight_decay)
        self.optimizer_meta = torch.optim.AdamW(meta_params, lr=self.config.lr_meta, weight_decay=self.config.weight_decay)

        self.consistency_tools = ConsistencyTools()
        self.evaluator = EvaluationStep(self.model, self.serializer, self.deserializer, self.observer, self.device)

        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.global_step = 0

    def _save_checkpoint(self):
        state = {
            'step': self.global_step,
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
        latest_checkpoint = checkpoints[0]
        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_main.load_state_dict(checkpoint['optimizer_main_state_dict'])
        self.optimizer_meta.load_state_dict(checkpoint['optimizer_meta_state_dict'])
        self.global_step = checkpoint['step']
        self.console.print(f"[bold blue]Loaded checkpoint from {latest_checkpoint} at step {self.global_step}[/bold blue]")

    def _calculate_token_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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


    def _run_step(self, task_data: dict[str, Any], view_idx: int, epoch: int):
        start_time = time.time()
        transforms = self.consistency_tools.get_transforms()
        input_grid = torch.tensor(task_data['test'][0]['input'])
        output_grid = torch.tensor(task_data['test'][0]['output'])
        input_grid_aug, output_grid_aug = transforms[view_idx](input_grid), transforms[view_idx](output_grid)

        view_task_data = {'train': task_data['train'], 'test': [{'input': input_grid_aug.tolist(), 'output': output_grid_aug.tolist()}]}
        input_ids_list, labels_list = self.serializer.serialize_task_with_context(view_task_data)

        if len(input_ids_list) > self.config.model.max_position_embeddings: return None

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)
        labels = torch.tensor([labels_list], dtype=torch.long, device=self.device)

        self.model.train()
        self.optimizer_main.zero_grad()
        self.optimizer_meta.zero_grad()

        logits, _, _, _, _, all_block_raw_weights, _ = self.model(input_ids)
        main_loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, self.config.model.vocab_size), labels[:, 1:].contiguous().view(-1), ignore_index=-100)

        if not torch.isfinite(main_loss): return None
        
        main_loss.backward(create_graph=True)

        proto_loss_total = torch.tensor(0.0, device=self.device)
        gate_loss_total = torch.tensor(0.0, device=self.device)
        surprise_norms_for_smp = []
        
        spl_module_type = type(self.model.blocks[0].attn.sbl_qkv)
        spl_layers_in_order = [m for m in self.model.modules() if isinstance(m, spl_module_type)]
        
        for spl_layer in spl_layers_in_order:
            if spl_layer.mu_weight.grad is not None:
                # SAPS V5: 梯度强度作为惊奇度 - 使用绝对值创建对称排斥场
                surprise_map_raw = spl_layer.mu_weight.grad.detach()
                surprise_map = torch.abs(surprise_map_raw)  # 关键：统一扰动强度
                
                # 原型对齐：推动p与|S|反向
                proto_loss = -F.cosine_similarity(spl_layer.proto_weight, -surprise_map, dim=-1).mean()
                proto_loss_total += proto_loss
                
                # 门控校准：推动g与|S|的L1范数正相关
                surprise_norm = torch.norm(surprise_map, p=1, dim=-1)  # L1范数：sum(|S_j|)
                gate_loss = -(spl_layer.gate_param * surprise_norm).mean()
                gate_loss_total += gate_loss
                surprise_norms_for_smp.append(surprise_norm)

        meta_loss = self.config.w_proto * proto_loss_total + self.config.w_gate * gate_loss_total
        if torch.isfinite(meta_loss):
            meta_loss.backward()
            self.optimizer_meta.step()

        with torch.no_grad():
            for i, spl_layer in enumerate(spl_layers_in_order):
                if i < len(surprise_norms_for_smp):
                    s_norm = surprise_norms_for_smp[i]
                    num_total = s_norm.shape[0]
                    # SAPS V4: 熵驱动的自适应可塑性
                    probs = F.softmax(logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
                    max_entropy = torch.log(torch.tensor(self.config.model.vocab_size, device=self.device))
                    p_dyn = entropy / max_entropy
                    num_winners = int(num_total * p_dyn)
                    
                    if num_winners < num_total:
                        winner_indices = s_norm.argsort()[:num_winners]
                        grad_mask = torch.zeros_like(s_norm)
                        grad_mask[winner_indices] = 1.0
                        
                        if spl_layer.mu_weight.grad is not None:
                            spl_layer.mu_weight.grad *= grad_mask.unsqueeze(-1)
                        if spl_layer.mu_bias.grad is not None:
                            spl_layer.mu_bias.grad *= grad_mask

        torch.nn.utils.clip_grad_norm_([p for p in self.optimizer_main.param_groups[0]['params'] if p.grad is not None], 1.0)
        self.optimizer_main.step()

        with torch.no_grad():
            complexity_cost = sum(p.grad.norm(p=2).item() ** 2 for p in self.optimizer_main.param_groups[0]['params'] if p.grad is not None and p.grad is not None) ** 0.5
            pi_score = torch.exp(-(main_loss + complexity_cost)).item()
            
            seq_len = input_ids.shape[1]
            token_counts = torch.bincount(input_ids.flatten(), minlength=self.tokenizer.vocab_size)
            token_probs = token_counts / seq_len if seq_len > 0 else torch.zeros_like(token_counts, dtype=torch.float)
            token_probs = token_probs[token_probs > 0]
            seq_entropy = -torch.sum(token_probs * torch.log2(token_probs)).item() if token_probs.numel() > 0 else 0.0

            token_acc = self._calculate_token_accuracy(logits, labels).item()
            
            all_raw_weights = [item for sublist in all_block_raw_weights for item in sublist]
            activation_rate_avg = torch.cat([rw.view(-1) for rw in all_raw_weights]).gt(0).float().mean().item() if all_raw_weights else 0.0
            act_rate_l0 = torch.cat([rw.view(-1) for rw in all_block_raw_weights[0]]).gt(0).float().mean().item() if all_block_raw_weights and all_block_raw_weights[0] else 0.0
            act_rate_ln = torch.cat([rw.view(-1) for rw in all_block_raw_weights[-1]]).gt(0).float().mean().item() if all_block_raw_weights and all_block_raw_weights[-1] else 0.0
            
            all_proto_weights = [m.proto_weight for m in self.model.modules() if hasattr(m, 'proto_weight')]
            proto_norm = torch.mean(torch.stack([p.norm(p=2) for p in all_proto_weights])).item() if all_proto_weights else 0.0
            avg_proto = torch.mean(torch.stack([p.mean() for p in all_proto_weights])).item() if all_proto_weights else 0.0
            
            all_gate_params = [m.gate_param for m in self.model.modules() if hasattr(m, 'gate_param')]
            avg_gate = torch.mean(torch.stack([p.mean() for p in all_gate_params])).item() if all_gate_params else 0.0

            # 计算元损失相关指标
            meta_loss_val = meta_loss.item() if torch.isfinite(meta_loss) else 0.0
            gate_loss_val = gate_loss_total.item() if torch.isfinite(gate_loss_total) else 0.0
            proto_loss_val = proto_loss_total.item() if torch.isfinite(proto_loss_total) else 0.0

            metrics = {
                "main_loss": main_loss.item(), "proto_loss": proto_loss_val,
                "gate_loss": gate_loss_val, "meta_loss": meta_loss_val,
                "token_acc": token_acc, "pi_score": pi_score, "seq_entropy": seq_entropy,
                "proto_norm": proto_norm, "avg_proto": avg_proto, "avg_gate": avg_gate,
                "activation_rate_avg": activation_rate_avg,
                "activation_rate_l0": act_rate_l0,
                "activation_rate_ln": act_rate_ln,
                "seq_len": float(seq_len),
                "p_dyn": p_dyn.item(),
            }

        elapsed_time = time.time() - start_time
        if self.global_step % self.config.log_interval == 0:
            self.observer.log_step(epoch, self.global_step, metrics, elapsed_time)
            self._save_checkpoint()

        if self.global_step > 0 and self.global_step % self.config.eval_interval == 0:
            self.evaluator.run(self.eval_loader, self.global_step, quick_eval=True)

        self.global_step += 1
        torch.cuda.empty_cache()
        return main_loss.item()

    def _train_epoch(self, epoch: int):
        self.model.train()
        num_tasks = len(self.train_loader.dataset)
        for task_idx in range(num_tasks):
            task_data = self.train_loader.dataset[task_idx]

            best_loss = float('inf')
            steps_without_improvement = 0
            inner_step = 0
            MAX_INNER_STEPS = 200

            while inner_step < MAX_INNER_STEPS:
                view_idx = inner_step % 8
                current_loss = self._run_step(task_data, view_idx, epoch)

                if current_loss is None:
                    if inner_step == 0:
                        self.console.print(f"[yellow]Skipping task {task_idx} (too long).[/yellow]")
                        break
                    inner_step += 1
                    continue

                if current_loss < 0.1:
                    self.console.print(f"Task {task_idx} CONVERGED with loss {current_loss:.4f} in {inner_step + 1} steps.")
                    break
                
                if current_loss < best_loss:
                    best_loss = current_loss
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1
                
                if steps_without_improvement >= 16:
                    self.console.print(f"Task {task_idx} STOPPED early after {inner_step + 1} steps (no improvement).")
                    break
                
                inner_step += 1

            if inner_step == MAX_INNER_STEPS:
                self.console.print(f"[red]Task {task_idx} hit MAX_INNER_STEPS, moving on.[/red]")

        self.console.print(f"[bold yellow]End of Epoch {epoch}.[/bold yellow]")

    def train(self):
        self._load_checkpoint()
        self.console.print("[bold green]Starting Training...[/bold green]")
        for epoch in range(self.config.num_epochs):
            self._train_epoch(epoch)
        self.console.print("[bold green]Training Finished.[/bold green]")

def main():
    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
