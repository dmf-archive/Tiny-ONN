import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
import math
import time

@dataclass
class DynNSAConfig:
    batch_size: int = 8
    vocab_size: int = 1000
    seq_len: int = 256
    hidden_size: int = 128
    num_heads: int = 4
    depth: int = 4
    dropout: float = 0.1

    window_size: int = 32
    block_size: int = 16
    num_selected_blocks: int = 2

    def __post_init__(self):
        self.head_size = self.hidden_size // self.num_heads
        self.num_blocks = self.seq_len // self.block_size

class DynNSAttention(nn.Module):
    def __init__(self, config: DynNSAConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.g_proj = nn.Linear(config.hidden_size, 3)
        self.router_proj = nn.Linear(config.hidden_size, config.num_blocks)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('causal_mask', torch.triu(torch.ones(config.seq_len, config.seq_len, dtype=torch.bool), diagonal=1))

    def forward(self, hidden_states: torch.Tensor, top_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = hidden_states.size()
        H = self.config.num_heads
        HS = self.config.head_size

        router_logits = self.router_proj(hidden_states)

        q = self.q_proj(hidden_states).view(B, T, H, HS).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, H, HS).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, H, HS).transpose(1, 2)

        # Entropy Calculation (inside the computation graph)
        k_blocks_for_entropy = k.view(B, H, self.config.num_blocks, self.config.block_size, HS)
        block_entropies_list = []
        for i in range(self.config.num_blocks):
            k_block = k_blocks_for_entropy[:, :, i, :, :]
            attn_scores = torch.matmul(q, k_block.transpose(-2, -1)) / math.sqrt(HS)
            attn_probs = F.softmax(attn_scores, dim=-1)
            entropy = -(attn_probs * torch.log(attn_probs + 1e-9)).sum(dim=-1)
            block_entropies_list.append(entropy.unsqueeze(-1))
        block_entropies = torch.cat(block_entropies_list, dim=-1)

        gates = F.softmax(self.g_proj(hidden_states), dim=-1)
        g_sliding, g_selected, g_compressed = gates.chunk(3, dim=-1)

        # Branch 1: Sliding Window Attention
        sliding_window_mask = torch.zeros((T, T), dtype=torch.bool, device=hidden_states.device)
        if self.config.window_size > 0:
            sliding_window_mask.fill_diagonal_(False)
            sliding_window_mask = ~torch.tril(torch.triu(sliding_window_mask, -self.config.window_size), self.config.window_size-1)
        final_mask = torch.logical_or(self.causal_mask, sliding_window_mask)
        attn_sliding_out = F.scaled_dot_product_attention(q, k, v, attn_mask=final_mask.logical_not(), dropout_p=self.dropout.p if self.training else 0.0)

        # Prepare for Block-based Attention
        k_blocks = k.view(B, H, self.config.num_blocks, self.config.block_size, HS)
        v_blocks = v.view(B, H, self.config.num_blocks, self.config.block_size, HS)
        
        # Branch 2: Dynamic Selective Attention (using provided top_indices)
        num_selected_blocks = top_indices.shape[2]
        expanded_indices = top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.config.block_size, HS)
        
        selected_k = torch.gather(k_blocks, 2, expanded_indices)
        selected_v = torch.gather(v_blocks, 2, expanded_indices)
        
        selected_k = selected_k.reshape(B, H, num_selected_blocks * self.config.block_size, HS)
        selected_v = selected_v.reshape(B, H, num_selected_blocks * self.config.block_size, HS)

        attn_selected_out = F.scaled_dot_product_attention(q, selected_k, selected_v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0)

        # Branch 3: Compressed Attention
        k_compressed = k_blocks.mean(dim=3)
        v_compressed = v_blocks.mean(dim=3)
        attn_compressed_out = F.scaled_dot_product_attention(q, k_compressed, v_compressed, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0)
        
        # Gated Aggregation
        attn_sliding_out = attn_sliding_out.transpose(1, 2).reshape(B, T, C)
        attn_selected_out = attn_selected_out.transpose(1, 2).reshape(B, T, C)
        attn_compressed_out = attn_compressed_out.transpose(1, 2).reshape(B, T, C)

        output = (
            g_sliding * attn_sliding_out +
            g_selected * attn_selected_out +
            g_compressed * attn_compressed_out
        )
        
        return self.o_proj(output), router_logits, block_entropies


class MLP(nn.Module):
    def __init__(self, config: DynNSAConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.fc2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config: DynNSAConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.attn = DynNSAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, top_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_output, router_logits, block_entropies = self.attn(self.ln1(x), top_indices)
        x = x + attn_output
        x = x + self.mlp(self.ln2(x))
        return x, router_logits, block_entropies

class DynNSAModel(nn.Module):
    def __init__(self, config: DynNSAConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.seq_len, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.depth)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, top_indices_list: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        B, T = idx.size()
        tok_emb = self.token_embed(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_embed(pos)
        x = tok_emb + pos_emb

        all_router_logits = []
        all_block_entropies = []
        for i, block in enumerate(self.blocks):
            x, router_logits, block_entropies = block(x, top_indices_list[i])
            all_router_logits.append(router_logits)
            all_block_entropies.append(block_entropies)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, all_router_logits, all_block_entropies

def get_gating_inputs_and_loss(model: DynNSAModel, hidden_states: torch.Tensor, config: DynNSAConfig) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    
    entropies_list = []
    router_logits_list = []
    gating_loss = torch.tensor(0.0, device=hidden_states.device)
    
    x = hidden_states
    for block in model.blocks:
        x_ln = block.ln1(x)
        
        # Router logits are calculated *before* attention
        router_logits = block.attn.router_proj(x_ln) # Shape: (B, T, num_blocks)
        router_logits_list.append(router_logits)

        with torch.no_grad():
            q = block.attn.q_proj(x_ln).view(config.batch_size, config.seq_len, config.num_heads, config.head_size).transpose(1, 2)
            k = block.attn.k_proj(x_ln).view(config.batch_size, config.seq_len, config.num_heads, config.head_size).transpose(1, 2)
            k_blocks = k.view(config.batch_size, config.num_heads, config.num_blocks, config.block_size, config.head_size)
            
            block_entropies = []
            for i in range(config.num_blocks):
                k_block = k_blocks[:, :, i, :, :]
                attn_scores = torch.matmul(q, k_block.transpose(-2, -1)) / math.sqrt(config.head_size)
                attn_probs = F.softmax(attn_scores, dim=-1)
                entropy = -(attn_probs * torch.log(attn_probs + 1e-9)).sum(dim=-1)
                block_entropies.append(entropy.unsqueeze(-1))
            
            layer_entropies = torch.cat(block_entropies, dim=-1)
            entropies_list.append(layer_entropies)

        # Hybrid Gating Loss on a Per-Token basis
        B, T, NB = router_logits.shape
        H = layer_entropies.shape[1]
        
        # Reshape for per-token loss
        router_logits_flat = router_logits.view(B * T, NB)
        layer_entropies_mean_H = layer_entropies.mean(dim=1) # Average over heads -> (B, T, NB)
        layer_entropies_flat = layer_entropies_mean_H.view(B * T, NB)

        target_indices = torch.argmin(layer_entropies_flat, dim=-1)
        ce_loss = F.cross_entropy(router_logits_flat, target_indices)
        
        kl_loss = F.kl_div(
            F.log_softmax(router_logits_flat, dim=-1),
            F.log_softmax(-layer_entropies_flat, dim=-1),
            log_target=True,
            reduction='batchmean'
        )
        
        gating_loss += (0.5 * ce_loss + 0.5 * kl_loss)

        x = block(x, torch.randint(0, config.num_blocks, (config.batch_size, config.num_heads, config.num_blocks//2), device=x.device))

    return entropies_list, router_logits_list, gating_loss / len(model.blocks)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DynNSAConfig()
    print(f"Using device: {device}")

    model = DynNSAModel(config).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)
    
    batch_size = config.batch_size
    
    # Initial top_indices
    base_k = config.num_blocks // 2
    top_indices = torch.randint(0, config.num_blocks, (batch_size, config.num_heads, base_k), device=device)
    top_indices_list = [top_indices for _ in range(config.depth)]

    for step in range(3001):
        t0 = time.time()
        
        idx = torch.randint(0, config.vocab_size, (batch_size, config.seq_len), device=device)
        targets = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len), device=device)

        tok_emb = model.token_embed(idx)
        pos = torch.arange(0, config.seq_len, dtype=torch.long, device=idx.device)
        pos_emb = model.pos_embed(pos)
        initial_hidden_states = tok_emb + pos_emb
        
        surprise_matrices, _, gating_loss = get_gating_inputs_and_loss(model, initial_hidden_states, config)

        logits = model(idx, top_indices_list)
        main_loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
        
        combined_loss = main_loss + 10.0 * gating_loss

        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            new_top_indices_list = []
            total_entropy = 0.0
            for surprise in surprise_matrices:
                block_scores = -surprise.mean(dim=2) # score = -entropy
                avg_entropy = surprise.mean().item()
                total_entropy += avg_entropy

                base_k = config.num_blocks // 2
                # Calibrate the threshold and steepness
                k = max(base_k, int(config.num_blocks / (1 + math.exp(5.0 * (avg_entropy - 2.6)))))
                
                _, top_indices = torch.topk(block_scores, k, dim=-1)
                new_top_indices_list.append(top_indices)
            top_indices_list = new_top_indices_list
        
        t1 = time.time()
        
        if step % 10 == 0:
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                accuracy = (preds == targets).float().mean()
                k_val = top_indices_list[0].shape[2]
                avg_total_entropy = total_entropy / config.depth
            print(f"Step {step:4d} | Loss: {main_loss.item():.4f} | GateLoss: {gating_loss.item():.4f} | Acc: {accuracy.item():.4f} | k: {k_val} | Entropy: {avg_total_entropy:.4f} | Time: {(t1 - t0)*1000:.2f}ms")

if __name__ == "__main__":
    main()