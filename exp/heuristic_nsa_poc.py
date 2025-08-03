import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
import math
import time

@dataclass
class HeuristicNSAConfig:
    vocab_size: int = 1000
    seq_len: int = 256
    hidden_size: int = 128
    num_heads: int = 4
    depth: int = 4
    dropout: float = 0.1

    window_size: int = 32
    block_size: int = 64
    num_selected_blocks: int = 2

    def __post_init__(self):
        self.head_size = self.hidden_size // self.num_heads
        self.num_blocks = self.seq_len // self.block_size

class HeuristicNSAttention(nn.Module):
    def __init__(self, config: HeuristicNSAConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.g_proj = nn.Linear(config.hidden_size, 3)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('causal_mask', torch.triu(torch.ones(config.seq_len, config.seq_len, dtype=torch.bool), diagonal=1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, C = hidden_states.size()
        H = self.config.num_heads
        HS = self.config.head_size

        q = self.q_proj(hidden_states).view(B, T, H, HS).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, H, HS).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, H, HS).transpose(1, 2)

        gates = F.softmax(self.g_proj(hidden_states), dim=-1)
        g_sliding, g_selected, g_compressed = gates.chunk(3, dim=-1)

        # Branch 1: Sliding Window Attention
        sliding_window_mask = torch.zeros((T, T), dtype=torch.bool, device=hidden_states.device)
        if self.config.window_size > 0:
            sliding_window_mask.fill_diagonal_(False)
            sliding_window_mask = ~torch.tril(torch.triu(sliding_window_mask, -self.config.window_size), self.config.window_size-1)
        final_mask = torch.logical_or(self.causal_mask, sliding_window_mask)
        attn_sliding_out = F.scaled_dot_product_attention(q, k, v, attn_mask=final_mask.logical_not(), dropout_p=self.dropout.p if self.training else 0.0)

        # Branch 2: Heuristic Selective Attention
        k_blocks = k.view(B, H, self.config.num_blocks, self.config.block_size, HS)
        v_blocks = v.view(B, H, self.config.num_blocks, self.config.block_size, HS)
        k_block_reps = k_blocks.mean(dim=3)
        
        q_expanded = q.unsqueeze(3)
        sim = F.cosine_similarity(q_expanded, k_block_reps.unsqueeze(2), dim=-1)
        global_scores = sim.mean(dim=2)
        
        _, top_indices = torch.topk(global_scores, self.config.num_selected_blocks, dim=-1)
        
        top_indices = top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.config.block_size, HS)
        
        selected_k = torch.gather(k_blocks, 2, top_indices)
        selected_v = torch.gather(v_blocks, 2, top_indices)
        
        selected_k = selected_k.reshape(B, H, self.config.num_selected_blocks * self.config.block_size, HS)
        selected_v = selected_v.reshape(B, H, self.config.num_selected_blocks * self.config.block_size, HS)

        attn_selected_out = F.scaled_dot_product_attention(q, selected_k, selected_v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0)

        # Branch 3: Compressed Attention
        k_compressed = k_block_reps
        v_compressed = v.view(B, H, self.config.num_blocks, self.config.block_size, HS).mean(dim=3)
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
        
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: HeuristicNSAConfig):
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
    def __init__(self, config: HeuristicNSAConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.attn = HeuristicNSAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class HeuristicNSAModel(nn.Module):
    def __init__(self, config: HeuristicNSAConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.seq_len, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.depth)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        tok_emb = self.token_embed(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_embed(pos)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = HeuristicNSAConfig()
    model = HeuristicNSAModel(config).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)
    
    batch_size = 8
    
    for step in range(101):
        t0 = time.time()
        
        idx = torch.randint(0, config.vocab_size, (batch_size, config.seq_len), device=device)
        targets = torch.randint(0, config.vocab_size, (batch_size, config.seq_len), device=device)

        logits = model(idx)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        t1 = time.time()
        
        if step % 10 == 0:
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                accuracy = (preds == targets).float().mean()
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.4f} | Time: {(t1 - t0)*1000:.2f}ms")

if __name__ == "__main__":
    main()