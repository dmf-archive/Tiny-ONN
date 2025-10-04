# %% [markdown]
# # Tiny-ONN ARC Kaggle Training Notebook
#
# This script contains a minimal, self-contained version of the training logic
# for the ARC experiment, designed to be easily converted into a Jupyter Notebook
# for headless execution on Kaggle.
#
# ## Workflow
#
# 1.  **Environment Setup**: Installs dependencies and adds the code dataset to `sys.path`.
# 2.  **Configuration**: Defines paths and hyperparameters for the Kaggle environment.
# 3.  **Imports**: Imports all necessary modules from the ARC code dataset.
# 4.  **Simplified Components**: Defines minimal versions of `Observer` and `Trainer` for headless operation.
# 5.  **Execution**: Initializes and runs the training loop.

# %%
# =============================================================================
# Cell 1: Environment Setup
# =============================================================================
print("Setting up environment...")

# 1.1. Install necessary dependencies
# In a real Kaggle environment, you would uncomment the following line:
# !pip install -q rich

# 1.2. Add our code package to the Python path
import sys
import os

# This path assumes you have created a Kaggle dataset named 'tiny-onn-arc-code'
# containing the 'exp' folder at its root.
# We use a placeholder for local execution.
if 'kaggle' in os.environ.get('KAGGLE_KERNEL_RUN_TYPE', ''):
    CODE_ROOT = '/kaggle/input/tiny-onn-arc-code'
    sys.path.append(CODE_ROOT)
    # Ensure the correct submodule path is also available
    sys.path.append(os.path.join(CODE_ROOT, 'exp'))
else:
    # For local development, assume the script is run from the project root
    # and the 'exp' directory is at the same level.
    # This allows us to test imports locally.
    sys.path.append(os.path.abspath('.'))


# 1.3. Import core libraries
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from torch.utils.data import DataLoader
from torch.nn.attention import SDPBackend, sdpa_kernel

# Imports for TPU support
try:
    import torch_xla.core.xla_model as xm
except ImportError:
    # This allows the script to run locally without TPU libraries
    pass

print("✅ Environment setup complete.")


# %%
# =============================================================================
# Cell 2: Configuration
# =============================================================================
print("Defining configuration...")

# Hardware-specific configuration
try:
    import torch_xla.core.xla_model as xm
    IS_TPU = True
except ImportError:
    IS_TPU = False

if IS_TPU:
    print("✅ TPU detected. Using bfloat16 and enhanced routing gain.")
    DTYPE = torch.bfloat16
    ROUTING_GAIN = 10.0
else:
    print("✅ No TPU detected. Using float32 for GPU/CPU.")
    DTYPE = torch.float32
    ROUTING_GAIN = 1.0

@dataclass
class ModelConfig:
    vocab_size: int = 16
    hidden_size: int = 768
    num_layers: int = 6
    max_position_embeddings: int = 4096
    d_ffn_factor: int = 1
    routing_gain: float = ROUTING_GAIN

@dataclass
class DataConfig:
    # Adjusted path for Kaggle single-file datasets
    challenges_path: str = "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json"
    solutions_path: str = "/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json" 
    batch_size: int = 1
    num_workers: int = 2

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Paths for Kaggle environment
    checkpoint_dir: str = "/kaggle/working/checkpoints/" if 'kaggle' in os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') else "notebook/checkpoints/"

    lr: float = 1e-3
    w_route_jsd: float = 1.1
    num_epochs: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    log_interval: int = 10
    max_checkpoints: int = 5

config = TrainConfig()

# Create checkpoint directory if it doesn't exist
os.makedirs(config.checkpoint_dir, exist_ok=True)

print(f"✅ Configuration defined. Device set to: {config.device}")


# %%
# =============================================================================
# Cell 3: Core Code Definitions
# =============================================================================
print("Defining core classes and functions...")

import json
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import math

class ArcColorTokenizer:
    def __init__(self):
        self.control_tokens = {
            "<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2, "problem": 3,
            "solution": 4, "\n": 5, "<im_start>": 6, "<im_end>": 7,
        }
        self.color_token_offset = len(self.control_tokens)
        self.num_colors = 10
        self.color_tokens = {str(i): self.color_token_offset + i for i in range(self.num_colors)}
        self.vocab = {**self.control_tokens, **self.color_tokens}
        self.inv_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}

    def color_to_token_id(self, color: int) -> int:
        return self.color_token_offset + color

    def token_id_to_color(self, token_id: int) -> int | None:
        if token_id < self.color_token_offset or token_id >= self.color_token_offset + self.num_colors:
            return None
        return token_id - self.color_token_offset

    @property
    def row_sep_token_id(self) -> int:
        return self.control_tokens["\n"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.control_tokens["<|pad|>"]

    @property
    def bos_token_id(self) -> int:
        return self.control_tokens["<|bos|>"]

    @property
    def eos_token_id(self) -> int:
        return self.control_tokens["<|eos|>"]

class GridSerializer:
    def __init__(self, tokenizer: ArcColorTokenizer):
        self.tokenizer = tokenizer

    def _serialize_grid(self, grid: list[list[int]]) -> tuple[list[int], list[tuple[int, int]]]:
        tokens, coords = [], []
        for r, row in enumerate(grid):
            if r > 0:
                tokens.append(self.tokenizer.row_sep_token_id)
                coords.append((-1, -1))
            for c, color in enumerate(row):
                tokens.append(self.tokenizer.color_to_token_id(color))
                coords.append((r, c))
        return tokens, coords

    def serialize_task(self, task_data: dict[str, Any]) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        full_ids: list[int] = [self.tokenizer.bos_token_id]
        full_coords: list[tuple[int, int]] = [(-1, -1)]
        
        im_start_id = self.tokenizer.vocab["<im_start>"]
        im_end_id = self.tokenizer.vocab["<im_end>"]

        # Build prompt section
        for pair in task_data["train"]:
            input_ids, input_coords = self._serialize_grid(pair["input"])
            output_ids, output_coords = self._serialize_grid(pair["output"])
            
            full_ids.extend([im_start_id] + input_ids + [im_end_id])
            full_coords.extend([(-1, -1)] + input_coords + [(-1, -1)])
            
            full_ids.extend([im_start_id] + output_ids + [im_end_id])
            full_coords.extend([(-1, -1)] + output_coords + [(-1, -1)])

        test_input_ids, test_input_coords = self._serialize_grid(task_data["test"][0]["input"])
        full_ids.extend([im_start_id] + test_input_ids + [im_end_id])
        full_coords.extend([(-1, -1)] + test_input_coords + [(-1, -1)])

        # Mark the start of the target sequence for labeling
        target_start_index = len(full_ids)

        # Build target section
        test_output_ids, test_output_coords = self._serialize_grid(task_data["test"][0]["output"])
        full_ids.extend([im_start_id] + test_output_ids + [im_end_id])
        full_coords.extend([(-1, -1)] + test_output_coords + [(-1, -1)])
        full_ids.append(self.tokenizer.eos_token_id)
        full_coords.append((-1, -1))

        # Create labels based on the full sequence, shifted by one
        labels = list(full_ids[1:]) + [-100]
        
        # Mask out the prompt section from the labels
        for i in range(target_start_index):
            labels[i] = -100
            
        return full_ids, labels, full_coords

class InMemoryArcDataset(Dataset):
    def __init__(self, challenges_path: str, solutions_path: str):
        with open(challenges_path, 'r') as f:
            challenges = json.load(f)
        with open(solutions_path, 'r') as f:
            solutions = json.load(f)

        self.tasks = []
        for task_id, task_data in challenges.items():
            if task_id in solutions:
                # Ensure the 'test' list exists and is not empty
                if "test" not in task_data:
                    task_data["test"] = [{}]
                elif not task_data["test"]:
                    task_data["test"].append({})
                
                # Merge the solution into the corresponding test case
                task_data["test"][0]["output"] = solutions[task_id][0]
                self.tasks.append(task_data)

        # Sort tasks by sequence length for efficiency, similar to the original implementation
        tokenizer_for_sorting = ArcColorTokenizer()
        serializer_for_sorting = GridSerializer(tokenizer_for_sorting)
        tasks_with_lengths = [(task, len(serializer_for_sorting.serialize_task(task)[0])) for task in self.tasks]
        self.tasks = [task for task, length in sorted(tasks_with_lengths, key=lambda x: x[1])]

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.tasks[idx]

class ArcCollator:
    def __init__(self, tokenizer: ArcColorTokenizer, max_len: int):
        self.tokenizer, self.serializer, self.max_len = tokenizer, GridSerializer(tokenizer), max_len

    @staticmethod
    def _calculate_sample_entropy(labels: list[int]) -> float:
        valid_labels = [l for l in labels if l != -100]
        if not valid_labels: return 0.0
        counts = torch.bincount(torch.tensor(valid_labels))
        probs = counts.float() / len(valid_labels)
        return -torch.sum(probs[probs > 0] * torch.log2(probs[probs > 0])).item()

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        all_input_ids, all_labels, all_coords, all_entropies = [], [], [], []
        for task_data in batch:
            input_ids, labels, coords = self.serializer.serialize_task(task_data)
            if len(input_ids) <= self.max_len:
                all_entropies.append(self._calculate_sample_entropy(labels))
                all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                all_labels.append(torch.tensor(labels, dtype=torch.long))
                all_coords.append(torch.tensor(coords, dtype=torch.long))
        if not all_input_ids: return {}
        return {
            "input_ids": pad_sequence(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "labels": pad_sequence(all_labels, batch_first=True, padding_value=-100),
            "coords": pad_sequence(all_coords, batch_first=True, padding_value=-1),
            "sample_entropy": torch.tensor(all_entropies, dtype=torch.float32),
        }

@torch.jit.script
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

@torch.jit.script
def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

@torch.jit.script
def mas_normalize(logits: torch.Tensor) -> torch.Tensor:
    max_abs_val = torch.max(torch.abs(logits), dim=-1, keepdim=True).values
    return F.relu(logits / (max_abs_val + 1e-9))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device: torch.device | None = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dim, self.max_position_embeddings, self.base = dim, max_position_embeddings, base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0).to(dtype=x.dtype), emb.sin().unsqueeze(0).to(dtype=x.dtype)

class SparseProtoLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.proto_weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.mu_bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        self.gate_param = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.mu_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.proto_weight, a=math.sqrt(5))
        nn.init.zeros_(self.mu_bias)
        nn.init.zeros_(self.gate_param)

    def forward(self, x: torch.Tensor, effective_proto: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        match_values = F.linear(x, effective_proto) / math.sqrt(x.size(-1))
        gate_logit = torch.matmul(x, self.gate_param.t())
        computation_output = F.linear(x, self.mu_weight, self.mu_bias)
        return computation_output, match_values, gate_logit

class DynamicInfiniteHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.spl_q = SparseProtoLinear(config.hidden_size, config.hidden_size, dtype=dtype)
        self.spl_k = SparseProtoLinear(config.hidden_size, config.hidden_size, dtype=dtype)
        self.spl_v = SparseProtoLinear(config.hidden_size, config.hidden_size, dtype=dtype)
        self.spl_o = SparseProtoLinear(config.hidden_size, config.hidden_size, dtype=dtype)

class DynamicInfiniteExpert(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        d_ffn = config.hidden_size * config.d_ffn_factor
        self.spl1 = SparseProtoLinear(config.hidden_size, d_ffn, dtype=dtype)
        self.spl2 = SparseProtoLinear(d_ffn, config.hidden_size, dtype=dtype)

class MoIETransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = DynamicInfiniteHeadAttention(config, dtype=dtype)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ffn = DynamicInfiniteExpert(config, dtype=dtype)
        self.routing_gain = config.routing_gain
        d_ffn = config.hidden_size * config.d_ffn_factor
        self.proto_transforms = nn.ModuleDict({
            "attn_q": nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype),
            "attn_k": nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype),
            "attn_v": nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype),
            "attn_o": nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype),
            "ffn_spl1": nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=dtype),
            "ffn_spl2": nn.Linear(d_ffn, d_ffn, bias=False, dtype=dtype),
        })
        self.proto_layernorms = nn.ModuleDict({name: nn.LayerNorm(p.in_features, eps=1e-5) for name, p in self.proto_transforms.items()})

    def forward(self, x: torch.Tensor, pos_emb: tuple, past_kv: tuple | None = None, prev_protos: dict | None = None) -> tuple:
        effective_protos = {}
        spl_modules = {"attn_q": self.attn.spl_q, "attn_k": self.attn.spl_k, "attn_v": self.attn.spl_v, "attn_o": self.attn.spl_o, "ffn_spl1": self.ffn.spl1, "ffn_spl2": self.ffn.spl2}
        for name, module in spl_modules.items():
            residual = self.proto_layernorms[name](self.proto_transforms[name](prev_protos[name])) if prev_protos else 0
            effective_protos[name] = module.proto_weight + residual
        
        ln1_out = self.ln1(x)
        c_q, mv_q, pc_q = self.attn.spl_q(ln1_out, effective_protos["attn_q"])
        c_k, mv_k, pc_k = self.attn.spl_k(ln1_out, effective_protos["attn_k"])
        c_v, mv_v, pc_v = self.attn.spl_v(ln1_out, effective_protos["attn_v"])
        
        all_masked, all_comp, all_raw, all_routing_logits, all_spl_inputs = [], [], [], [], []
        
        q, k, v = torch.zeros_like(c_q), torch.zeros_like(c_k), torch.zeros_like(c_v)
        comp_qkv, match_qkv, costs_qkv = [c_q, c_k, c_v], [mv_q, mv_k, mv_v], [pc_q, pc_k, pc_v]
        for i in range(3):
            cost_score = mas_normalize(costs_qkv[i])
            routing_logits = (match_qkv[i] - cost_score) * self.routing_gain
            raw_weights = mas_normalize(routing_logits)
            masked = (ln1_out + comp_qkv[i]) * raw_weights
            if i == 0: q = masked
            elif i == 1: k = masked
            else: v = masked
            all_masked.append(masked); all_comp.append(comp_qkv[i]); all_raw.append(raw_weights); all_routing_logits.append(routing_logits)
        
        cos, sin = pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_kv is not None: k, v = torch.cat([past_kv[0], k], dim=1), torch.cat([past_kv[1], v], dim=1)
        present_kv = (k, v)
        attn_out = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=past_kv is None).squeeze(1)

        c_o, mv_o, pc_o = self.attn.spl_o(attn_out, effective_protos["attn_o"])
        routing_logits_o = (mv_o - mas_normalize(pc_o)) * self.routing_gain
        m_o = (attn_out + c_o) * mas_normalize(routing_logits_o)
        x = x + m_o

        ln2_out = self.ln2(x)
        c1, mv1, pc1 = self.ffn.spl1(ln2_out, effective_protos["ffn_spl1"])
        routing_logits_f1 = (mv1 - mas_normalize(pc1)) * self.routing_gain
        h_act = F.relu((ln2_out + c1) * mas_normalize(routing_logits_f1))

        c2, mv2, pc2 = self.ffn.spl2(h_act, effective_protos["ffn_spl2"])
        routing_logits_f2 = (mv2 - mas_normalize(pc2)) * self.routing_gain
        m2 = (h_act + c2) * mas_normalize(routing_logits_f2)
        x_out = x + m2

        all_masked.extend([m_o, h_act, m2]); all_comp.extend([c_o, c1, c2]); all_raw.extend([mas_normalize(routing_logits_o), mas_normalize(routing_logits_f1), mas_normalize(routing_logits_f2)])
        all_routing_logits.extend([routing_logits_o, routing_logits_f1, routing_logits_f2]); all_spl_inputs.extend([ln1_out] * 3 + [attn_out, ln2_out, h_act])
        return x_out, all_masked, all_comp, all_raw, all_spl_inputs, all_routing_logits, present_kv, effective_protos

class ArcEmbedding(nn.Module):
    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.color_embedding = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.row_embedding = nn.Embedding(31, config.hidden_size, dtype=dtype)
        self.col_embedding = nn.Embedding(31, config.hidden_size, dtype=dtype)

    def forward(self, input_ids: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        color_embed = self.color_embedding(input_ids)
        row_embed = self.row_embedding(coords[..., 0].clamp(min=0, max=30))
        col_embed = self.col_embedding(coords[..., 1].clamp(min=0, max=30))
        pos_embed = row_embed + col_embed
        return color_embed + torch.where((coords[..., 0] == -1).unsqueeze(-1), torch.zeros_like(pos_embed), pos_embed)

class ArcTransformer(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device | str):
        super().__init__()
        self.config, self.device = config, device
        self.embedding = ArcEmbedding(config, dtype=DTYPE)
        self.rotary_emb = RotaryEmbedding(dim=config.hidden_size, max_position_embeddings=config.max_position_embeddings, device=device, dtype=DTYPE)
        self.blocks = nn.ModuleList([MoIETransformerBlock(config, dtype=DTYPE) for _ in range(config.num_layers)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=DTYPE)

    def forward(self, input_ids: torch.Tensor, coords: torch.Tensor, past_key_values: list | None = None, return_dict: bool = False):
        x = self.embedding(input_ids, coords)
        pos_emb = self.rotary_emb(x, seq_len=input_ids.size(1))
        past_key_values = past_key_values or [None] * len(self.blocks)
        all_masked, all_comp, all_spl_in, all_raw, all_protos, all_routing_logits, presents = [], [], [], [], [], [], []
        prev_protos = None
        for i, block in enumerate(self.blocks):
            x, masked, comp, raw, spl_inputs, routing_logits, present_kv, effective_protos = block(x, pos_emb, past_key_values[i], prev_protos)
            presents.append(present_kv); all_masked.extend(masked); all_comp.extend(comp); all_spl_in.extend(spl_inputs)
            all_raw.extend(raw); all_protos.append(effective_protos); all_routing_logits.extend(routing_logits); prev_protos = effective_protos
        logits = self.lm_head(x)
        if not return_dict: return logits, presents
        return {"logits": logits, "hidden_states": x, "masked_outputs": all_masked, "computation_outputs": all_comp, "proto_states": all_protos, "spl_inputs": all_spl_in, "raw_weights": all_raw, "routing_logits": all_routing_logits, "past_key_values": presents}

@torch.jit.script
def _jsd_from_distributions(p_dist_unnorm: torch.Tensor, q_dist_unnorm: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-9
    p_dist = p_dist_unnorm / (p_dist_unnorm.sum(dim=-1, keepdim=True) + epsilon)
    q_dist = q_dist_unnorm / (q_dist_unnorm.sum(dim=-1, keepdim=True) + epsilon)
    m_dist = 0.5 * (p_dist + q_dist)
    kl_p_m = torch.sum(p_dist * (torch.log(p_dist + epsilon) - torch.log(m_dist + epsilon)), dim=-1)
    kl_q_m = torch.sum(q_dist * (torch.log(q_dist + epsilon) - torch.log(m_dist + epsilon)), dim=-1)
    return (0.5 * kl_p_m + 0.5 * kl_q_m).mean()

@torch.jit.script
def _augment_and_map_kernel(grids: list[torch.Tensor], transform_idx: int, color_map: torch.Tensor) -> list[torch.Tensor]:
    transformed_grids = []
    for x in grids:
        if transform_idx == 0: transformed_x = x
        elif transform_idx == 1: transformed_x = torch.rot90(x, 1, [0, 1])
        elif transform_idx == 2: transformed_x = torch.rot90(x, 2, [0, 1])
        elif transform_idx == 3: transformed_x = torch.rot90(x, 3, [0, 1])
        elif transform_idx == 4: transformed_x = torch.flip(x, [0])
        elif transform_idx == 5: transformed_x = torch.flip(x, [1])
        elif transform_idx == 6: transformed_x = torch.transpose(x, 0, 1)
        else: transformed_x = torch.rot90(torch.flip(x, [0]), 1, [0, 1])
        transformed_grids.append(color_map[transformed_x])
    return transformed_grids

class LearningDynamics:
    def __init__(self, config: TrainConfig, model: nn.Module, computation_params: list, routing_params_with_names: list[tuple[str, nn.Parameter]]):
        self.config, self.model = config, model
        self.computation_params, self.routing_params_with_names = computation_params, routing_params_with_names
        self.routing_params = [p for _, p in routing_params_with_names]
        self.optimizer_comp = torch.optim.AdamW(computation_params, lr=self.config.lr)
        self.optimizer_route = torch.optim.AdamW(self.routing_params, lr=self.config.lr)

    @staticmethod
    @torch.jit.script
    def _calculate_jsd_loss(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        return _jsd_from_distributions(F.relu(mas_normalize(p_logits)), mas_normalize(q_logits).detach())

    def compute_and_apply_gradients(self, main_loss: torch.Tensor, model_outputs: dict, device: torch.device) -> dict[str, Any]:
        self.optimizer_comp.zero_grad(); self.optimizer_route.zero_grad()
        computation_outputs, masked_outputs = model_outputs["computation_outputs"], model_outputs["masked_outputs"]
        mu_weights = [p for name, p in self.model.named_parameters() if "mu_weight" in name]
        params_to_grad = self.computation_params + computation_outputs + mu_weights
        all_grads = torch.autograd.grad(main_loss, params_to_grad, retain_graph=True, allow_unused=True)
        
        comp_grads = all_grads[:len(self.computation_params)]
        intermediate_grads = all_grads[len(self.computation_params):len(self.computation_params) + len(computation_outputs)]
        mu_weight_grads = all_grads[len(self.computation_params) + len(computation_outputs):]

        c_output_norms = [torch.norm(g, p=2, dim=(0, 1)) for g in intermediate_grads if g is not None]
        c_param_norms = [torch.norm(g, p=2, dim=-1) for g in mu_weight_grads if g is not None]
        i_effective_norms = [torch.norm(mo, p=2, dim=(0, 1)) for mo in masked_outputs if mo is not None]

        with torch.no_grad():
            all_goodness = []
            num_modules = min(len(i_effective_norms), len(c_output_norms), len(c_param_norms))
            for i in range(num_modules):
                benefit_eff, benefit_rel = mas_normalize(i_effective_norms[i]), mas_normalize(c_output_norms[i])
                synergistic_benefit = mas_normalize(benefit_eff * benefit_rel)
                learning_cost = mas_normalize(c_param_norms[i])
                all_goodness.append(F.relu(synergistic_benefit / (learning_cost + 1e-9)))

        for param, grad in zip(self.computation_params, comp_grads):
            if grad is not None: param.grad = grad.clone()
        
        meta_losses = [self._calculate_jsd_loss(logit, good) for logit, good in zip(model_outputs["routing_logits"], all_goodness) if logit.numel() > 0 and good.numel() > 0 and logit.shape[-1] == good.shape[-1]]
        if meta_losses:
            total_meta_loss = self.config.w_route_jsd * torch.stack(meta_losses).mean()
            meta_grads = torch.autograd.grad(total_meta_loss, self.routing_params, allow_unused=True)
            for param, grad in zip(self.routing_params, meta_grads):
                if grad is not None: param.grad = grad.clone()

        torch.nn.utils.clip_grad_norm_(self.computation_params, max_norm=1.0)
        if IS_TPU:
            xm.optimizer_step(self.optimizer_comp)
            xm.optimizer_step(self.optimizer_route)
        else:
            self.optimizer_comp.step()
            self.optimizer_route.step()
        return {"route_jsd_loss": torch.stack(meta_losses).mean() if meta_losses else torch.tensor(0.0)}

class MinimalObserver:
    def __init__(self, console: Console, config: TrainConfig):
        self.console, self.config = console, config

    def calculate_metrics(self, main_loss: torch.Tensor, model_outputs: dict, signals: dict) -> dict[str, float]:
        logits, labels = model_outputs["logits"], model_outputs["labels"]
        raw_weights = model_outputs.get("raw_weights", [])
        mask = labels[:, 1:] != -100
        active_logits = logits[:, :-1, :][mask]
        acc = (torch.argmax(active_logits, dim=-1) == labels[:, 1:][mask]).float().mean().item() if mask.any() else 0.0
        act_rates = [rw.gt(0).float().mean().item() for rw in raw_weights] if raw_weights else [0.0]
        routing_failure_rate = sum(torch.all(rw == 0, dim=-1).float().mean().item() for rw in raw_weights) / len(raw_weights) if raw_weights else 0.0
        
        flat_logits = torch.cat([rl.detach().float().view(-1) for rl in model_outputs.get("routing_logits", []) if rl.numel() > 0])
        metrics = {"main_loss": main_loss.item(), "token_acc": acc, "route_jsd_loss": signals.get("route_jsd_loss", torch.tensor(0.0)).item(),
                   "sample_entropy": model_outputs.get("sample_entropy", torch.tensor(0.0)).mean().item(), "seq_len": float(labels.shape[1]),
                   "activation_rate_avg": sum(act_rates) / len(act_rates) if act_rates else 0.0, "routing_failure_rate": routing_failure_rate}
        if flat_logits.numel() > 0:
            metrics.update({"gate_logit_avg": flat_logits.mean().item(), "gate_logit_sigma": flat_logits.std().item()})
        return metrics

    def log_step(self, epoch: int, step: int, task_idx: int, view_idx: int, metrics: dict, elapsed_time: float):
        steps_per_sec = 1 / elapsed_time if elapsed_time > 0 else float("inf")
        log_str = (f"E{epoch} S{step} T{task_idx} V{view_idx} | L({metrics['main_loss']:.3f}/{metrics['route_jsd_loss']:.4f}) | "
                   f"Acc: {metrics['token_acc']:.3f} | Act%: {metrics['activation_rate_avg']*100:.1f} | "
                   f"Fail: {metrics['routing_failure_rate']*100:.1f}% | Speed: {steps_per_sec:.2f} st/s")
        self.console.print(log_str)

class KaggleTrainer:
    def __init__(self, config: TrainConfig):
        self.config, self.device = config, torch.device(config.device)
        torch.manual_seed(config.seed)
        self.console = Console()
        self.observer = MinimalObserver(self.console, config)
        self.tokenizer, self.serializer = ArcColorTokenizer(), GridSerializer(ArcColorTokenizer())
        self._setup_data()
        self._setup_model_and_optimizer()
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.global_step, self.epoch, self.start_task_idx, self.start_view_idx = 0, 0, 0, 0

    def _setup_data(self):
        self.train_dataset = InMemoryArcDataset(
            challenges_path=self.config.data.challenges_path,
            solutions_path=self.config.data.solutions_path
        )
        collator = ArcCollator(self.tokenizer, max_len=self.config.model.max_position_embeddings)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            collate_fn=lambda x: x,  # We process one task at a time, so no collation needed at loader level
            num_workers=self.config.data.num_workers,
            shuffle=False, # We iterate sequentially
        )

    def _setup_model_and_optimizer(self):
        self.config.model.vocab_size = self.tokenizer.vocab_size
        self.model = ArcTransformer(self.config.model, device=self.device).to(self.device)
        computation_params = [p for name, p in self.model.named_parameters() if "proto_weight" not in name and "gate_param" not in name]
        routing_params_with_names = [(name, p) for name, p in self.model.named_parameters() if "proto_weight" in name or "gate_param" in name]
        self.dynamics = LearningDynamics(self.config, self.model, computation_params, routing_params_with_names)

    def _prepare_batch(self, task_data: dict, view_idx: int) -> dict[str, torch.Tensor] | None:
        grids_cpu_lists = [pair[k] for pair in task_data["train"] for k in ("input", "output")]
        grids_cpu_lists.extend([task_data["test"][0]["input"], task_data["test"][0]["output"]])
        all_colors = {c for grid in grids_cpu_lists for row in grid for c in row}
        active_colors = [c for c in all_colors if c != 0]
        color_map_cpu = torch.arange(10, dtype=torch.long)
        if len(active_colors) >= 2:
            c1, c2 = random.sample(active_colors, 2)
            color_map_cpu[c1], color_map_cpu[c2] = c2, c1
        augmented_grids = [g.tolist() for g in _augment_and_map_kernel([torch.tensor(g, dtype=torch.long) for g in grids_cpu_lists], view_idx, color_map_cpu)]
        
        transformed_train, ptr = [], 0
        for _ in task_data["train"]:
            transformed_train.append({"input": augmented_grids[ptr], "output": augmented_grids[ptr + 1]}); ptr += 2
        augmented_task = {"train": transformed_train, "test": [{"input": augmented_grids[ptr], "output": augmented_grids[ptr + 1]}]}
        
        ids, labels, coords = self.serializer.serialize_task(augmented_task)
        if len(ids) > self.config.model.max_position_embeddings: return None
        return {"input_ids": torch.tensor([ids], dtype=torch.long), "labels": torch.tensor([labels], dtype=torch.long), "coords": torch.tensor([coords], dtype=torch.long), "sample_entropy": torch.tensor([ArcCollator._calculate_sample_entropy(labels)], dtype=torch.float32)}

    def _train_step(self, batch: dict, epoch: int, task_idx: int, view_idx: int):
        start_time = time.time()
        self.model.train()
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION), torch.autocast(device_type=self.config.device, dtype=DTYPE):
            model_outputs = self.model(batch["input_ids"], coords=batch["coords"], return_dict=True)
            main_loss = F.cross_entropy(model_outputs["logits"][:, :-1, :].contiguous().view(-1, self.config.model.vocab_size), batch["labels"][:, 1:].contiguous().view(-1), ignore_index=-100)
        
        if not torch.isfinite(main_loss):
            self.console.print(f"[bold red]NaN detected in main_loss at step {self.global_step}. Aborting step.[/bold red]"); return None
        
        signals = self.dynamics.compute_and_apply_gradients(main_loss, model_outputs, self.device)
        model_outputs.update({"labels": batch["labels"], "sample_entropy": batch["sample_entropy"]})
        metrics = self.observer.calculate_metrics(main_loss, model_outputs, signals)
        
        # Only log and save checkpoints on the master process
        if not IS_TPU or xm.is_master_ordinal():
             if self.global_step % self.config.log_interval == 0:
                self.observer.log_step(epoch, self.global_step, task_idx, view_idx, metrics, time.time() - start_time)
                self._save_checkpoint(task_idx, view_idx)

        self.global_step += 1
        return metrics

    def _train_epoch(self, epoch: int):
        dataset = self.train_dataset
        for task_idx in range(self.start_task_idx, len(dataset)):
            task_data = dataset[task_idx]
            start_view = self.start_view_idx if task_idx == self.start_task_idx else 0
            for view_idx in range(start_view, 8):
                batch_cpu = self._prepare_batch(task_data, view_idx)
                if not batch_cpu:
                    if not IS_TPU or xm.is_master_ordinal():
                        self.console.print(f"[yellow]Skipping Task {task_idx} View {view_idx} due to excessive length.[/yellow]")
                    continue

                batch = {k: v.to(self.device) for k, v in batch_cpu.items()}
                
                converged = False
                for step in range(500):
                    metrics = self._train_step(batch, epoch, task_idx, view_idx)
                    if not metrics:
                        break
                    if metrics["main_loss"] <= 0.03 and metrics["token_acc"] >= 0.999:
                        if not IS_TPU or xm.is_master_ordinal():
                            self.console.print(f"Task {task_idx} view {view_idx} converged in {step + 1} steps.")
                        converged = True
                        break
                
                if not converged and (not IS_TPU or xm.is_master_ordinal()):
                    self.console.print(f"[red]Task {task_idx} view {view_idx} hit MAX_STEPS.[/red]")

            self.start_view_idx = 0
        self.start_task_idx, self.start_view_idx = 0, 0

    def train(self):
        self._load_checkpoint()
        self.console.print("[bold green]Starting Training...[/bold green]")
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch; self._train_epoch(epoch)
        self.console.print("[bold green]Training Finished.[/bold green]")

    def _save_checkpoint(self, task_idx: int, view_idx: int):
        state = {"epoch": self.epoch, "step": self.global_step, "task_idx": task_idx, "view_idx": view_idx,
                 "model_state_dict": self.model.state_dict(), "optimizer_comp_state_dict": self.dynamics.optimizer_comp.state_dict(),
                 "optimizer_route_state_dict": self.dynamics.optimizer_route.state_dict()}
        path = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        torch.save(state, path)
        # Rotate checkpoints
        ckpts = sorted(self.checkpoint_dir.glob("*.pt"), key=os.path.getmtime)
        if len(ckpts) > self.config.max_checkpoints:
            os.remove(ckpts[0])

    def _load_checkpoint(self):
        # Priority 1: Check working directory for existing session checkpoints
        ckpts = sorted(self.checkpoint_dir.glob("*.pt"), key=os.path.getmtime, reverse=True)
        if ckpts:
            try:
                ckpt = torch.load(ckpts[0], map_location=self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.dynamics.optimizer_comp.load_state_dict(ckpt["optimizer_comp_state_dict"])
                self.dynamics.optimizer_route.load_state_dict(ckpt["optimizer_route_state_dict"])
                self.global_step, self.epoch, self.start_task_idx, self.start_view_idx = ckpt["step"], ckpt["epoch"], ckpt["task_idx"], ckpt["view_idx"]
                self.console.print(f"[bold green]Loaded session checkpoint from {ckpts[0]} at step {self.global_step}.[/bold green]")
                return
            except Exception as e:
                self.console.print(f"[bold red]Failed to load session checkpoint {ckpts[0]}: {e}. Starting from scratch.[/bold red]")
                return

        # Priority 2: If no session checkpoints, try to load the initial checkpoint from input
        initial_ckpt_path = Path("/kaggle/input/tiny-onn-arc/pytorch/default/1/checkpoint_21440.pt")
        if initial_ckpt_path.is_file():
            try:
                ckpt = torch.load(initial_ckpt_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.dynamics.optimizer_comp.load_state_dict(ckpt["optimizer_comp_state_dict"])
                self.dynamics.optimizer_route.load_state_dict(ckpt["optimizer_route_state_dict"])
                self.global_step, self.epoch, self.start_task_idx, self.start_view_idx = ckpt["step"], ckpt["epoch"], ckpt["task_idx"], ckpt["view_idx"]
                self.console.print(f"[bold green]Loaded initial checkpoint from {initial_ckpt_path} at step {self.global_step}.[/bold green]")
                return
            except Exception as e:
                self.console.print(f"[bold red]Failed to load initial checkpoint {initial_ckpt_path}: {e}. Starting from scratch.[/bold red]")
                return

        # Final fallback
        self.console.print("[yellow]No valid checkpoints found. Starting from scratch.[/yellow]")

print("✅ Core logic defined.")


# %%
# =============================================================================
# Cell 4: Execution
# =============================================================================
def main():
    print("🚀 Starting training process...")
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    if IS_TPU:
        config.device = xm.xla_device()
    
    trainer = KaggleTrainer(config)
    trainer.train()
    
    print("✅ Training process finished.")


if __name__ == "__main__":
    main()
