# %% [markdown]
# # Standalone ARC Sampler & DFS Debugger
#
# This script is a self-contained utility for loading a model checkpoint and
# running inference on a single ARC task file. Its primary purpose is to provide
# a controlled environment for debugging generation algorithms, especially DFS,
# without interfering with the main training loop.

# %%
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.text import Text
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

# %% [markdown]
# ## Core Imports from Existing Modules
MIN_PROB = 0.5

# %%
# Add project root to the Python path to resolve imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from exp.arc.config import GenerationConfig, ModelConfig
from exp.arc.data import ArcColorTokenizer, GridDeserializer, GridSerializer
from exp.arc.model import ArcTransformer

# %% [markdown]
# ## Re-implemented & Adapted Logic for Debugging
# We redefine the Observer and Generator classes here to allow for isolated modifications
# and to keep this script standalone from the training/evaluation pipeline.

# %%
@dataclass
class DFSState:
    max_new_tokens: int
    max_score: float
    pos: int
    cache: list
    coords: torch.Tensor
    score: float = 0.0
    pbar: tqdm | None = None
    max_depth_seen: int = 0

class DebugObserver:
    ARC_COLORS: list[str] = [
        "black", "blue", "red", "green", "yellow",
        "grey", "magenta", "orange", "cyan", "brown",
    ]

    def __init__(self, console: Console):
        self.console = console

    def _create_grid_text(self, grid: torch.Tensor) -> Text:
        if grid is None or not isinstance(grid, torch.Tensor):
            return Text("Invalid Grid", style="bold red")
        text = Text()
        h, w = grid.shape
        for r in range(h):
            for p in range(w):
                pixel = int(grid[r, p].item())
                color = self.ARC_COLORS[pixel] if 0 <= pixel < len(self.ARC_COLORS) else "white"
                text.append("■ ", style=color)
            if r < h - 1:
                text.append("\n")
        return text

    def visualize_debug_sample(
        self,
        task_data: dict[str, Any],
        pred_grid: torch.Tensor | None,
        pred_tokens: list[int],
        probabilities: list[float],
        strategy: str,
    ):
        self.console.print()
        self.console.print(f"--- Debug Sample Result ({strategy}) ---", style="bold yellow")

        if pred_grid is None:
            pred_grid = torch.zeros((1, 1), dtype=torch.long)

        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Input", justify="center")
        table.add_column("Target", justify="center")
        table.add_column("Prediction", justify="center")

        input_grid = torch.tensor(task_data['test'][0]['input'])
        target_grid = torch.tensor(task_data['test'][0]['output'])

        input_text = self._create_grid_text(input_grid)
        target_text = self._create_grid_text(target_grid)
        pred_text = self._create_grid_text(pred_grid)

        table.add_row(input_text, target_text, pred_text)
        self.console.print(table)

        if pred_tokens:
            self.console.print(f"[bold]Generated Token IDs ({len(pred_tokens)} tokens):[/bold]")
            self.console.print(" ".join(map(str, pred_tokens)))
        else:
            self.console.print("[bold]Generated Token IDs:[/bold] [red]N/A[/red]")

        if probabilities:
            if len(probabilities) > 0 and isinstance(probabilities[0], float) and probabilities[0] > 0 and probabilities[0] <= 1.0:
                prob_str = f"{probabilities[0]:.6f} (Path Score)"
            else:
                prob_str = " ".join([f"{p:.2f}" for p in probabilities])
            self.console.print(f"[bold]Probabilities:[/bold] {prob_str}")

        self.console.print()


class DebugGenerator:
    def __init__(self, model: ArcTransformer, serializer: GridSerializer, deserializer: GridDeserializer, device: torch.device):
        self.model = model
        self.serializer = serializer
        self.deserializer = deserializer
        self.device = device
        self.eos_token_id = self.serializer.tokenizer.eos_token_id

    @torch.no_grad()
    def generate(self, task_data: dict[str, Any], config: GenerationConfig) -> tuple[torch.Tensor, list[int], list[float]]:
        # Use the maximum sequence length from ARC dataset analysis as a principled upper bound
        max_new_tokens = 932

        prompt_ids, prompt_coords = self.serializer.serialize_for_inference(task_data)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        prompt_coords_tensor = torch.tensor([prompt_coords], dtype=torch.long, device=self.device)

        if config.use_dfs:
            results = self._dfs_search(prompt_tensor, prompt_coords_tensor, max_new_tokens, config)
            generated_tokens = results[0][0].tolist() if results else []
            probabilities = [results[0][1]] * len(generated_tokens) if results else []
        else:
            generated_tokens, probabilities = self._greedy_search(prompt_tensor, prompt_coords_tensor, max_new_tokens)

        pred_grid = self.deserializer.deserialize(generated_tokens)
        return pred_grid, generated_tokens, probabilities

    def _greedy_search(self, input_ids: torch.Tensor, input_coords: torch.Tensor, max_new_tokens: int) -> tuple[list[int], list[float]]:
        tokens = input_ids.clone()
        coords = input_coords.clone()
        past_key_values = None
        current_r, current_c = 0, -1
        probabilities = []

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION), torch.autocast(
            device_type=self.device.type, dtype=torch.float16
        ):
            for _ in range(max_new_tokens):
                model_input = tokens if past_key_values is None else tokens[:, -1:]
                coords_input = coords if past_key_values is None else coords[:, -1:]
                outputs = self.model(
                    model_input, coords=coords_input, past_key_values=past_key_values, return_dict=True
                )
                logits, past_key_values = outputs["logits"], outputs["past_key_values"]

                next_token_logits = logits[:, -1, :]
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(next_token_probs, dim=-1, keepdim=True)

                prob = next_token_probs[0, next_token.item()].item()
                probabilities.append(prob)

                tokens = torch.cat([tokens, next_token], dim=-1)

                next_token_item = next_token.item()
                if self.serializer.tokenizer.token_id_to_color(next_token_item) is not None:
                    current_c += 1
                    next_coord_tuple = (current_r, current_c)
                elif next_token_item == self.serializer.tokenizer.row_sep_token_id:
                    current_r += 1
                    current_c = 0
                    next_coord_tuple = (current_r, 0)
                else:
                    next_coord_tuple = (-1, -1)

                next_coord = torch.tensor([[next_coord_tuple]], dtype=torch.long, device=self.device)
                coords = torch.cat([coords, next_coord], dim=1)

                if next_token_item == self.eos_token_id or next_token_item == self.serializer.tokenizer.vocab["<im_end>"]:
                    break

        return tokens[0, input_ids.shape[1] :].tolist(), probabilities

    def _dfs_search(self, input_ids: torch.Tensor, input_coords: torch.Tensor, max_new_tokens: int, config: GenerationConfig) -> list[tuple[np.ndarray, float]]:
        sys.setrecursionlimit(1000 + max_new_tokens)
        input_ids_squeezed = input_ids.squeeze(0)
        pos = len(input_ids_squeezed)

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION), torch.autocast(
            device_type=self.device.type, dtype=torch.float16
        ):
            outputs = self.model(input_ids, coords=input_coords, return_dict=True)
            logits, cache = outputs["logits"], [outputs["past_key_values"]]

        logits_sliced = logits[0, pos - 1 :]
        initial_state = DFSState(
            max_new_tokens=max_new_tokens,
            max_score=-np.log(config.min_prob),
            pos=pos,
            cache=cache,
            coords=input_coords,
            pbar=tqdm(total=None, desc="DFS Exploring", unit="nodes"),
            max_depth_seen=pos
        )
        result = self._explore(logits_sliced, [], initial_state)
        if initial_state.pbar is not None:
            initial_state.pbar.close()
        print(f"[DEBUG] _explore returned {len(result)} paths.")
        if result:
            print(f"[DEBUG] Best path score: {result[0][1]:.3f}, length: {len(result[0][0])}")
        else:
            print("[DEBUG] _explore returned no paths.")
        return sorted([(np.array(suffix[::-1]), np.exp(-score_val)) for suffix, score_val in result], key=lambda x: x[1], reverse=True)

    def _explore(self, logits: torch.Tensor, path: list[int], state: DFSState) -> list[tuple[list[int], float]]:
        if state.pbar is not None:
            # Update progress bar: count explored nodes, show max depth in postfix
            state.pbar.update(1)
            current_depth = state.pos
            if current_depth > getattr(state, 'max_depth_seen', 0):
                state.max_depth_seen = current_depth
                state.pbar.set_postfix({'max_depth': current_depth})
        # Print entrance info for every recursive call (debug only)
        # print(f"[DFS-ENTER] depth={state.pos}, remaining={state.max_new_tokens}, path_len={len(path)}, cache_size={state.cache[0][0][0].shape[2] if state.cache else 'None'}, score={state.score:.2f}")
        first_token_logits, remaining_logits = logits[0], (logits[1:] if len(logits) > 1 else None)
        logits_cpu = first_token_logits.detach().float().cpu()
        nll = -logits_cpu.log_softmax(-1)
        next_token_probs = torch.softmax(first_token_logits, dim=-1)
        softmax = list(enumerate(nll))
        # Removed the problematic heuristic reordering that was causing EOS to be repeatedly prioritized.
        # This was leading to the search tree collapsing after only a few tokens.

        return_suffixes = []
        current_r, current_c = state.coords[0, -1, 0].item(), state.coords[0, -1, 1].item()

        # Print softmax summary for first few tokens to reduce spam (debug only)
        # print(f"[DFS-SOFTMAX] depth={state.pos}, top3 tokens: {[idx for idx, _ in softmax[:3]]}, logits: {[f'{logits_cpu[idx].item():.2f}' for idx, _ in softmax[:3]]}, scores: {[f'{s:.2f}' for _, s in softmax[:3]]}")
        for i, s in softmax:
            next_score = state.score + s.item()
            if next_score >= state.max_score:
                continue
            
            # Forbid EOS in the first few tokens to force a minimal length
            if i == self.eos_token_id and state.pos < 10:
                continue
            if i == self.eos_token_id:
                suffixes = [([], next_score)]
            elif state.max_new_tokens > 1:
                if remaining_logits is None:
                    if state.pos < state.cache[0][0][0].shape[2]:
                        state.cache[0] = tuple(tuple(c[:, :, :state.pos] for c in l) for l in state.cache[0])

                    next_token_tensor = torch.tensor([[i]], device=self.device)
                    if self.serializer.tokenizer.token_id_to_color(i) is not None:
                        current_c += 1
                        next_coord_tuple = (current_r, current_c)
                    elif i == self.serializer.tokenizer.row_sep_token_id:
                        current_r += 1
                        current_c = 0
                        next_coord_tuple = (current_r, 0)
                    else:
                        next_coord_tuple = (-1, -1)
                    next_coord = torch.tensor([[next_coord_tuple]], dtype=torch.long, device=self.device)

                    outputs = self.model(
                        next_token_tensor, coords=next_coord, past_key_values=state.cache[0], return_dict=True
                    )
                    new_logits, state.cache[0] = outputs["logits"], outputs["past_key_values"]
                    new_logits = new_logits[0]
                else:
                    new_logits, next_coord = remaining_logits, state.coords

                new_state = DFSState(
                    max_new_tokens=state.max_new_tokens - 1,
                    max_score=state.max_score,
                    pos=state.pos + 1,
                    cache=state.cache,
                    coords=next_coord,
                    score=next_score,
                    pbar=state.pbar,
                    max_depth_seen=state.max_depth_seen
                )
                # print(f"[DFS-RECUR] token={i}, depth={state.pos}→{state.pos+1}, remaining={state.max_new_tokens-1}")
                suffixes = self._explore(new_logits, path, new_state)
                # if not suffixes:
                #     print(f"[DFS-FAIL]  token={i} yielded no suffixes")
            else:
                # Force return current path when reaching max depth
                suffixes = [([], next_score)]

            for suffix in suffixes:
                suffix[0].append(i)
            return_suffixes.extend(suffixes)
            remaining_logits = None
        # print(f"[DFS-KEEP]  depth={state.pos}, kept {len(kept_tokens)} tokens: {kept_tokens[:5]}")
        return return_suffixes

# %% [markdown]
# ## Main Execution Logic

# %%
import os
import random


def _find_and_load_latest_checkpoint(model: ArcTransformer, device: str, console: Console) -> bool:
    console.print("Searching for the latest checkpoint...")
    checkpoint_dir = Path(__file__).parent.parent / "exp" / "arc" / "checkpoints"
    
    if not checkpoint_dir.exists():
        console.print(f"Error: Checkpoint directory not found at {checkpoint_dir}", style="bold red")
        return False

    checkpoint_folders = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir()],
        key=os.path.getmtime,
        reverse=True
    )

    for folder in checkpoint_folders:
        ckpt_path = folder / "model.safetensors"
        if ckpt_path.is_file():
            try:
                console.print(f"Loading checkpoint from: {ckpt_path}")
                state_dict = load_file(ckpt_path, device=device)
                model.load_state_dict(state_dict)
                console.print("Model loaded successfully.", style="bold green")
                return True
            except Exception as e:
                console.print(f"Failed to load {ckpt_path}: {e}", style="bold red")
    
    console.print("No valid checkpoint found.", style="bold yellow")
    return False


def _load_random_task(console: Console) -> dict | None:
    console.print("Selecting a random task...")
    task_dir = Path(__file__).parent.parent / "data" / "ARC-AGI-2" / "data" / "training"
    
    if not task_dir.exists():
        console.print(f"Error: Task directory not found at {task_dir}", style="bold red")
        return None
        
    task_files = list(task_dir.glob("*.json"))
    if not task_files:
        console.print(f"Error: No task files found in {task_dir}", style="bold red")
        return None
        
    random_task_path = random.choice(task_files)
    console.print(f"Selected task: {random_task_path.name}")
    
    with open(random_task_path) as f:
        return json.load(f)


def main():
    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}", style="bold green")

    # --- Configuration ---
    model_config = ModelConfig()
    greedy_config = GenerationConfig(use_dfs=False)
    dfs_config = GenerationConfig(use_dfs=True, min_prob=MIN_PROB) # Lower min_prob for wider search

    # --- Initialization ---
    tokenizer = ArcColorTokenizer()
    serializer = GridSerializer(tokenizer)
    deserializer = GridDeserializer(tokenizer)
    observer = DebugObserver(console)

    model_config.vocab_size = tokenizer.vocab_size
    model = ArcTransformer(model_config, device=str(device)).to(device)

    # --- Load Checkpoint ---
    if not _find_and_load_latest_checkpoint(model, str(device), console):
        return
    model.eval()

    # --- Load Task ---
    task_data = _load_random_task(console)
    if not task_data:
        return

    # --- Generation & Visualization ---
    generator = DebugGenerator(model, serializer, deserializer, device)

    # 1. Greedy Search
    greedy_grid, greedy_tokens, greedy_probs = generator.generate(task_data, greedy_config)
    observer.visualize_debug_sample(task_data, greedy_grid, greedy_tokens, greedy_probs, "Greedy Search")

    # 2. DFS Search
    console.print("Starting DFS Search... this may take a while.", style="cyan")
    dfs_grid, dfs_tokens, dfs_probs = generator.generate(task_data, dfs_config)
    observer.visualize_debug_sample(task_data, dfs_grid, dfs_tokens, dfs_probs, "DFS Search")


if __name__ == "__main__":
    main()