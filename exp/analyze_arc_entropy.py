import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

# Ensure the script can find the exp package
import sys
sys.path.append(str(Path(__file__).parent.parent))

from exp.arc.data import GridSerializer
from exp.arc.tokenizer import ArcColorTokenizer

def calculate_shannon_entropy(token_ids: list[int]) -> float:
    """Calculates the Shannon entropy of a sequence of token IDs in bits."""
    if not token_ids:
        return 0.0
    
    counts = Counter(token_ids)
    total_tokens = len(token_ids)
    
    entropy = 0.0
    for count in counts.values():
        probability = count / total_tokens
        entropy -= probability * math.log2(probability)
        
    return entropy

def analyze_dataset(data_path: Path, serializer: GridSerializer, console: Console, context_limit: int = 4096):
    console.print(f"\n[bold cyan]Starting analysis of ARC dataset at:[/] {data_path}")
    
    file_paths = sorted(list(data_path.glob("*.json")))
    
    entropies = []
    token_lengths = []

    for file_path in file_paths:
        with open(file_path) as f:
            task_data = json.load(f)
        
        # We use serialize_for_inference to get the prompt tokens (train examples + test input)
        prompt_token_ids = serializer.serialize_for_inference(task_data)
        
        # Truncate to context limit
        prompt_token_ids = prompt_token_ids[:context_limit]
        
        token_lengths.append(len(prompt_token_ids))
        entropy = calculate_shannon_entropy(prompt_token_ids)
        entropies.append(entropy)

    if not entropies:
        console.print("[bold red]No valid tasks found to analyze.[/]")
        return

    # --- Print Statistics ---
    table = Table(title=f"Information Entropy Distribution (in bits) for {data_path.name}")
    table.add_column("Metric", justify="right", style="magenta")
    table.add_column("Value", justify="left", style="green")

    table.add_row("Total Tasks Analyzed", str(len(entropies)))
    table.add_row("Min Entropy", f"{np.min(entropies):.4f}")
    table.add_row("Max Entropy", f"{np.max(entropies):.4f}")
    table.add_row("Mean Entropy", f"{np.mean(entropies):.4f}")
    table.add_row("Median Entropy (50th percentile)", f"{np.median(entropies):.4f}")
    table.add_row("80th percentile", f"{np.percentile(entropies, 80):.4f}")
    table.add_row("90th percentile", f"{np.percentile(entropies, 90):.4f}")
    table.add_row("95th percentile", f"{np.percentile(entropies, 95):.4f}")
    
    console.print(table)
    
    # Also show token length stats as it's highly relevant
    len_table = Table(title=f"Token Length Distribution for {data_path.name}")
    len_table.add_column("Metric", justify="right", style="magenta")
    len_table.add_column("Value", justify="left", style="green")
    
    len_table.add_row("Min Length", str(np.min(token_lengths)))
    len_table.add_row("Max Length", str(np.max(token_lengths)))
    len_table.add_row("Mean Length", f"{np.mean(token_lengths):.2f}")
    
    console.print(len_table)


def main():
    console = Console()
    try:
        tokenizer = ArcColorTokenizer()
        serializer = GridSerializer(tokenizer)
        base_path = Path(__file__).parent.parent / "data" / "ARC-AGI-2" / "data"
        
        training_path = base_path / "training"
        evaluation_path = base_path / "evaluation"
        
        analyze_dataset(training_path, serializer, console)
        analyze_dataset(evaluation_path, serializer, console)

    except ImportError:
        console.print("[bold red]Error:[/bold red] Could not import ARC modules.")
        console.print("Please ensure you are running this script from the project's root directory, e.g., 'python -m exp.analyze_arc_entropy'")
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Data directory not found at '{base_path}'.")
        console.print("Please ensure the ARC dataset is correctly placed.")


if __name__ == "__main__":
    main()