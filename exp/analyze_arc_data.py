import json
from pathlib import Path
from collections import Counter
import numpy as np
from rich.console import Console
from rich.table import Table

# Temporarily add exp path to sys.path to import from sibling directories
import sys
# Add the parent directory of 'exp' to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from exp.dynsiha_moie_arc.tokenizer import ArcChatMLTokenizer
from exp.dynsiha_moie_arc.data import GridSerializer

def analyze_dataset(data_path: Path):
    console = Console()
    console.print(f"[bold cyan]Starting analysis of ARC dataset at: {data_path}[/bold cyan]")

    tokenizer = ArcChatMLTokenizer()
    serializer = GridSerializer(tokenizer)

    token_lengths = []
    grid_sizes = []

    file_paths = sorted(list(data_path.glob("**/*.json")))

    if not file_paths:
        console.print(f"[bold red]No JSON files found in {data_path}. Please check the path.[/bold red]")
        return

    for file_path in file_paths:
        with open(file_path) as f:
            task_data = json.load(f)

        # 1. Calculate token length
        try:
            input_ids, _ = serializer.serialize_task_with_context(task_data)
            token_lengths.append(len(input_ids))
        except Exception as e:
            console.print(f"[yellow]Could not serialize {file_path.name}: {e}[/yellow]")
            continue

        # 2. Collect grid sizes
        for pair in task_data.get('train', []) + task_data.get('test', []):
            for grid_type in ['input', 'output']:
                if grid_type in pair:
                    grid = pair[grid_type]
                    height = len(grid)
                    width = len(grid[0]) if height > 0 else 0
                    grid_sizes.append(f"{height}x{width}")

    # --- Analysis and Reporting ---

    # Token Length Analysis
    if token_lengths:
        token_array = np.array(token_lengths)
        table = Table(title="Token Length Distribution")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Total Tasks Analyzed", f"{len(token_array)}")
        table.add_row("Min Length", f"{token_array.min()}")
        table.add_row("Max Length", f"{token_array.max()}")
        table.add_row("Mean Length", f"{token_array.mean():.2f}")
        table.add_row("Median (50th percentile)", f"{np.median(token_array):.2f}")
        for p in [80, 90, 95, 98, 99]:
            table.add_row(f"{p}th percentile", f"{np.percentile(token_array, p):.2f}")
        
        console.print(table)
    else:
        console.print("[red]No token lengths were calculated.[/red]")

    # Grid Size Analysis
    if grid_sizes:
        size_counts = Counter(grid_sizes)
        table = Table(title="Top 15 Most Common Grid Sizes")
        table.add_column("Grid Size (HxW)", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Percentage", style="green")
        
        total_grids = len(grid_sizes)
        for size, count in size_counts.most_common(15):
            percentage = (count / total_grids) * 100
            table.add_row(size, str(count), f"{percentage:.2f}%")
            
        console.print(table)
    else:
        console.print("[red]No grid sizes were collected.[/red]")


if __name__ == "__main__":
    # The script assumes it's being run from the root of the project directory
    # where 'data' is a subdirectory.
    base_data_path = Path("data/ARC-AGI-2/data")
    
    training_path = base_data_path / "training"
    evaluation_path = base_data_path / "evaluation"

    if training_path.exists():
        analyze_dataset(training_path)
    else:
        print(f"Training path not found: {training_path}")

    if evaluation_path.exists():
        analyze_dataset(evaluation_path)
    else:
        print(f"Evaluation path not found: {evaluation_path}")