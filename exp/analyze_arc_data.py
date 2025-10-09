import json
from pathlib import Path
from collections import Counter
import numpy as np
from rich.console import Console
from rich.table import Table

from exp.arc.tokenizer import ArcColorTokenizer
from exp.arc.data import GridSerializer

def analyze_split(data_path: Path, split_name: str, console: Console):
    console.print(f"[bold cyan]Analyzing {split_name} ARC dataset at: {data_path}[/bold cyan]")
    
    tokenizer = ArcColorTokenizer()
    serializer = GridSerializer(tokenizer)
    
    token_lengths = []
    grid_sizes = []
    task_files = list(data_path.glob("*.json"))

    if not task_files:
        console.print(f"[red]No JSON files found in {data_path}[/red]")
        return

    for task_file in task_files:
        with open(task_file, 'r') as f:
            task_data = json.load(f)
            
            input_ids, _, _ = serializer.serialize_task(task_data)
            token_lengths.append(len(input_ids))
            
            for part in ['train', 'test']:
                if part in task_data:
                    for item in task_data[part]:
                        for grid_type in ['input', 'output']:
                            if item.get(grid_type):
                                grid = item[grid_type]
                                h = len(grid)
                                w = len(grid[0]) if h > 0 else 0
                                grid_sizes.append(f"{h}x{w}")

    console.print(f"\n      Token Length Distribution for {split_name}")
    table_len = Table(box=None, show_header=True, header_style="bold magenta")
    table_len.add_column("Metric", style="cyan")
    table_len.add_column("Value", style="green")
    
    if token_lengths:
        arr = np.array(token_lengths)
        table_len.add_row("Total Tasks Analyzed", str(len(arr)))
        table_len.add_row("Min Length", str(np.min(arr)))
        table_len.add_row("Max Length", str(np.max(arr)))
        table_len.add_row("Mean Length", f"{np.mean(arr):.2f}")
        for p in [50, 80, 90, 95, 98, 99]:
            table_len.add_row(f"{p}th percentile", f"{np.percentile(arr, p):.2f}")
    
    console.print(table_len)
    
    console.print(f"     Top 15 Most Common Grid Sizes for {split_name}")
    table_grid = Table(box=None, show_header=True, header_style="bold magenta")
    table_grid.add_column("Grid Size (HxW)", style="cyan")
    table_grid.add_column("Count", style="green")
    table_grid.add_column("Percentage", style="yellow")

    if grid_sizes:
        counts = Counter(grid_sizes)
        total = sum(counts.values())
        for size, count in counts.most_common(15):
            table_grid.add_row(size, str(count), f"{(count/total)*100:.2f}%")
    
    console.print(table_grid)

def main():
    console = Console()
    base_path = Path("data/ARC-AGI-2/data")
    analyze_split(base_path / "training", "training", console)
    analyze_split(base_path / "evaluation", "evaluation", console)

if __name__ == "__main__":
    main()