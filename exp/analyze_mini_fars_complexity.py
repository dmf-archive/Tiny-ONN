import json
import math
import numpy as np
from pathlib import Path
from collections import Counter
from rich.console import Console
from rich.table import Table

def calculate_entropy(grid):
    if not grid: return 0.0
    flat = [cell for row in grid for cell in row]
    counts = Counter(flat)
    total = len(flat)
    return -sum((c/total) * math.log2(c/total) for c in counts.values())

def analyze_mini_subset(data_path, max_size=10):
    console = Console()
    path = Path(data_path)
    tasks = []
    
    for p in sorted(path.glob("*.json")):
        with open(p) as f:
            task = json.load(f)
        
        # 检查是否属于 mini_fars 子集
        all_grids = []
        for pair in task['train'] + task['test']:
            all_grids.append(pair['input'])
            if 'output' in pair: all_grids.append(pair['output'])
        
        if all(len(g) <= max_size and len(g[0]) <= max_size for g in all_grids if g):
            # 计算输出熵 (以测试集输出为准)
            out_grid = task['test'][0]['output']
            h, w = len(out_grid), len(out_grid[0])
            entropy = calculate_entropy(out_grid)
            
            # 计算 ADL 复杂度 (变化像素数)
            in_grid = task['test'][0]['input']
            if len(in_grid) == h and len(in_grid[0]) == w:
                diffs = sum(in_grid[i][j] != out_grid[i][j] for i in range(h) for j in range(w))
            else:
                diffs = h * w # 尺寸变化视为全量变化
                
            tasks.append({
                "id": p.stem,
                "entropy": entropy,
                "total_bits": entropy * (h * w),
                "diff_ratio": diffs / (h * w),
                "size": f"{h}x{w}"
            })

    table = Table(title=f"Mini-FARS Task Complexity Analysis (N={len(tasks)})")
    table.add_column("Task ID", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("H(Y) (bits/px)", justify="right")
    table.add_column("Total Info (bits)", justify="right")
    table.add_column("ADL Ratio", justify="right")

    # 按总信息量排序
    tasks.sort(key=lambda x: x['total_bits'], reverse=True)
    
    for t in tasks[:10]: # 展示最复杂的10个
        table.add_row(t['id'], t['size'], f"{t['entropy']:.3f}", f"{t['total_bits']:.1f}", f"{t['diff_ratio']:.2%}")
    
    console.print(table)
    return tasks

if __name__ == "__main__":
    analyze_mini_subset("data/ARC-AGI/data/training")
