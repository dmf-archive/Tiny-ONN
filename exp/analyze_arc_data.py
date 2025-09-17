#!/usr/bin/env python3
"""
ARC-AGI-2 Mini-Task 数据分析脚本
适配新的 mini-task 格式，统计总任务数、token 长度分布和网格尺寸分布。
"""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from rich.console import Console
from rich.table import Table

# 导入新的 tokenizer 和 serializer
from exp.arc.tokenizer import ArcColorTokenizer
from exp.arc.data import GridSerializer, InMemoryArcDataset

def analyze_dataset(data_path: Path, split: str, console: Console):
    """分析指定 split 的数据集"""
    console.print(f"[bold cyan]开始分析 ARC-{split} 数据集: {data_path}[/bold cyan]")

    # 使用 InMemoryArcDataset 自动展平所有样本对
    dataset = InMemoryArcDataset(data_path=data_path, split=split)
    tokenizer = ArcColorTokenizer()
    serializer = GridSerializer(tokenizer)

    token_lengths = []
    grid_sizes = []
    total_mini_tasks = len(dataset)

    console.print(f"[cyan]正在分析 {total_mini_tasks} 个 mini-tasks...[/cyan]")

    # 遍历所有 mini-tasks 收集数据
    for i, mini_task in enumerate(dataset):
        if i % 2000 == 0 and i > 0:
            console.print(f"已处理 {i} 个 mini-tasks...")

        # 1. 计算当前 mini-task 的 token 长度
        try:
            input_ids, _ = serializer.serialize_mini_task(mini_task)
            token_lengths.append(len(input_ids))
        except Exception as e:
            console.print(f"[yellow]无法序列化 mini-task {i}: {e}[/yellow]")
            continue

        # 2. 收集网格尺寸
        for grid_type in ['input', 'output']:
            if grid_type in mini_task:
                grid = mini_task[grid_type]
                height = len(grid)
                width = len(grid[0]) if height > 0 else 0
                grid_sizes.append(f"{height}x{width}")

    # --- 统计与报告 ---
    console.print(f"\n[bold green]{split.upper()} 数据集分析结果:[/bold green]")

    # 1. Mini-Task 总数
    console.print(f"[bold magenta]总 Mini-Tasks 数量: {total_mini_tasks}[/bold magenta]")

    # 2. Token 长度分布
    if token_lengths:
        token_array = np.array(token_lengths)
        table = Table(title=f"Token 长度分布 ({split})")
        table.add_column("指标", style="cyan")
        table.add_column("数值", style="magenta")
        table.add_row("最小长度", f"{token_array.min()}")
        table.add_row("最大长度", f"{token_array.max()}")
        table.add_row("平均长度", f"{token_array.mean():.2f}")
        table.add_row("中位数", f"{np.median(token_array):.2f}")
        for p in [80, 90, 95, 98, 99]:
            table.add_row(f"{p} 分位数", f"{np.percentile(token_array, p):.2f}")
        console.print(table)
    else:
        console.print(f"[red]未能计算 {split} 的 token 长度。[/red]")

    # 3. 网格尺寸分布
    if grid_sizes:
        size_counts = Counter(grid_sizes)
        table = Table(title=f"最常见网格尺寸 (前15) ({split})")
        table.add_column("网格尺寸 (HxW)", style="cyan")
        table.add_column("计数", style="magenta")
        table.add_column("百分比", style="green")
        
        total_grids = len(grid_sizes)
        for size, count in size_counts.most_common(15):
            percentage = (count / total_grids) * 100
            table.add_row(size, str(count), f"{percentage:.2f}%")
        console.print(table)
    else:
        console.print(f"[red]未能收集 {split} 的网格尺寸。[/red]")

    return total_mini_tasks

def main():
    console = Console()
    console.print("[bold green]=== ARC-AGI-2 Mini-Task 数据分析 ===[/bold green]")

    base_data_path = Path("data/ARC-AGI-2/data")
    training_path = base_data_path / "training"
    evaluation_path = base_data_path / "evaluation"

    total_mini_tasks_all = 0

    if training_path.exists():
        total_mini_tasks_all += analyze_dataset(training_path, "training", console)
    else:
        console.print(f"[red]未找到训练路径: {training_path}[/red]")

    console.print() # 空行分隔训练和评估结果

    if evaluation_path.exists():
        total_mini_tasks_all += analyze_dataset(evaluation_path, "evaluation", console)
    else:
        console.print(f"[red]未找到评估路径: {evaluation_path}[/red]")

    console.print(f"\n[bold green]=== 总计 ===[/bold green]")
    console.print(f"[bold magenta]所有数据集中的总 Mini-Tasks 数量: {total_mini_tasks_all}[/bold magenta]")

if __name__ == "__main__":
    main()