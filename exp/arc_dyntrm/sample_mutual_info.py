import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

# Ensure the script can find the exp package
import sys
sys.path.append(str(Path(__file__).parent.parent))

from exp.arc_dyntrm.data import GridSerializer
from exp.arc_dyntrm.tokenizer import ArcColorTokenizer

def calculate_grid_mutual_info(grid1: List[List[int]], grid2: List[List[int]]) -> float:
    """计算两个网格之间的互信息"""
    if not grid1 or not grid2:
        return 0.0
    
    # 将2D网格展平为1D序列
    flat1 = [item for row in grid1 for item in row]
    flat2 = [item for row in grid2 for item in row]
    
    if len(flat1) != len(flat2):
        return 0.0
    
    # 计算互信息
    return calculate_sequence_mutual_info(flat1, flat2)

def calculate_sequence_mutual_info(seq1: List[int], seq2: List[int]) -> float:
    """计算两个序列之间的互信息"""
    if len(seq1) != len(seq2) or len(seq1) == 0:
        return 0.0
    
    # 联合概率分布
    joint_counts = defaultdict(int)
    x_counts = defaultdict(int)
    y_counts = defaultdict(int)
    
    for xi, yi in zip(seq1, seq2):
        joint_counts[(xi, yi)] += 1
        x_counts[xi] += 1
        y_counts[yi] += 1
    
    n = len(seq1)
    mi = 0.0
    
    # 计算互信息: I(X;Y) = ΣΣ p(x,y) log(p(x,y)/(p(x)p(y)))
    for (xi, yi), joint_count in joint_counts.items():
        if joint_count > 0:
            p_xy = joint_count / n
            p_x = x_counts[xi] / n
            p_y = y_counts[yi] / n
            
            if p_x > 0 and p_y > 0:
                mi += p_xy * math.log2(p_xy / (p_x * p_y))
    
    return max(0.0, mi)  # 确保非负

def calculate_spatial_mutual_info(grid: List[List[int]], max_distance: int = 2) -> float:
    """计算网格内空间位置的互信息"""
    if not grid or not grid[0]:
        return 0.0
    
    rows, cols = len(grid), len(grid[0])
    total_mi = 0.0
    pair_count = 0
    
    # 计算每个位置与其邻居的互信息
    for r in range(rows):
        for c in range(cols):
            for dr in range(-max_distance, max_distance + 1):
                for dc in range(-max_distance, max_distance + 1):
                    if dr == 0 and dc == 0:
                        continue
                    
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        # 计算当前位置与邻居的互信息
                        mi = calculate_sequence_mutual_info([grid[r][c]], [grid[nr][nc]])
                        total_mi += mi
                        pair_count += 1
    
    return total_mi / pair_count if pair_count > 0 else 0.0

def calculate_task_sample_mutual_info(task_data: Dict[str, Any]) -> Dict[str, float]:
    """计算ARC任务中的样本互信息"""
    results = {}
    
    if "train" not in task_data or not task_data["train"]:
        return results
    
    train_pairs = task_data["train"]
    
    # 1. 输入-输出互信息（同一训练对）
    input_output_mis = []
    for pair in train_pairs:
        if "input" in pair and "output" in pair:
            mi = calculate_grid_mutual_info(pair["input"], pair["output"])
            input_output_mis.append(mi)
    
    if input_output_mis:
        results["input_output_mi"] = np.mean(input_output_mis)
    
    # 2. 训练样本间的互信息
    if len(train_pairs) > 1:
        sample_mis = []
        for i in range(len(train_pairs)):
            for j in range(i + 1, len(train_pairs)):
                # 比较输入之间的相似性
                if "input" in train_pairs[i] and "input" in train_pairs[j]:
                    mi = calculate_grid_mutual_info(
                        train_pairs[i]["input"], 
                        train_pairs[j]["input"]
                    )
                    sample_mis.append(mi)
        
        if sample_mis:
            results["inter_sample_mi"] = np.mean(sample_mis)
    
    # 3. 空间结构互信息
    spatial_mis = []
    for pair in train_pairs:
        if "input" in pair:
            spatial_mi = calculate_spatial_mutual_info(pair["input"])
            spatial_mis.append(spatial_mi)
    
    if spatial_mis:
        results["spatial_structure_mi"] = np.mean(spatial_mis)
    
    # 4. 颜色分布互信息
    color_mis = []
    for pair in train_pairs:
        if "input" in pair and "output" in pair:
            # 计算颜色分布的互信息
            input_colors = [item for row in pair["input"] for item in row]
            output_colors = [item for row in pair["output"] for item in row]
            
            # 创建颜色直方图
            input_counter = Counter(input_colors)
            output_counter = Counter(output_colors)
            
            # 获取所有颜色
            all_colors = set(input_counter.keys()) | set(output_counter.keys())
            
            # 创建概率分布
            input_probs = [input_counter.get(color, 0) / len(input_colors) for color in all_colors]
            output_probs = [output_counter.get(color, 0) / len(output_colors) for color in all_colors]
            
            # 计算颜色分布的互信息
            color_mi = calculate_sequence_mutual_info(
                [int(p * 1000) for p in input_probs],  # 转换为整数避免浮点问题
                [int(p * 1000) for p in output_probs]
            )
            color_mis.append(color_mi)
    
    if color_mis:
        results["color_distribution_mi"] = np.mean(color_mis)
    
    return results

def analyze_arc_sample_mutual_info(data_path: Path, console: Console, max_samples: int = 100):
    """分析ARC数据集的样本互信息特征"""
    console.print(f"\n[bold cyan]Starting ARC sample mutual information analysis at:[/] {data_path}")
    
    file_paths = sorted(list(data_path.glob("*.json")))
    if max_samples:
        file_paths = file_paths[:max_samples]
    
    # 存储所有指标
    all_metrics = defaultdict(list)
    
    for idx, file_path in enumerate(file_paths):
        if idx % 20 == 0:
            console.print(f"Processing {idx}/{len(file_paths)} files...")
        
        try:
            with open(file_path) as f:
                task_data = json.load(f)
            
            # 计算该任务的互信息指标
            metrics = calculate_task_sample_mutual_info(task_data)
            
            for key, value in metrics.items():
                all_metrics[key].append(value)
                
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to process {file_path}: {e}[/]")
            continue

    if not all_metrics:
        console.print("[bold red]No valid tasks found to analyze.[/]")
        return

    # --- 打印统计结果 ---
    table = Table(title=f"ARC Sample Mutual Information Analysis for {data_path.name}")
    table.add_column("Metric", justify="right", style="magenta")
    table.add_column("Mean (bits)", justify="left", style="green")
    table.add_column("Std (bits)", justify="left", style="cyan")
    table.add_column("Max (bits)", justify="left", style="yellow")

    table.add_row("Total Tasks Analyzed", str(len(all_metrics.get("input_output_mi", []))), "", "")
    
    for key, values in all_metrics.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            max_val = np.max(values)
            
            key_name = key.replace("_", " ").title()
            table.add_row(key_name, f"{mean_val:.6f}", f"{std_val:.6f}", f"{max_val:.6f}")
    
    console.print(table)
    
    # 计算香农熵作为对比
    console.print("\n[bold blue]Calculating Shannon entropy for comparison...[/]")
    
    shannon_entropies = []
    for file_path in file_paths[:len(all_metrics.get("input_output_mi", []))]:
        try:
            with open(file_path) as f:
                task_data = json.load(f)
            
            # 计算所有网格的香农熵平均值
            task_entropies = []
            for pair in task_data.get("train", []):
                if "input" in pair:
                    flat_grid = [item for row in pair["input"] for item in row]
                    if flat_grid:
                        counts = Counter(flat_grid)
                        total = len(flat_grid)
                        entropy = -sum((count/total) * math.log2(count/total) for count in counts.values())
                        task_entropies.append(entropy)
            
            if task_entropies:
                shannon_entropies.append(np.mean(task_entropies))
        except:
            continue
    
    if shannon_entropies and all_metrics.get("input_output_mi"):
        comparison_table = Table(title="Entropy Comparison")
        comparison_table.add_column("Entropy Type", justify="right", style="magenta")
        comparison_table.add_column("Mean Value", justify="left", style="green")
        comparison_table.add_column("Ratio to Shannon", justify="left", style="cyan")
        
        shannon_mean = np.mean(shannon_entropies)
        input_output_mi_mean = np.mean(all_metrics["input_output_mi"])
        
        comparison_table.add_row("Shannon Entropy", f"{shannon_mean:.6f}", "1.000")
        comparison_table.add_row("Input-Output MI", f"{input_output_mi_mean:.6f}", f"{input_output_mi_mean/shannon_mean:.6f}")
        
        if "inter_sample_mi" in all_metrics:
            inter_sample_mean = np.mean(all_metrics["inter_sample_mi"])
            comparison_table.add_row("Inter-Sample MI", f"{inter_sample_mean:.6f}", f"{inter_sample_mean/shannon_mean:.6f}")
        
        if "spatial_structure_mi" in all_metrics:
            spatial_mean = np.mean(all_metrics["spatial_structure_mi"])
            comparison_table.add_row("Spatial Structure MI", f"{spatial_mean:.6f}", f"{spatial_mean/shannon_mean:.6f}")
        
        console.print(comparison_table)
    
    # 分析高互信息任务特征
    if "input_output_mi" in all_metrics and all_metrics["input_output_mi"]:
        high_mi_threshold = np.percentile(all_metrics["input_output_mi"], 90)
        high_mi_count = sum(1 for mi in all_metrics["input_output_mi"] if mi > high_mi_threshold)
        
        console.print(f"\n[bold green]High Mutual Information Analysis:[/]")
        console.print(f"Tasks with MI > {high_mi_threshold:.4f}: {high_mi_count}/{len(all_metrics['input_output_mi'])} ({high_mi_count/len(all_metrics['input_output_mi'])*100:.1f}%)")
        
        # 这些任务可能更容易学习（更高的可预测性）
        console.print(f"High MI tasks may represent more learnable patterns")

def main():
    console = Console()
    try:
        console.print("[bold blue]ARC Sample Mutual Information Entropy Analysis[/bold blue]")
        console.print("This measures the statistical dependencies between different parts of ARC tasks")
        console.print("at the structural level (grids) rather than token level")
        
        base_path = Path(__file__).parent.parent.parent / "data" / "ARC-AGI-2" / "data"
        
        training_path = base_path / "training"
        evaluation_path = base_path / "evaluation"
        
        analyze_arc_sample_mutual_info(training_path, console, max_samples=100)
        analyze_arc_sample_mutual_info(evaluation_path, console, max_samples=100)

    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] Data directory not found: {e}")
        console.print("Please ensure the ARC dataset is correctly placed.")

if __name__ == "__main__":
    main()