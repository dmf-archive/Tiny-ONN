import random

import matplotlib.pyplot as plt
import numpy as np

from utils.data_quantization import dequantize_linear_symmetric, dequantize_log_norm
from utils.logging_utils import log_debug


def update_plot(token_idx, view_mode, db_conn, total_tokens_processed):
    if total_tokens_processed == 0:
        return plt.figure()

    token_idx = max(0, min(token_idx, total_tokens_processed - 1))
    
    cursor = db_conn.cursor()
    # Query from block_metrics table
    cursor.execute("SELECT param_name, block_idx, activation, grad_norm, absmax FROM block_metrics WHERE token_idx=?", (token_idx,))
    data = cursor.fetchall()
    log_debug(f"Fetched {len(data)} rows from block_metrics for token_idx {token_idx}")

    if not data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data for current token", ha='center', va='center')
        return fig

    param_names, block_indices, activations, grad_norms, absmaxes = zip(*data, strict=False)

    # --- Sample data for logging ---
    log_sample_size = 30
    if len(param_names) > log_sample_size:
        log_sampled_indices = random.sample(range(len(param_names)), log_sample_size)
    else:
        log_sampled_indices = range(len(param_names))

    log_debug("\n--- Sampled Block Metrics from DB ---")
    for idx in log_sampled_indices:
        param_name = param_names[idx]
        block_idx = block_indices[idx]
        activation = activations[idx]
        grad_norm = grad_norms[idx]
        absmax = absmaxes[idx]
        
        log_debug(f"Param: {param_name}, Block: {block_idx}")
        log_debug(f"  Act: {activation:.4f}, Grad: {grad_norm:.4f}, AbsMax: {absmax:.4f}")
    log_debug("--- End Sampled Metrics ---")

    fig, ax = plt.subplots(figsize=(12, 8))

    # --- Structured Scatter-Heatmap ---
    coords_x, coords_y, plot_values = [], [], []
    
    # --- Dynamically build the block type map from the data ---
    unique_block_types = set()
    for name in param_names:
        try:
            parts = name.split('.')
            if len(parts) >= 5 and parts[1] == 'layers':
                block_type_key = f"{parts[3]}.{parts[4]}"
                unique_block_types.add(block_type_key)
        except (IndexError, ValueError):
            continue # Skip names that don't fit the expected format
            
    # Create a sorted map for consistent ordering
    block_type_map = {name: i for i, name in enumerate(sorted(list(unique_block_types)))}

    if not block_type_map:
        ax.text(0.5, 0.5, "Could not identify any plottable module types.", ha='center', va='center')
        return fig

    if view_mode == "Activation":
        values_to_plot = activations
        title = f"Token {token_idx}: Activation Scatter-Heatmap (Blocks: {len(data)})"
        cmap = 'viridis'
    elif view_mode == "Gradient Norm":
        values_to_plot = grad_norms
        title = f"Token {token_idx}: Gradient Norm Scatter-Heatmap (Blocks: {len(data)})"
        cmap = 'plasma'
    elif view_mode == "AbsMax":
        values_to_plot = absmaxes
        title = f"Token {token_idx}: AbsMax Scatter-Heatmap (Blocks: {len(data)})"
        cmap = 'cividis'
    else: # PI Diff Scatter (re-using the old logic for now, but will need block-level PI Diff)
        # For now, we'll just show a placeholder or adapt if PI Diff is calculated per block
        # As per note-2, PI Diff is calculated per param, not per block yet.
        # So, we'll default to Activation view if PI Diff Scatter is selected for block_metrics
        log_debug("PI Diff Scatter not yet implemented for block-level data. Defaulting to Activation view.")
        values_to_plot = activations
        title = f"Token {token_idx}: Activation Scatter-Heatmap (Blocks: {len(data)})"
        cmap = 'viridis'


    for i, name in enumerate(param_names):
        try:
            parts = name.split('.')
            # Extract layer index (e.g., 'model.layers.0.mlp.down_proj' -> 0)
            # This assumes a common naming convention like model.layers.X.module_type
            layer_idx_str = parts[2] if len(parts) > 2 and parts[1] == 'layers' else None
            
            if layer_idx_str and layer_idx_str.isdigit():
                layer_idx = int(layer_idx_str)
            else:
                # Fallback for non-layered modules or different naming conventions
                layer_idx = -1 # Assign a special index or skip
                # log_debug(f"Could not determine layer_idx for {name}. Skipping or assigning -1.")
                # continue # Uncomment to skip modules without clear layer index

            # Extract block type key (e.g., 'mlp.down_proj')
            # Assuming param_name format like 'model.layers.X.module_type.sub_module.weight'
            # We need 'module_type.sub_module'
            if len(parts) >= 5:
                block_type_key = f"{parts[3]}.{parts[4]}"
            else:
                block_type_key = "" # Fallback if format is unexpected
            
            if block_type_key in block_type_map:
                x_base = block_type_map[block_type_key]
                # Add jitter to x-coordinate for better visualization of overlapping points
                x_jitter = (random.random() - 0.5) * 0.8 # Jitter between -0.4 and 0.4
                
                coords_x.append(x_base + x_jitter)
                coords_y.append(layer_idx + (block_indices[i] * 0.01)) # Small offset for block_idx within layer
                plot_values.append(values_to_plot[i])
            else:
                # log_debug(f"Unknown block type key: {block_type_key} for param {name}. Skipping.")
                pass # Skip if block type is not mapped
        except (IndexError, ValueError) as e:
            log_debug(f"Error parsing param_name {name}: {e}. Skipping.")
            continue
    
    if not coords_x:
        ax.text(0.5, 0.5, "No plottable data for this view or parsing failed", ha='center', va='center')
        return fig

    scatter = ax.scatter(coords_x, coords_y, c=plot_values, cmap=cmap, s=15, alpha=0.7)
    
    ax.set_title(title)
    ax.set_ylabel("Layer Index")
    
    # Dynamically set y-ticks based on actual layer indices present in data
    unique_layers = sorted(list(set([int(y) for y in coords_y])))
    if unique_layers:
        ax.set_yticks(unique_layers)
    
    ax.set_xticks(list(block_type_map.values()))
    ax.set_xticklabels(list(block_type_map.keys()), rotation=45, ha='right')
    ax.invert_yaxis() # Layers typically go from 0 at top to N at bottom
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=ax, label="Value")

    plt.tight_layout()
    plt.close(fig) # Explicitly close the figure to free memory
    return fig
