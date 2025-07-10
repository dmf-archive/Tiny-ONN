import json
import os
import re
import sqlite3

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOT_CONFIG, QUANTIZATION_SCALE, VIEW_MODE_MAP

from .logging_utils import log_debug

# Global map for param_name to ID and vice-versa
id_to_param_name_map = {}

def get_plot_config(view_mode, **kwargs):
    """
    Retrieves the plot configuration for a given view mode and allows overrides.
    """
    config_key = VIEW_MODE_MAP.get(view_mode, "pi_diff")
    config = PLOT_CONFIG[config_key].copy()
    config.update(kwargs)
    return config

def fetch_and_prepare_data(db_conn, token_idx):
    """
    Fetches data for a specific token index, then calculates layer-wise z-scores.
    """
    global id_to_param_name_map
    if not id_to_param_name_map:
        map_path = os.path.join("metadata", "param_name_map.json")
        if os.path.exists(map_path):
            with open(map_path, 'r', encoding='utf-8') as f:
                id_to_param_name_map = json.load(f)
        else:
            log_debug("Parameter name map not found. Cannot map param_ids to names.")
            return None

    query = f"SELECT param_name, block_idx, activation, grad_norm, absmax FROM block_metrics WHERE token_idx = {token_idx}"
    df = pd.read_sql_query(query, db_conn)

    if df.empty:
        log_debug(f"No data found for token_idx {token_idx}")
        return None

    # Convert param_name (which is now ID) back to string name
    df['param_name'] = df['param_name'].astype(str).map(id_to_param_name_map)
    df.dropna(subset=['param_name'], inplace=True) # Drop rows where mapping failed

    # Dequantize values
    df['activation'] = df['activation'] / QUANTIZATION_SCALE
    df['grad_norm'] = df['grad_norm'] / QUANTIZATION_SCALE
    df['absmax'] = df['absmax'] / QUANTIZATION_SCALE

    # --- 1. Extract Layer Info (Robust Version) ---
    def get_layer_info(param_name):
        match = re.search(r'model\.layers\.(\d+)', param_name)
        if match:
            return int(match.group(1))
        elif 'embed_tokens' in param_name:
            return -1  # Special index for embeddings
        elif 'lm_head' in param_name:
            return 999 # Special index for the head
        elif 'model.norm' in param_name:
            return 998 # Special index for final norm
        return -2 # Other modules

    def get_layer_type(param_name):
        parts = param_name.split('.')
        # Heuristic to get a meaningful type label
        if 'layers' in parts:
            try:
                idx = parts.index('layers')
                return ".".join(parts[idx+2:]) # e.g., self_attn.q_proj
            except (ValueError, IndexError):
                return param_name
        return param_name

    df['layer_index'] = df['param_name'].apply(get_layer_info)
    df['layer_type'] = df['param_name'].apply(get_layer_type)

    # Dynamically assign final layer indices for plotting order
    if not df[df['layer_index'] < 900].empty:
        max_real_layer = df[df['layer_index'] < 900]['layer_index'].max()
        if 998 in df['layer_index'].unique(): # model.norm
             df.loc[df['layer_index'] == 998, 'layer_index'] = max_real_layer + 1
        if 999 in df['layer_index'].unique(): # lm_head
            df.loc[df['layer_index'] == 999, 'layer_index'] = max_real_layer + 2

    df = df[df['layer_index'] != -2].copy() # Use .copy() to avoid SettingWithCopyWarning
    df.dropna(subset=['layer_index', 'layer_type'], inplace=True)
    df['layer_index'] = df['layer_index'].astype(int)
    df = df.sort_values(by=['layer_index', 'layer_type', 'block_idx'])

    # --- 2. Calculate Layer-wise Z-Scores ---
    grouped = df.groupby('layer_index')
    
    def normalize_group(group):
        for metric in ['activation', 'grad_norm']:
            mean = group[metric].mean()
            std = group[metric].std()
            if std > 1e-9:
                group[f'{metric}_z_score'] = (group[metric] - mean) / std
            else:
                group[f'{metric}_z_score'] = 0.0
        return group

    df = grouped.apply(normalize_group, include_groups=False).reset_index()
    
    # --- 3. Prepare for Plotting ---
    layer_type_map = {t: i for i, t in enumerate(df['layer_type'].unique())}
    df['layer_type_idx'] = df['layer_type'].map(layer_type_map)
    
    log_debug(f"Processed {len(df)} records for token_idx {token_idx}")
    return df

def create_heatmap(df, value_col, title, cmap, vmin, vmax):
    """
    Creates a scatter plot heatmap from the dataframe.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    if df.empty:
        ax.text(0.5, 0.5, "No data available for this token.", 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title(title)
        plt.tight_layout()
        return fig
    
    unique_layer_types = sorted(df['layer_type'].unique(), key=lambda x: df[df['layer_type'] == x]['layer_type_idx'].iloc[0])
    
    x_coords = df['layer_type_idx'] + (df['block_idx'] / (df['block_idx'].max() + 1) * 0.8 - 0.4)
    
    scatter = ax.scatter(
        x_coords,
        df['layer_index'],
        c=df[value_col],
        cmap=cmap,
        s=10,
        alpha=0.7,
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax)
    )
    
    ax.set_xticks(range(len(unique_layer_types)))
    ax.set_xticklabels(unique_layer_types, rotation=45, ha='right')
    ax.set_xlabel("Layer Type")
    ax.set_ylabel("Layer Index")
    ax.set_title(title)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_yticks(np.arange(0, df['layer_index'].max() + 1, 1))
    ax.invert_yaxis()
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Value")
    
    plt.tight_layout()
    return fig

def update_plot(token_idx, view_mode, db_conn, total_tokens_processed, **kwargs):
    """
    Main function to update the plot based on the selected token and view mode.
    """
    df = fetch_and_prepare_data(db_conn, token_idx)
    config = get_plot_config(view_mode, **kwargs)
    title = f"Token {token_idx}: {view_mode}"

    if df is None or df.empty:
        log_debug(f"No data for token {token_idx}, returning empty plot.")
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.text(0.5, 0.5, f"No data available for token_idx: {token_idx}",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title(title)
        plt.tight_layout()
        return fig

    title = f"Token {token_idx}: {view_mode} (Blocks: {len(df)})"
    
    value_col_map = {
        "Activation": "activation",
        "Gradient Norm": "grad_norm",
        "AbsMax": "absmax",
        "Activation Z-Score": "activation_z_score",
        "Gradient Z-Score": "grad_norm_z_score",
        "S_p": "S_p"
    }
    
    value_col = value_col_map.get(view_mode, 'activation')

    # Dynamically set vmin/vmax for raw activation/gradient views
    if view_mode in ["Activation", "Gradient Norm", "AbsMax"]:
        # Use the max of the actual data for vmax, and 0 for vmin
        # For AbsMax view, use the absmax column directly
        if value_col in df.columns and not df[value_col].empty:
            config['vmin'] = 0.0
            config['vmax'] = df[value_col].max() * 1.1 # Add a small buffer
        else:
            config['vmin'] = 0.0
            config['vmax'] = 1.0 # Default if no data or column missing
    
    if view_mode == "S_p":
        df['S_p'] = df['activation_z_score'] - df['grad_norm_z_score']
        title = f"Token {token_idx}: S_p (Fixed Weights)"

    if value_col not in df.columns:
        log_debug(f"Column '{value_col}' not found in DataFrame. Defaulting to 'activation'.")
        value_col = 'activation'
        
    return create_heatmap(df, value_col, title, config['cmap'], config['vmin'], config['vmax'])
