import re
from typing import Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_CONFIG = {
    "activation": {"vmin": 0, "vmax": 450, "cmap": "viridis"},
    "gradient_norm": {"vmin": 0, "vmax": 1e-7, "cmap": "plasma"},
    "z_score": {"vmin": -5, "vmax": 5, "cmap": "coolwarm"},
    "sps": {"vmin": -5, "vmax": 5, "cmap": "coolwarm"},
}

def _get_plot_config(view_mode):
    if "Z-Score" in view_mode or "SPS" in view_mode:
        return PLOT_CONFIG["sps"].copy()
    key = "activation" if "Activation" in view_mode else "gradient_norm"
    return PLOT_CONFIG[key].copy()

def _prepare_base_data(data: np.ndarray, token_idx: Union[int, slice], id_to_param_name_map: dict[str, str]):
    if data is None or not id_to_param_name_map:
        return None

    if isinstance(token_idx, int):
        token_data = data[data["token_idx"] == token_idx]
    elif isinstance(token_idx, slice) and token_idx == slice(None):
        token_data = data
    else:
        return None

    if token_data.size == 0:
        return None

    df = pd.DataFrame(token_data)
    df["param_name"] = df["param_id"].astype(str).map(id_to_param_name_map)
    df.dropna(subset=["param_name"], inplace=True)
    for col in ["activation", "grad_norm"]:
        df[col] = df[col].astype(float)

    def get_layer_info(param_name):
        match = re.search(r'model\.layers\.(\d+)', param_name)
        if match: return int(match.group(1))
        if 'embed_tokens' in param_name: return -1
        if 'lm_head' in param_name: return 999
        if 'model.norm' in param_name: return 998
        return -2

    def get_layer_type(param_name):
        parts = param_name.split('.')
        if 'layers' in parts:
            try:
                idx = parts.index('layers')
                return ".".join(parts[idx+2:])
            except (ValueError, IndexError):
                return param_name
        return param_name

    df['layer_index'] = df['param_name'].apply(get_layer_info)
    df['layer_type'] = df['param_name'].apply(get_layer_type)

    if not df[df['layer_index'] < 900].empty:
        max_real_layer = df[df['layer_index'] < 900]['layer_index'].max()
        if 998 in df['layer_index'].unique():
             df.loc[df['layer_index'] == 998, 'layer_index'] = max_real_layer + 1
        if 999 in df['layer_index'].unique():
            df.loc[df['layer_index'] == 999, 'layer_index'] = max_real_layer + 2

    df = df[df['layer_index'] != -2].copy()
    df.dropna(subset=['layer_index', 'layer_type'], inplace=True)
    df['layer_index'] = df['layer_index'].astype(int)
    df = df.sort_values(by=["layer_index", "layer_type", "block_idx"])
    
    layer_type_map = {t: i for i, t in enumerate(df['layer_type'].unique())}
    df['layer_type_idx'] = df['layer_type'].map(layer_type_map)
    return df

def _calculate_z_scores(df: pd.DataFrame, scope: str = "layer"):
    metrics = ["activation", "grad_norm"]
    if scope == "layer":
        def normalize_group(group):
            for metric in metrics:
                mean = group[metric].mean()
                std = group[metric].std()
                group[f"{metric}_z_score"] = (group[metric] - mean) / std if std > 1e-9 else 0.0
            return group
        return df.groupby("layer_index").apply(normalize_group, include_groups=False).reset_index()
    
    # Global scope
    for metric in metrics:
        mean = df[metric].mean()
        std = df[metric].std()
        df[f"{metric}_z_score"] = (df[metric] - mean) / std if std > 1e-9 else 0.0
    return df

def create_heatmap(df, value_col, title, cmap, vmin, vmax):
    fig, ax = plt.subplots(figsize=(14, 10))
    if df.empty:
        ax.text(0.5, 0.5, "No data available for this token.", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        plt.tight_layout()
        return fig

    unique_layer_types = sorted(df['layer_type'].unique(), key=lambda x: df[df['layer_type'] == x]['layer_type_idx'].iloc[0])
    x_coords = df['layer_type_idx'] + (df['block_idx'] / (df['block_idx'].max() + 1) * 0.8 - 0.4)

    scatter = ax.scatter(x_coords, df['layer_index'], c=df[value_col], cmap=cmap, s=10, alpha=0.7, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))

    ax.set_xticks(range(len(unique_layer_types)))
    ax.set_xticklabels(unique_layer_types, rotation=45, ha="right")
    ax.set_xlabel("Layer Type")
    ax.set_ylabel("Layer Index")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_yticks(np.arange(0, df['layer_index'].max() + 1, 1))
    ax.invert_yaxis()
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Value")
    plt.tight_layout()
    return fig

def update_plot(token_idx, view_mode, data, total_tokens_processed, id_to_param_name_map, normalization_scope="layer", vmin=-5.0, vmax=5.0, **kwargs):
    df = _prepare_base_data(data, token_idx, id_to_param_name_map)
    config = _get_plot_config(view_mode)
    
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.text(0.5, 0.5, f"No data available for token_idx: {token_idx}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Token {token_idx}: {view_mode}")
        plt.tight_layout()
        return fig

    title = f"Token {token_idx}: {view_mode} ({normalization_scope.capitalize()})"
    value_col = "activation" # Default value

    # --- Z-Score based views ---
    if "Z-Score" in view_mode or "SPS" in view_mode:
        df = _calculate_z_scores(df.copy(), scope=normalization_scope)
        if view_mode == "Activation Z-Score":
            value_col = "activation_z_score"
        elif view_mode == "Gradient Z-Score":
            value_col = "grad_norm_z_score"
        elif "SPS" in view_mode:
            df["SPS"] = df["activation_z_score"] - df["grad_norm_z_score"]
            value_col = "SPS"
            if view_mode == "∫SPS":
                all_df = _prepare_base_data(data, slice(None), id_to_param_name_map)
                all_df = _calculate_z_scores(all_df, scope=normalization_scope)
                all_df["SPS"] = all_df["activation_z_score"] - all_df["grad_norm_z_score"]
                
                integrated_sps = all_df.groupby(["param_name", "block_idx"])["SPS"].sum().reset_index()
                integrated_sps.rename(columns={'SPS': 'integrated_SPS'}, inplace=True)
                
                df = df.merge(integrated_sps, on=['param_name', 'block_idx'], how='left')
                df['integrated_SPS'] = df['integrated_SPS'].fillna(0)
                
                # Normalize for visualization
                mean_isps = df['integrated_SPS'].mean()
                std_isps = df['integrated_SPS'].std()
                df['integrated_SPS'] = (df['integrated_SPS'] - mean_isps) / std_isps if std_isps > 1e-9 else 0.0
                
                value_col = 'integrated_SPS'
                title = f"Integrated SPS (Tokens: {total_tokens_processed}, Norm: {normalization_scope.capitalize()})"
    
    # --- Raw value views ---
    else:
        value_col = "activation" if "Activation" in view_mode else "gradient_norm"
        config['vmin'] = df[value_col].min()
        config['vmax'] = df[value_col].max()

    if value_col not in df.columns:
        print(f"DEBUG: Column '{value_col}' not found. Defaulting to 'activation'.")
        value_col = 'activation'

    return create_heatmap(df, value_col, title, config['cmap'], vmin, vmax)
