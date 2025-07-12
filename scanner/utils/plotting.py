import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_CONFIG = {
    "activation": {
        "vmin": 0,
        "vmax": 450,
        "cmap": "viridis"
    },
    "gradient_norm": {
        "vmin": 0,
        "vmax": 1e-7,
        "cmap": "plasma"
    },
    "pi_diff": {
        "vmin": -50,
        "vmax": 450,
        "cmap": "coolwarm",
        "gradient_weight": 1.0e8
    },
    "absmax": {
        "vmin": 0,
        "vmax": 256,
        "cmap": "cividis"
    }
}

VIEW_MODE_MAP = {
    "Activation": "activation",
    "Gradient Norm": "gradient_norm",
    "PI Diff Scatter": "pi_diff",
    "AbsMax": "absmax"
}

QUANTIZATION_SCALE = 1000


def get_plot_config(view_mode, **kwargs):
    config_key = VIEW_MODE_MAP.get(view_mode, "pi_diff")
    config = PLOT_CONFIG[config_key].copy()
    config.update(kwargs)
    return config

def fetch_and_prepare_data(
    data: np.ndarray,
    token_idx: int,
    id_to_param_name_map: dict[str, str],
    normalization_scope: str = "layer",
):
    if data is None or not id_to_param_name_map:
        return None

    token_data = data[data["token_idx"] == token_idx]
    if token_data.size == 0:
        print(f"DEBUG: No data found for token_idx {token_idx}")
        return None

    df = pd.DataFrame(token_data)

    df["param_name"] = df["param_id"].astype(str).map(id_to_param_name_map)
    df.dropna(subset=["param_name"], inplace=True)

    for col in ["activation", "grad_norm"]:
        df[col] = df[col].astype(float)

    def get_layer_info(param_name):
        match = re.search(r'model\.layers\.(\d+)', param_name)
        if match:
            return int(match.group(1))
        elif 'embed_tokens' in param_name:
            return -1
        elif 'lm_head' in param_name:
            return 999
        elif 'model.norm' in param_name:
            return 998
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

    if normalization_scope == "layer":
        grouped = df.groupby("layer_index")

        def normalize_group(group):
            for metric in ["activation", "grad_norm"]:
                mean = group[metric].mean()
                std = group[metric].std()
                if std > 1e-9:
                    group[f"{metric}_z_score"] = (group[metric] - mean) / std
                else:
                    group[f"{metric}_z_score"] = 0.0
            return group

        df = grouped.apply(normalize_group, include_groups=False).reset_index()
    elif normalization_scope == "global":
        for metric in ["activation", "grad_norm"]:
            mean = df[metric].mean()
            std = df[metric].std()
            if std > 1e-9:
                df[f"{metric}_z_score"] = (df[metric] - mean) / std
            else:
                df[f"{metric}_z_score"] = 0.0

    layer_type_map = {t: i for i, t in enumerate(df['layer_type'].unique())}
    df['layer_type_idx'] = df['layer_type'].map(layer_type_map)

    print(f"DEBUG: Processed {len(df)} records for token_idx {token_idx}")
    return df

def create_heatmap(df, value_col, title, cmap, vmin, vmax):
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

def update_plot(
    token_idx,
    view_mode,
    data,
    total_tokens_processed,
    id_to_param_name_map,
    normalization_scope="layer",
    **kwargs,
):
    df = fetch_and_prepare_data(
        data, token_idx, id_to_param_name_map, normalization_scope
    )
    config = get_plot_config(view_mode, **kwargs)
    title = f"Token {token_idx}: {view_mode}"

    if df is None or df.empty:
        print(f"DEBUG: No data for token {token_idx}, returning empty plot.")
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
        "Activation Z-Score": "activation_z_score",
    "Gradient Z-Score": "grad_norm_z_score",
    "Synergistic Prediction Score (SPS)": "S_p",
    "∫SPS": "integrated_SPS"
}

    value_col = value_col_map.get(view_mode, 'activation')

    if view_mode in ["Activation", "Gradient Norm"]:
        if value_col in df.columns and not df[value_col].empty:
            config['vmin'] = 0.0
            config['vmax'] = df[value_col].max() * 1.1
        else:
            config['vmin'] = 0.0
            config['vmax'] = 1.0

    if view_mode == "Synergistic Prediction Score (SPS)":
        df['S_p'] = df['activation_z_score'] - df['grad_norm_z_score']
        title = f"Token {token_idx}: Synergistic Prediction Score (SPS)"
    elif view_mode == "∫SPS":
        all_df = pd.DataFrame(data)
        all_df["param_name"] = all_df["param_id"].astype(str).map(id_to_param_name_map)
        all_df.dropna(subset=["param_name"], inplace=True)

        for col in ["activation", "grad_norm"]:
            all_df[col] = all_df[col].astype(float)

        # Global Z-score calculation across the entire sequence
        for metric in ["activation", "grad_norm"]:
            mean = all_df[metric].mean()
            std = all_df[metric].std()
            if std > 1e-9:
                all_df[f"{metric}_z_score"] = (all_df[metric] - mean) / std
            else:
                all_df[f"{metric}_z_score"] = 0.0

        all_df["S_p"] = all_df["activation_z_score"] - all_df["grad_norm_z_score"]

        integrated_sps = (
            all_df.groupby(["param_name", "block_idx"])["S_p"].sum().reset_index()
        )
        integrated_sps.rename(columns={'S_p': 'integrated_SPS'}, inplace=True)

        df = df.merge(integrated_sps, on=['param_name', 'block_idx'], how='left')
        df['integrated_SPS'] = df['integrated_SPS'].fillna(0)

        mean_integrated_sps = df['integrated_SPS'].mean()
        std_integrated_sps = df['integrated_SPS'].std()
        if std_integrated_sps > 1e-9:
            df['integrated_SPS'] = (df['integrated_SPS'] - mean_integrated_sps) / std_integrated_sps
        else:
            df['integrated_SPS'] = 0.0

        title = f"Integrated Synergistic Prediction Score (∫SPS) for entire sequence (Tokens: {total_tokens_processed})"
        value_col = 'integrated_SPS'
        config['cmap'] = 'coolwarm'
        # Use the provided vmin/vmax from the UI instead of hardcoded values
        config['vmin'] = kwargs.get('vmin', -5.0)
        config['vmax'] = kwargs.get('vmax', 5.0)

    if value_col not in df.columns:
        print(f"DEBUG: Column '{value_col}' not found in DataFrame. Defaulting to 'activation'.")
        value_col = 'activation'

    return create_heatmap(df, value_col, title, config['cmap'], config['vmin'], config['vmax'])
