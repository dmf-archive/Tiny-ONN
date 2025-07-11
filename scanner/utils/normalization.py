
import pandas as pd


def process_token_data(token_data: dict[str, dict]) -> pd.DataFrame:
    if not token_data:
        return pd.DataFrame()

    # Convert the dictionary to a list of records
    records = []
    for param_name, metrics in token_data.items():
        num_blocks = len(metrics.get("activation", []))
        # Ensure all metrics have the same length, providing defaults if missing
        activations = metrics.get("activation", [0.0] * num_blocks)
        gradients = metrics.get("gradient", [0.0] * num_blocks)
        weights = metrics.get("weight", [0.0] * num_blocks)

        for i in range(num_blocks):
            records.append({
                "param_name": param_name,
                "block_idx": i,
                "activation": activations[i],
                "grad_norm": gradients[i],
                "absmax": weights[i]
            })

    df = pd.DataFrame(records)

    # --- 1. Extract Layer Info ---
    layer_info = df['param_name'].str.extract(r'model\.layers\.(\d+)\.(.+?)\.weight')
    if layer_info.empty:
        return df # Return early if no layer info can be extracted

    df['layer_index'] = pd.to_numeric(layer_info[0])
    df.dropna(subset=['layer_index'], inplace=True)
    df['layer_index'] = df['layer_index'].astype(int)

    # --- 2. Calculate Layer-wise Z-Scores ---
    grouped = df.groupby('layer_index')

    def normalize_group(group):
        for metric in ['activation', 'grad_norm']:
            mean = group[metric].mean()
            std = group[metric].std()
            if std > 1e-9:  # Use a small epsilon to avoid division by zero
                group[f'{metric}_z_score'] = (group[metric] - mean) / std
            else:
                group[f'{metric}_z_score'] = 0.0
        return group

    df_normalized = grouped.apply(normalize_group)
    df_normalized.reset_index(drop=True, inplace=True)

    # --- 3. Calculate S_p Score ---
    # Rename for clarity and consistency with replay_analysis
    df_normalized.rename(columns={
        'activation_z_score': 'activation_z_score',
        'grad_norm_z_score': 'gradient_z_score'
    }, inplace=True)

    if 'activation_z_score' in df_normalized.columns and 'gradient_z_score' in df_normalized.columns:
        df_normalized['S_p'] = df_normalized['activation_z_score'] - df_normalized['gradient_z_score']
    else:
        # Ensure column exists even if calculation fails
        df_normalized['S_p'] = 0.0


    return df_normalized
