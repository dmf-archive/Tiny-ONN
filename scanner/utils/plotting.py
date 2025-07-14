from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_scan_results(metadata: dict[str, Any], data: np.ndarray) -> plt.Figure:
    df = pd.DataFrame(data)

    id_to_name = metadata.get("id_to_param_name_map", {})
    df["param_name"] = df["param_id"].astype(str).map(id_to_name)

    # Calculate z-scores
    df["activation_z"] = (df["activation"] - df["activation"].mean()) / df["activation"].std()
    df["grad_norm_z"] = (df["grad_norm"] - df["grad_norm"].mean()) / df["grad_norm"].std()

    # Calculate DeltaSC
    df["delta_sc"] = df["activation_z"] - df["grad_norm_z"]

    # Aggregate by parameter name
    agg_df = df.groupby("param_name").agg({
        "activation": "mean",
        "grad_norm": "mean",
        "delta_sc": "mean"
    }).reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(15, 20), sharex=True)
    fig.suptitle(f"Scan Analysis for {metadata.get('model_name', 'Unknown Model')}", fontsize=16)

    # Plot 1: Mean Activation
    axes[0].bar(agg_df["param_name"], agg_df["activation"], color="skyblue")
    axes[0].set_title("Mean Activation Norm per Parameter")
    axes[0].set_ylabel("Mean Activation Norm")
    axes[0].tick_params(axis='x', rotation=90)

    # Plot 2: Mean Gradient Norm
    axes[1].bar(agg_df["param_name"], agg_df["grad_norm"], color="salmon")
    axes[1].set_title("Mean Gradient Norm per Parameter")
    axes[1].set_ylabel("Mean Gradient Norm")
    axes[1].tick_params(axis='x', rotation=90)

    # Plot 3: DeltaSC
    axes[2].bar(agg_df["param_name"], agg_df["delta_sc"], color="lightgreen")
    axes[2].set_title("Mean ΔSC (Synergistic Contribution) per Parameter")
    axes[2].set_ylabel("Mean ΔSC")
    axes[2].set_xlabel("Parameter Name")
    axes[2].axhline(0, color='grey', linewidth=0.8, linestyle='--')
    axes[2].tick_params(axis='x', rotation=90)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
