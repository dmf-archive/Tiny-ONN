import json
import os
import re

import gradio as gr
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Plotting & Data Handling ---

PLOT_CONFIG = {
    "activation": {"vmin": 0, "vmax": 450, "cmap": "viridis"},
    "gradient_norm": {"vmin": 0, "vmax": 1e-7, "cmap": "plasma"},
    "pi_diff": {"vmin": -50, "vmax": 450, "cmap": "coolwarm", "gradient_weight": 1.0e8},
    "absmax": {"vmin": 0, "vmax": 256, "cmap": "cividis"},
}

VIEW_MODE_MAP = {
    "Activation": "activation",
    "Gradient Norm": "gradient_norm",
    "PI Diff Scatter": "pi_diff",
    "AbsMax": "absmax",
}

QUANTIZATION_SCALE = 1000
id_to_param_name_map: dict[str, str] = {}


def get_plot_config(view_mode, **kwargs):
    config_key = VIEW_MODE_MAP.get(view_mode, "pi_diff")
    config = PLOT_CONFIG.get(config_key, {}).copy()
    config.update(kwargs)
    return config


def fetch_and_prepare_data(db_conn, token_idx):
    global id_to_param_name_map
    if not id_to_param_name_map:
        map_path = os.path.join("data/metadata", "param_name_map.json")
        if os.path.exists(map_path):
            with open(map_path, encoding="utf-8") as f:
                id_to_param_name_map = json.load(f)
        else:
            print("DEBUG: Parameter name map not found.")
            return None

    query = f"SELECT param_name, block_idx, activation, grad_norm, absmax FROM block_metrics WHERE token_idx = {token_idx}"
    df = pd.read_sql_query(query, db_conn)

    if df.empty:
        return None

    df["param_name"] = df["param_name"].astype(str).map(id_to_param_name_map)
    df.dropna(subset=["param_name"], inplace=True)

    for col in ["activation", "grad_norm", "absmax"]:
        df[col] = df[col] / QUANTIZATION_SCALE

    def get_layer_info(param_name):
        match = re.search(r"model\.layers\.(\d+)", param_name)
        if match:
            return int(match.group(1))
        elif "embed_tokens" in param_name:
            return -1
        elif "lm_head" in param_name:
            return 999
        elif "model.norm" in param_name:
            return 998
        return -2

    def get_layer_type(param_name):
        parts = param_name.split(".")
        if "layers" in parts:
            try:
                idx = parts.index("layers")
                return ".".join(parts[idx + 2 :])
            except (ValueError, IndexError):
                return param_name
        return param_name

    df["layer_index"] = df["param_name"].apply(get_layer_info)
    df["layer_type"] = df["param_name"].apply(get_layer_type)

    if not df[df["layer_index"] < 900].empty:
        max_real_layer = df[df["layer_index"] < 900]["layer_index"].max()
        if 998 in df["layer_index"].unique():
            df.loc[df["layer_index"] == 998, "layer_index"] = max_real_layer + 1
        if 999 in df["layer_index"].unique():
            df.loc[df["layer_index"] == 999, "layer_index"] = max_real_layer + 2

    df = df[df["layer_index"] != -2].copy()
    df.dropna(subset=["layer_index", "layer_type"], inplace=True)
    df["layer_index"] = df["layer_index"].astype(int)
    df = df.sort_values(by=["layer_index", "layer_type", "block_idx"])

    grouped = df.groupby("layer_index")

    def normalize_group(group):
        for metric in ["activation", "grad_norm"]:
            mean = group[metric].mean()
            std = group[metric].std()
            group[f"{metric}_z_score"] = (
                (group[metric] - mean) / std if std > 1e-9 else 0.0
            )
        return group

    df = grouped.apply(normalize_group, include_groups=False).reset_index()

    layer_type_map = {t: i for i, t in enumerate(df["layer_type"].unique())}
    df["layer_type_idx"] = df["layer_type"].map(layer_type_map)
    return df


def create_heatmap(df, value_col, title, cmap, vmin, vmax):
    fig, ax = plt.subplots(figsize=(14, 10))

    if df.empty:
        ax.text(
            0.5,
            0.5,
            "No data available for this token.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        plt.tight_layout()
        return fig

    unique_layer_types = sorted(
        df["layer_type"].unique(),
        key=lambda x: df[df["layer_type"] == x]["layer_type_idx"].iloc[0],
    )
    x_coords = df["layer_type_idx"] + (
        df["block_idx"] / (df["block_idx"].max() + 1) * 0.8 - 0.4
    )

    scatter = ax.scatter(
        x_coords,
        df["layer_index"],
        c=df[value_col],
        cmap=cmap,
        s=10,
        alpha=0.7,
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
    )

    ax.set_xticks(range(len(unique_layer_types)))
    ax.set_xticklabels(unique_layer_types, rotation=45, ha="right")
    ax.set_xlabel("Layer Type")
    ax.set_ylabel("Layer Index")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_yticks(np.arange(0, df["layer_index"].max() + 1, 1))
    ax.invert_yaxis()

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Value")
    plt.tight_layout()
    return fig


def update_plot(token_idx, view_mode, db_conn, total_tokens_processed, **kwargs):
    df = fetch_and_prepare_data(db_conn, token_idx)
    config = get_plot_config(view_mode, **kwargs)
    title = f"Token {token_idx}: {view_mode}"

    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.text(
            0.5,
            0.5,
            f"No data available for token_idx: {token_idx}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
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
        "S_p": "S_p",
    }
    value_col = value_col_map.get(view_mode, "activation")

    if view_mode in ["Activation", "Gradient Norm", "AbsMax"]:
        if value_col in df.columns and not df[value_col].empty:
            config["vmin"] = 0.0
            config["vmax"] = df[value_col].max() * 1.1
        else:
            config["vmin"], config["vmax"] = 0.0, 1.0

    if view_mode == "S_p":
        df["S_p"] = df["activation_z_score"] - df["grad_norm_z_score"]
        title = f"Token {token_idx}: S_p (Fixed Weights)"

    if value_col not in df.columns:
        value_col = "activation"

    return create_heatmap(
        df, value_col, title, config["cmap"], config["vmin"], config["vmax"]
    )


# --- UI Creation ---


def create_main_ui(
    process_input_fn, update_plot_fn, get_total_tokens_fn, is_replay_mode=False
):
    with gr.Blocks(css=".gradio-container {max-width: 1200px; margin: auto;}") as demo:
        gr.Markdown("# Digital fMRI: A Tiny-ONN Pilot Study")
        gr.Markdown("This interface provides a real-time visualization of a large language model's internal states. It operates by conducting a per-token (time-slice) scan of the model's activation values (digital neurons) and gradients (approximating hemodynamic responses), offering insights into its cognitive processes.")

        total_tokens = get_total_tokens_fn()

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    height=400,
                    label="Chat History",
                    type="messages",
                    bubble_full_width=False,
                )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=150):
                        use_fmri_checkbox = gr.Checkbox(
                            label="Enable fMRI Scan",
                            value=True,
                            info="Uncheck for pure inference",
                            visible=not is_replay_mode,
                        )
                    with gr.Column(scale=1, min_width=150):
                        no_think_checkbox = gr.Checkbox(
                            label="No Think Mode",
                            value=True,
                            info="Force model to avoid reasoning",
                            visible=not is_replay_mode,
                        )
                    with gr.Column(scale=2):
                         submit_btn = gr.Button("Send", visible=not is_replay_mode, variant="primary")

                msg = gr.Textbox(
                    label="Your Message",
                    visible=not is_replay_mode,
                    placeholder="Type your message here...",
                    lines=2,
                )

            with gr.Column(scale=1):
                plot_output = gr.Plot(label="Neural Activity Heatmap")
                view_selector = gr.Radio(
                    [
                        "Activation",
                        "Gradient Norm",
                        "AbsMax",
                        "Activation Z-Score",
                        "Gradient Z-Score",
                        "Synergistic Prediction Score (SPS)",
                        "∫SPS",
                    ],
                    label="Select View",
                    value="Synergistic Prediction Score (SPS)",
                )
                normalization_selector = gr.Radio(
                    ["layer", "global"],
                    label="Normalization Scope",
                    value="layer",
                )
                time_slider = gr.Slider(
                    minimum=0,
                    maximum=max(0, total_tokens - 1),
                    step=1,
                    value=max(0, total_tokens - 1),
                    label="Timeline (Token Index)",
                    interactive=True,
                )
                with gr.Accordion("Plotting Configuration", open=True):
                    vmin_slider = gr.Number(label="Min Value (vmin)", value=-5.0)
                    vmax_slider = gr.Number(label="Max Value (vmax)", value=5.0)

        plot_update_inputs = [
            time_slider,
            view_selector,
            vmin_slider,
            vmax_slider,
            normalization_selector,
        ]
        for control in plot_update_inputs:
            control.change(update_plot_fn, plot_update_inputs, [plot_output])

        if not is_replay_mode:
            submit_inputs = [
                msg,
                chatbot,
                view_selector,
                vmin_slider,
                vmax_slider,
                use_fmri_checkbox,
                no_think_checkbox,
            ]
            submit_outputs = [msg, chatbot, plot_output, time_slider]
            submit_btn.click(process_input_fn, submit_inputs, submit_outputs)
            msg.submit(process_input_fn, submit_inputs, submit_outputs)

    return demo
