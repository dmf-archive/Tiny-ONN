import argparse
import sqlite3

import gradio as gr
import pandas as pd

from ui.gradio_ui import create_visualization_ui
from utils.plotting import update_plot


def get_total_tokens(db_conn):
    """Get the maximum token index from the database."""
    try:
        df = pd.read_sql_query("SELECT MAX(token_idx) FROM block_metrics", db_conn)
        return 0 if df.empty or pd.isna(df.iloc[0, 0]) else int(df.iloc[0, 0])
    except (pd.io.sql.DatabaseError, IndexError):
        return 0

def build_replay_ui(db_path="tiny_onn_metrics.db"):
    """Builds the Gradio UI for replaying and analyzing stored data."""
    db_conn = sqlite3.connect(db_path, check_same_thread=False)

    def get_plot_for_replay_ui(*args):
        """Wrapper to call update_plot with UI values for the replay app."""
        token_idx, view_mode, vmin, vmax, w_act, w_grad = args
        total_tokens = get_total_tokens(db_conn)
        
        override_kwargs = {
            'vmin': vmin,
            'vmax': vmax,
            'w_act': w_act,
            'w_grad': w_grad
        }
        
        return update_plot(
            int(token_idx), 
            view_mode, 
            db_conn, 
            total_tokens, 
            **override_kwargs
        )

    with gr.Blocks(css=".gradio-container {max-width: 800px; margin: auto;}") as demo:
        gr.Markdown("# Tiny-ONN: Replay & Analysis")
        
        with gr.Row():
            # The create_visualization_ui function now builds the right-hand side
            create_visualization_ui(get_plot_for_replay_ui, get_total_tokens, db_conn)

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay and analyze Tiny-ONN metrics.")
    parser.add_argument("--db_path", type=str, default="tiny_onn_metrics.db",
                        help="Path to the SQLite database file (e.g., tiny_onn_pruned_metrics.db)")
    args = parser.parse_args()

    replay_demo = build_replay_ui(db_path=args.db_path)
    replay_demo.launch()
