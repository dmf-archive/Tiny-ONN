from datetime import datetime
from pathlib import Path

import gradio as gr
import torch

from .core.scheduler import Scheduler
from .mscan import load_mscan_file
from .utils.plotting import plot_scan_results


def live_scan(model_name, text_input):
    if not model_name or not text_input:
        return None, "Model name and text input are required."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler = Scheduler(model_name=model_name, device=device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{model_name.replace('/', '--')}_live_{timestamp}"
    output_path = Path("data/scans") / output_filename

    try:
        scheduler.scan_text(text_input, output_path)
        metadata, data = load_mscan_file(output_path)
        if data is None:
            return None, "Scan completed, but no data was generated."

        fig = plot_scan_results(metadata, data)
        return fig, f"Scan complete. Results saved to {output_path.with_suffix('.mscan')}"
    except Exception as e:
        return None, f"An error occurred: {e}"

def replay_scan(mscan_file):
    if mscan_file is None:
        return None, "Please upload a .mscan file."

    try:
        filepath = Path(mscan_file.name)
        metadata, data = load_mscan_file(filepath)
        if data is None:
            return None, "File loaded, but it contains no data records."

        fig = plot_scan_results(metadata, data)
        return fig, f"Successfully replayed scan from {filepath.name}"
    except Exception as e:
        return None, f"Failed to load or plot file: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# ΔSC Scanner")

    with gr.Tabs():
        with gr.TabItem("Live Scan"):
            with gr.Row():
                model_name_input = gr.Textbox(label="Model Name", value="Qwen/Qwen3-1.7B")
                text_input = gr.Textbox(label="Text Input", lines=5, placeholder="Enter text to scan...")
            scan_button = gr.Button("Run Scan")
            live_output_plot = gr.Plot()
            live_status_output = gr.Textbox(label="Status")

            scan_button.click(
                live_scan,
                inputs=[model_name_input, text_input],
                outputs=[live_output_plot, live_status_output]
            )

        with gr.TabItem("Replay Scan"):
            file_input = gr.File(label="Upload .mscan File")
            replay_button = gr.Button("Replay")
            replay_output_plot = gr.Plot()
            replay_status_output = gr.Textbox(label="Status")

            replay_button.click(
                replay_scan,
                inputs=[file_input],
                outputs=[replay_output_plot, replay_status_output]
            )

if __name__ == "__main__":
    demo.launch()
