import os
import sqlite3
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from inference import run_forward_pass
from utils import log_message, update_plot

# --- Global State ---
model = None
tokenizer = None
db_conn = None
total_tokens_processed = 0

# --- Core Application Logic ---

def setup_database(db_path="tiny_onn_metrics.db", memory=False):
    """Initializes a SQLite database, either in-memory or file-based."""
    global db_conn
    if memory:
        db_conn = sqlite3.connect(":memory:", check_same_thread=False)
        log_message("In-memory SQLite database initialized.")
    else:
        if os.path.exists(db_path):
            os.remove(db_path)
        db_conn = sqlite3.connect(db_path, check_same_thread=False)
        log_message(f"File-based SQLite database initialized at {db_path}.")

    cursor = db_conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS block_metrics (
        token_idx INTEGER,
        param_name TEXT,
        block_idx INTEGER,
        activation REAL,
        grad_norm REAL,
        absmax REAL,
        PRIMARY KEY (token_idx, param_name, block_idx)
    )
    """)
    db_conn.commit()
    return db_conn

def load_model_and_tokenizer(model_name="Qwen/Qwen3-1.7B"):
    """Loads the specified model and tokenizer with 4-bit quantization."""
    global model, tokenizer
    cache_path = os.path.join(os.getcwd(), "weights")
    os.makedirs(cache_path, exist_ok=True)

    log_message(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path)
    log_message("Tokenizer loaded.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    log_message(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        cache_dir=cache_path,
        device_map="auto"
    )
    log_message("Model loaded.")

    from inference.block_level_capture import patch_model_for_block_level_capture
    patch_model_for_block_level_capture(model)
    log_message("Model patched for block-level data capture.")
    return model, tokenizer

def run_analysis_pipeline(
    current_model, current_tokenizer, user_message, history, current_db_conn
):
    """
    Runs the core data generation and storage pipeline.
    This function is separated from Gradio logic for testability.
    """
    global total_tokens_processed
    
    if history is None:
        history = []

    # 1. Forward pass to generate response and capture block-level data
    generated_ids, per_token_block_data, _ = run_forward_pass(
        current_model, current_tokenizer, user_message, history
    )
    final_response = current_tokenizer.decode(generated_ids, skip_special_tokens=True)
    history.append([user_message, final_response])

    # 2. Store captured data in the database
    cursor = current_db_conn.cursor()
    num_generated_tokens = len(per_token_block_data)

    for i in range(num_generated_tokens):
        token_block_data = per_token_block_data[i]
        current_token_idx = total_tokens_processed + i
        
        for param_name, metrics in token_block_data.items():
            activations = metrics.get("activation", [])
            gradients = metrics.get("gradient", [])
            weights = metrics.get("weight", [])
            
            max_blocks = max(len(activations), len(gradients), len(weights))

            for block_idx in range(max_blocks):
                activation_val = activations[block_idx] if block_idx < len(activations) else 0.0
                grad_norm_val = gradients[block_idx] if block_idx < len(gradients) else 0.0
                absmax_val = weights[block_idx] if block_idx < len(weights) else 0.0

                cursor.execute("""
                INSERT OR REPLACE INTO block_metrics (token_idx, param_name, block_idx, activation, grad_norm, absmax)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (current_token_idx, param_name, block_idx, activation_val, grad_norm_val, absmax_val))

    current_db_conn.commit()
    total_tokens_processed += num_generated_tokens
    log_message(f"Stored metrics for {num_generated_tokens} tokens. Total processed: {total_tokens_processed}")
    
    return history, final_response

# --- Gradio Interface Logic ---

def process_input_for_gradio(user_message, history, current_token_idx, view_mode):
    """Gradio callback to process user input and update the interface."""
    global model, tokenizer, db_conn, total_tokens_processed
    
    updated_history, _ = run_analysis_pipeline(
        model, tokenizer, user_message, history, db_conn
    )

    final_plot = update_plot(
        total_tokens_processed - 1, view_mode, db_conn, total_tokens_processed
    )
    
    slider_update = gr.update(
        maximum=max(1, total_tokens_processed - 1), value=total_tokens_processed - 1
    )
    
    yield "", updated_history, final_plot, slider_update

def update_plot_from_history(token_idx, view_mode):
    """Gradio callback to update the plot based on the slider."""
    global db_conn, total_tokens_processed
    return update_plot(token_idx, view_mode, db_conn, total_tokens_processed)

def build_gradio_ui():
    """Builds and returns the Gradio interface."""
    with gr.Blocks() as demo:
        gr.Markdown("# Tiny-ONN Pilot Study: Real-time Analysis")
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(height=400, label="Chat History")
                msg = gr.Textbox(label="Your Message")
                gr.ClearButton([msg, chatbot])
                submit_btn = gr.Button("Send")
            with gr.Column(scale=1):
                plot_output = gr.Plot(label="Visualization")
                view_selector = gr.Radio(
                    ["Activation", "Gradient Norm", "AbsMax", "PI Diff Scatter"],
                    label="Select View",
                    value="Activation"
                )
                time_slider = gr.Slider(
                    minimum=0, maximum=1, step=1, value=0,
                    label="Timeline (Token Index)", interactive=True
                )
        
        # Event Listeners
        submit_action = msg.submit(
            process_input_for_gradio, 
            [msg, chatbot, time_slider, view_selector], 
            [msg, chatbot, plot_output, time_slider]
        )
        submit_btn.click(
            process_input_for_gradio, 
            [msg, chatbot, time_slider, view_selector], 
            [msg, chatbot, plot_output, time_slider],
            cancels=[submit_action]
        )
        time_slider.change(
            update_plot_from_history, [time_slider, view_selector], [plot_output]
        )
        view_selector.change(
            update_plot_from_history, [time_slider, view_selector], [plot_output]
        )
    return demo

def main():
    """Main function to set up and launch the Gradio app."""
    setup_database()
    load_model_and_tokenizer()
    demo = build_gradio_ui()
    demo.launch(share=False)

if __name__ == "__main__":
    main()
