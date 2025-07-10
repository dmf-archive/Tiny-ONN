import os
import json
import os
import sqlite3

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import QUANTIZATION_SCALE
from inference.backward_pass import run_per_token_backward_pass
from inference.forward_pass import run_forward_pass_and_capture_activations
from ui.gradio_ui import create_visualization_ui
from utils import log_message, update_plot

model = None
tokenizer = None
db_conn = None
total_tokens_processed = 0
param_name_to_id = {}
id_to_param_name = {}

def setup_database(db_path="tiny_onn_metrics.db", memory=False):
    global db_conn
    path = ":memory:" if memory else db_path
    if not memory and os.path.exists(path):
        os.remove(path) # Ensure a clean database for schema change
    
    conn = sqlite3.connect(path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS block_metrics (
        token_idx INTEGER,
        param_name TEXT,
        block_idx INTEGER,
        activation INTEGER,
        grad_norm INTEGER,
        absmax INTEGER,
        PRIMARY KEY (token_idx, param_name, block_idx)
    )
    """)
    conn.commit()
    
    if not memory:
        db_conn = conn
    return conn

def load_model_and_tokenizer(model_name="Qwen/Qwen3-1.7B"):
    global model, tokenizer, param_name_to_id, id_to_param_name
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
    log_message("Model compilation skipped on Windows due to Triton dependency.")

    # Load param_name_map
    map_path = os.path.join("metadata", "param_name_map.json")
    if os.path.exists(map_path):
        with open(map_path, 'r', encoding='utf-8') as f:
            id_to_param_name = json.load(f)
        param_name_to_id = {name: int(id_str) for id_str, name in id_to_param_name.items()}
        log_message("Parameter name map loaded.")
    else:
        log_message("Parameter name map not found. Please run scripts/generate_param_map.py first.")

    return model, tokenizer

def store_per_token_data(per_token_data, start_idx, current_db_conn):
    cursor = current_db_conn.cursor()
    for i, token_data in enumerate(per_token_data):
        current_token_idx = start_idx + i
        for param_name, metrics in token_data.items():
            param_id = param_name_to_id.get(param_name)
            if param_id is None:
                log_message(f"Warning: param_name '{param_name}' not found in map. Skipping.")
                continue

            activations = metrics.get("activation", [])
            gradients = metrics.get("gradient", [0.0] * len(activations))
            weights = metrics.get("weight", [0.0] * len(activations))
            for block_idx, (act, grad, w) in enumerate(zip(activations, gradients, weights, strict=False)):
                # Quantize values to integers before storing
                quantized_act = int(act * QUANTIZATION_SCALE)
                quantized_grad = int(grad * QUANTIZATION_SCALE)
                quantized_w = int(w * QUANTIZATION_SCALE)
                cursor.execute("""
                INSERT OR REPLACE INTO block_metrics (token_idx, param_name, block_idx, activation, grad_norm, absmax)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (current_token_idx, param_id, block_idx, quantized_act, quantized_grad, quantized_w))
    current_db_conn.commit()

def run_analysis_pipeline(current_model, current_tokenizer, user_message, history, current_db_conn, tokens_processed_offset):
    global total_tokens_processed
    log_message("--- [START] Per-Token Forward Pass & Activation Capture ---")
    history.append({"role": "user", "content": user_message})
    final_response, per_token_activation_data, full_sequence_ids, prompt_len = run_forward_pass_and_capture_activations(
        current_model, current_tokenizer, user_message, history
    )
    history.append({"role": "assistant", "content": final_response})
    log_message("--- [END] Forward Pass ---")

    if per_token_activation_data:
        log_message("--- [START] Per-Token Backward Pass & Gradient Capture ---")
        per_token_activation_data = run_per_token_backward_pass(
            current_model, full_sequence_ids, per_token_activation_data, prompt_len
        )
        log_message("--- [END] Backward Pass ---")

    num_generated_tokens = len(per_token_activation_data)
    store_per_token_data(per_token_activation_data, tokens_processed_offset, current_db_conn)
    
    total_tokens_processed = tokens_processed_offset + num_generated_tokens
    
    return history, final_response

def process_input_for_gradio(user_message, history, view_mode, vmin, vmax, w_act, w_grad):
    global model, tokenizer, db_conn, total_tokens_processed
    
    history, _ = run_analysis_pipeline(
        model, tokenizer, user_message, history, db_conn, total_tokens_processed
    )
    
    last_token_idx = total_tokens_processed - 1
    override_kwargs = {'vmin': vmin, 'vmax': vmax, 'w_act': w_act, 'w_grad': w_grad}
    final_plot = update_plot(last_token_idx, view_mode, db_conn, total_tokens_processed, **override_kwargs)
    slider_update = gr.update(maximum=max(0, last_token_idx), value=last_token_idx)
    
    return "", history, final_plot, slider_update

def update_plot_from_history(token_idx, view_mode, vmin, vmax, w_act, w_grad):
    global db_conn, total_tokens_processed
    override_kwargs = {'vmin': vmin, 'vmax': vmax, 'w_act': w_act, 'w_grad': w_grad}
    return update_plot(int(token_idx), view_mode, db_conn, total_tokens_processed, **override_kwargs)

def build_gradio_ui():
    with gr.Blocks(css=".gradio-container {max-width: 800px; margin: auto;}") as demo:
        gr.Markdown("# Tiny-ONN: Digital fMRI (Live Analysis)")
        
        def get_total_tokens_for_live(db_conn_ignored):
            # In live mode, the max token index is tracked by a global variable
            global total_tokens_processed
            return total_tokens_processed

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(height=400, label="Chat History", type="messages", bubble_full_width=False)
                msg = gr.Textbox(label="Your Message")
                submit_btn = gr.Button("Run fMRI")

            # Use the shared UI component for the visualization part
            plot_output, time_slider, view_selector, vmin_slider, vmax_slider, w_act_slider, w_grad_slider = create_visualization_ui(
                update_plot_from_history, 
                get_total_tokens_for_live,
                db_conn
            )

        # Link the submit button to the main processing function
        submit_inputs = [msg, chatbot, view_selector, vmin_slider, vmax_slider, w_act_slider, w_grad_slider]
        submit_outputs = [msg, chatbot, plot_output, time_slider]
        submit_btn.click(process_input_for_gradio, submit_inputs, submit_outputs)
        
    return demo

def main():
    setup_database()
    load_model_and_tokenizer()
    demo = build_gradio_ui()
    demo.launch(share=False)

if __name__ == "__main__":
    main()
