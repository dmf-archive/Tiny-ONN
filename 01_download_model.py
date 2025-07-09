import os
import random
import sqlite3

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Global variables
model = None
tokenizer = None
activation_data: dict = {}
gradient_data: dict = {}
db_conn = None
total_tokens_processed = 0

def quantize_to_int8(v: float) -> int:
    """Non-linearly quantizes a float to an 8-bit integer."""
    # Apply a non-linear transformation (e.g., log) to handle large range
    v_log = np.log1p(abs(v))
    # Normalize to 0-1, using a reasonable max value (e.g., log(100))
    v_norm = min(v_log / np.log1p(100), 1.0)
    # Scale to 0-255
    return int(v_norm * 255)

def dequantize_from_int8(v_int: int) -> float:
    """Dequantizes an 8-bit integer back to a float."""
    v_norm = v_int / 255.0
    v_log = v_norm * np.log1p(100)
    return np.expm1(v_log)

def setup_database():
    """Initializes an in-memory SQLite database with integer storage."""
    global db_conn
    db_conn = sqlite3.connect(":memory:", check_same_thread=False)
    cursor = db_conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS param_metrics (
        token_idx INTEGER,
        param_name TEXT,
        activation INTEGER,
        grad_norm INTEGER,
        pi_diff INTEGER,
        PRIMARY KEY (token_idx, param_name)
    )
    """)
    db_conn.commit()
    print("In-memory SQLite database initialized for quantized storage.")

def run_knowledge_extraction():
    global model, tokenizer

    model_name = "Qwen/Qwen3-4B"
    cache_path = os.path.join(os.getcwd(), "weights")
    
    print(f"Starting model and tokenizer setup for: {model_name}")
    print(f"Files will be saved to: {cache_path}")

    os.makedirs(cache_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_path,
    )
    print("Tokenizer loaded successfully.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        cache_dir=cache_path,
        device_map="auto"
    )
    print("Model loaded successfully.")
    
    print("\n--- Model Verification ---")
    print(f"Model Name: {model.config.name_or_path}")
    print(f"Model is on device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    print(f"Model loaded in 4-bit: {model.is_loaded_in_4bit}")

    setup_database()
    register_hooks(model)
    print("Hooks registered.")

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Tiny-ONN Pilot Study
            Activation | Grad Norm | PI Diff | Scatter Viz
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(height=400, label="Chat History")
                msg = gr.Textbox(label="Your Message")
                gr.ClearButton([msg, chatbot])
                submit_btn = gr.Button("Send")
            with gr.Column(scale=1):
                plot_output = gr.Plot(label="Visualization")
                view_selector = gr.Radio(
                    ["Activation", "Gradient Norm", "PI Diff Scatter"],
                    label="Select View",
                    value="PI Diff Scatter"
                )
                time_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=1,
                    value=0,
                    label="Timeline (Token Index)",
                    interactive=True
                )
        
        msg.submit(process_input_and_visualize, [msg, chatbot, time_slider, view_selector], [msg, chatbot, plot_output, time_slider])
        submit_btn.click(process_input_and_visualize, [msg, chatbot, time_slider, view_selector], [msg, chatbot, plot_output, time_slider])
        time_slider.change(update_plot_from_history, [time_slider, view_selector], [plot_output])
        view_selector.change(update_plot_from_history, [time_slider, view_selector], [plot_output])

    demo.launch(share=False)

def register_hooks(model):
    """Registers only the forward hooks for activation capturing."""
    global activation_data
    activation_data.clear()

    # Clear any existing hooks to prevent duplicates
    for module in model.modules():
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()
        if hasattr(module, '_backward_hooks'):
            module._backward_hooks.clear()

    for name, module in model.named_modules():
        if name.endswith('mlp.down_proj'):
            # Capture the *input* to the down_proj layer, which is the result of the FFN's activation function
            module.register_forward_hook(lambda mod, inp, outp, n=name: activation_data.setdefault(n, []).append(inp[0].detach().cpu().float()))

def process_input_and_visualize(user_message, history, current_token_idx, view_mode):
    global model, tokenizer, activation_data, db_conn, total_tokens_processed

    # --- Unified Generation and Analysis Loop ---
    print("\n--- [START] Unified Generation & Analysis ---")
    
    # 1. Prepare Inputs
    if history is None:
        history = []
    
    messages = []
    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        if assistant is not None:
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    input_ids = model_inputs.input_ids
    
    # 2. Manual Generation Loop with KV Cache
    generated_ids = []
    per_token_activations = []
    past_key_values = None
    model.eval()
    with torch.no_grad():
        # First, process the prompt to get the initial KV cache
        prompt_outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = prompt_outputs.past_key_values
        next_token_logits = prompt_outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # Start the generation loop
        for _ in tqdm(range(256), desc="[Unified] Generating & Logging"):
            activation_data.clear()
            
            # Use KV cache for efficient single-token forward pass
            outputs = model(input_ids=next_token_id, past_key_values=past_key_values, use_cache=True)
            
            # Capture activations for this step
            step_activations = {name: torch.stack(acts).mean().item() for name, acts in activation_data.items() if acts}
            per_token_activations.append(step_activations)
            
            # Update for next iteration
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # Append generated token
            generated_ids.append(next_token_id.item())
            
            # Update UI in real-time
            current_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            temp_history = history + [[user_message, current_response]]
            yield "", temp_history, None, gr.Slider(maximum=max(1, total_tokens_processed + len(generated_ids)), value=total_tokens_processed + len(generated_ids))

            if next_token_id.item() == tokenizer.eos_token_id:
                break
    
    # Finalize history for this turn
    final_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    history.append([user_message, final_response])
    print(f"\nGenerated response: {final_response}")

    # 3. Single Backward Pass for Gradients
    print("\n--- [Analysis] Start: Single Gradient Snapshot ---")
    full_sequence_ids = torch.cat([input_ids, torch.tensor([generated_ids], device=model.device)], dim=-1)
    model.train()
    model.zero_grad()
    
    outputs = model(input_ids=full_sequence_ids, attention_mask=torch.ones_like(full_sequence_ids), labels=full_sequence_ids)
    loss = outputs.loss
    if loss is not None and loss.requires_grad:
        loss.backward()
        print("Backward pass completed.")

    final_gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            final_gradients[name] = param.grad.norm().item()
    
    model.zero_grad()
    print(f"Collected {len(final_gradients)} final gradients.")

    # 4. Data Correlation and Storage
    print("\n--- [Analysis] Start: Data Correlation & Storage ---")
    cursor = db_conn.cursor()
    num_generated_tokens = len(per_token_activations)

    for i in range(num_generated_tokens):
        step_activations = per_token_activations[i]
        
        for param_name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            module_name = '.'.join(param_name.split('.')[:-1])
            activation = step_activations.get(module_name, 0.0)
            grad_norm = final_gradients.get(param_name, 0.0)
            pi_diff = activation - grad_norm
            
            q_activation = quantize_to_int8(activation)
            q_grad_norm = quantize_to_int8(grad_norm)
            q_pi_diff = quantize_to_int8(pi_diff + 100)

            cursor.execute("""
            INSERT OR REPLACE INTO param_metrics (token_idx, param_name, activation, grad_norm, pi_diff)
            VALUES (?, ?, ?, ?, ?)
            """, (total_tokens_processed + i, param_name, q_activation, q_grad_norm, q_pi_diff))

    db_conn.commit()
    total_tokens_processed += num_generated_tokens
    print(f"Stored metrics for {num_generated_tokens} tokens. Total processed: {total_tokens_processed}")
    print("--- [Analysis] End ---")

    # Final plot update after analysis is complete
    final_plot = update_plot(total_tokens_processed - 1, view_mode)
    yield "", history, final_plot, gr.Slider(maximum=max(1, total_tokens_processed - 1), value=total_tokens_processed - 1)

def update_plot_from_history(token_idx, view_mode):
    return update_plot(token_idx, view_mode)

def update_plot(token_idx, view_mode):
    global db_conn, total_tokens_processed
    
    if total_tokens_processed == 0:
        return plt.figure()

    token_idx = max(0, min(token_idx, total_tokens_processed - 1))
    
    cursor = db_conn.cursor()
    cursor.execute("SELECT param_name, activation, grad_norm, pi_diff FROM param_metrics WHERE token_idx=?", (token_idx,))
    data = cursor.fetchall()

    if not data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data for current token", ha='center', va='center')
        return fig

    param_names, q_activations, q_grad_norms, q_pi_diffs = zip(*data, strict=False)

    # --- Dequantize data for visualization ---
    activations = [dequantize_from_int8(v) for v in q_activations]
    grad_norms = [dequantize_from_int8(v) for v in q_grad_norms]
    pi_diffs = [dequantize_from_int8(v) - 100 for v in q_pi_diffs]

    # --- Sample data for visualization ---
    sample_size = 500000
    if len(param_names) > sample_size:
        sampled_indices = random.sample(range(len(param_names)), sample_size)
        param_names = [param_names[i] for i in sampled_indices]
        activations = [activations[i] for i in sampled_indices]
        grad_norms = [grad_norms[i] for i in sampled_indices]
        pi_diffs = [pi_diffs[i] for i in sampled_indices]

    fig, ax = plt.subplots(figsize=(10, 10))

    if view_mode == "PI Diff Scatter":
        scatter = ax.scatter(activations, grad_norms, c=pi_diffs, cmap='coolwarm', s=5, alpha=0.6)
        ax.set_title(f"Token {token_idx}: PI Diff Scatter (Sampled: {len(param_names)})")
        ax.set_xlabel("Activation")
        ax.set_ylabel("Gradient Norm")
        plt.colorbar(scatter, ax=ax, label='PI Diff (Activation - Gradient Norm)')
    else:
        # --- K-Means Clustering for Heatmap Layout ---
        features = StandardScaler().fit_transform(np.array([activations, grad_norms, pi_diffs]).T)
        num_clusters = min(int(len(features)**0.5 / 2), 100) # Heuristic for cluster count
        if num_clusters < 2:
             ax.text(0.5, 0.5, "Not enough features to cluster", ha='center', va='center')
             return fig

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(features)

        # --- Prepare data for Heatmap ---
        if view_mode == "Activation":
            plot_values = activations
            title = f"Token {token_idx}: Activation Heatmap (K-Means Layout, Sampled: {len(param_names)})"
            cmap = 'viridis'
        else: # "Gradient Norm"
            plot_values = grad_norms
            title = f"Token {token_idx}: Gradient Norm Heatmap (K-Means Layout, Sampled: {len(param_names)})"
            cmap = 'plasma'

        # --- Create Heatmap Grid ---
        # Determine grid size
        grid_dim = 300
        heatmap = np.zeros((grid_dim, grid_dim))
        
        # Assign points to grid based on cluster and position within cluster
        points_in_cluster = {i: [] for i in range(num_clusters)}
        for i, cluster_id in enumerate(clusters):
            points_in_cluster[cluster_id].append(i)

        current_x, current_y = 0, 0
        for cluster_id in range(num_clusters):
            points = points_in_cluster[cluster_id]
            for point_idx in points:
                if current_y >= grid_dim:
                    current_y = 0
                    current_x += 1
                if current_x < grid_dim:
                    heatmap[current_x, current_y] = plot_values[point_idx]
                    current_y += 1
        
        im = ax.imshow(heatmap, cmap=cmap, interpolation='nearest')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label="Value")

    plt.tight_layout()
    plt.close(fig) # Explicitly close the figure to free memory
    return fig

if __name__ == "__main__":
    run_knowledge_extraction()
