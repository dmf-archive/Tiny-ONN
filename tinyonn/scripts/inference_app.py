import os

import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tinyonn.configuration_tinyonn import TinyONNConfig
from tinyonn.model import TinyONNForCausalLM

# --- Register Custom Architecture with AutoClasses ---
# This allows `from_pretrained` to find our custom classes when it sees "tinyonn"
# in the config.json's `model_type` field.
AutoConfig.register("tinyonn", TinyONNConfig)
AutoModelForCausalLM.register(TinyONNConfig, TinyONNForCausalLM)

# --- Global Settings ---
# Per DeepWiki's recommendation, forcing high-precision reduction for bfloat16
# matrix multiplications to improve numerical stability.
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

# --- Global Variables ---
MODEL = None
TOKENIZER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_dir, tokenizer_name, cache_dir):
    """Loads the TinyONN model and tokenizer."""
    global MODEL, TOKENIZER

    print(f"Loading converted model from directory: {model_dir}")

    # --- Explicitly load config and verify model_type for debugging ---
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    print(f"--- !!! DEBUG: Loaded config.model_type = {config.model_type} !!! ---")

    # By trusting remote code, AutoModelForCausalLM will look at the config.json,
    # see the custom architecture, and use the locally defined TinyONNForCausalLM class.
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config, # Pass the loaded config object explicitly
        device_map=DEVICE,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 # Or the appropriate dtype
    )
    MODEL.eval()

    print("Loading tokenizer (offline)...")
    TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=True  # Force offline loading
    )
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    print("Model and tokenizer loaded successfully.")

def chat_interface(message, history):
    """The main chat function for the Gradio interface."""
    if MODEL is None or TOKENIZER is None:
        return "Error: Model not loaded. Please check the console."

    # Gradio history is a list of tuples. Convert it to a list of dicts for the template.
    history_for_template = []
    for user_msg, assistant_msg in (history or []):
        history_for_template.append({"role": "user", "content": user_msg})
        history_for_template.append({"role": "assistant", "content": assistant_msg})

    # Add the new user message
    history_for_template.append({"role": "user", "content": message})

    # Use the tokenizer to apply the chat template and tokenize
    tokenized_output = TOKENIZER.apply_chat_template(
        history_for_template,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # Ensure `inputs` is a dictionary, as `apply_chat_template` can sometimes
    # return a single tensor, which breaks the **inputs expansion.
    if isinstance(tokenized_output, torch.Tensor):
        inputs = {"input_ids": tokenized_output.to(DEVICE)}
    else:
        inputs = {k: v.to(DEVICE) for k, v in tokenized_output.items()}


    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=TOKENIZER.eos_token_id,
            pad_token_id=TOKENIZER.pad_token_id
        )

    # Decode the generated tokens, skipping the prompt
    prompt_len = inputs["input_ids"].shape[1]
    response = TOKENIZER.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    # The history for Gradio needs to be in (user, assistant) tuple format
    updated_history = history or []
    updated_history.append((message, response))

    # Return an empty string to clear the textbox, and the updated history
    return "", updated_history


def create_ui():
    """Creates the Gradio UI."""
    with gr.Blocks() as demo:
        gr.Markdown("# TinyONN Dense Equivalence Test")
        gr.Markdown("This interface tests the converted TinyONN model in its fully-activated state to verify it is functionally equivalent to the original dense Qwen3-1.7B model.")

        chatbot = gr.Chatbot(value=[], elem_id="chatbot", height=450)
        with gr.Row():
            with gr.Column(scale=4):
                textbox = gr.Textbox(
                    show_label=False,
                    placeholder="Enter your message and press enter",
                    container=False
                )
            with gr.Column(scale=1, min_width=0):
                submit_btn = gr.Button("Send")

        textbox.submit(chat_interface, [textbox, chatbot], [textbox, chatbot])
        submit_btn.click(chat_interface, [textbox, chatbot], [textbox, chatbot])

    return demo

if __name__ == "__main__":
    # --- Configuration ---
    TOKENIZER_PATH = "Qwen/Qwen3-1.7B"
    CACHE_DIR = "weights"
    CONVERTED_MODEL_DIR = os.path.join(CACHE_DIR, "tinyonn_converted_statedict")

    # 1. Always run the conversion script to ensure we are using the latest logic.
    # The script itself handles cleaning the output directory.
    print("Forcing re-conversion to ensure a fresh config and weights file...")
    from tinyonn.scripts.convert_qwen_to_tinyonn import convert_qwen_to_tinyonn
    convert_qwen_to_tinyonn(TOKENIZER_PATH, CONVERTED_MODEL_DIR, CACHE_DIR)
    print("Conversion script finished.")

    # 2. Add a verification step
    weights_path = os.path.join(CONVERTED_MODEL_DIR, "model.safetensors") # Corrected to look for the new format
    config_path = os.path.join(CONVERTED_MODEL_DIR, "config.json")
    print("Verifying existence of files before loading:")
    print(f"  - Config: {config_path} -> Exists: {os.path.exists(config_path)}")
    print(f"  - Weights: {weights_path} -> Exists: {os.path.exists(weights_path)}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"CRITICAL: Conversion script claimed to finish, but weights file not found at {weights_path}")

    # 3. Load the model
    load_model_and_tokenizer(CONVERTED_MODEL_DIR, TOKENIZER_PATH, CACHE_DIR)

    # 4. Launch the UI
    app = create_ui()
    app.launch(share=False)
