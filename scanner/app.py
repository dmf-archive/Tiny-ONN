import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scanner.engine import run_fmri_scan
from scanner.io import (
    METRICS_DTYPE,
    append_records_to_mscan,
    create_mscan_file,
    load_mscan_file,
)
from scanner.utils.plotting import update_plot
from scanner.utils.ui import create_main_ui


def log_message(message: str, level: str = "INFO"):
    print(f"{level}: {message}")


def initialize_app(model_path: str, mscan_path: str | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "model": None,
        "tokenizer": None,
        "mscan_filepath": None,
        "metadata": {},
        "data": None,
        "total_tokens_processed": 0,
        "param_name_to_id_map": {},
        "id_to_param_name_map": {},
        "is_replay_mode": mscan_path is not None,
    }

    if state["is_replay_mode"]:
        log_message(f"--- [REPLAY MODE] Loading mscan from: {mscan_path} ---")
        if mscan_path is None:
            log_message("Error: mscan_path cannot be None in replay mode.", level="ERROR")
            sys.exit(1)
        state["mscan_filepath"] = Path(mscan_path)
        try:
            state["metadata"], state["data"] = load_mscan_file(state["mscan_filepath"])
            state["total_tokens_processed"] = sum(
                seq["num_tokens"] for seq in state["metadata"].get("sequences", [])
            )
            state["id_to_param_name_map"] = state["metadata"].get(
                "id_to_param_name_map", {}
            )
            log_message(
                f"MSCAN file loaded. Total tokens: {state['total_tokens_processed']}"
            )
        except (FileNotFoundError, ValueError) as e:
            log_message(f"Error loading mscan file: {e}", level="ERROR")
            sys.exit(1)
    else:
        log_message("--- [LIVE MODE] ---")
        script_dir = os.path.dirname(__file__)
        cache_path = os.path.abspath(os.path.join(script_dir, "..", "weights"))
        os.makedirs(cache_path, exist_ok=True)

        log_message(f"Loading tokenizer and model from: {model_path}")
        state["tokenizer"] = AutoTokenizer.from_pretrained(
            model_path, cache_dir=cache_path, trust_remote_code=True
        )
        if state["tokenizer"].pad_token is None:
            state["tokenizer"].pad_token = state["tokenizer"].eos_token
            state["tokenizer"].pad_token_id = state["tokenizer"].eos_token_id
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        state["model"] = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_path,
        )
        log_message("Model loaded successfully.")

        map_path = Path("data/metadata/param_name_map.json")
        if map_path.exists():
            with open(map_path, encoding="utf-8") as f:
                state["id_to_param_name_map"] = json.load(f)
            state["param_name_to_id_map"] = {
                name: int(id_str)
                for id_str, name in state["id_to_param_name_map"].items()
            }
            log_message("Parameter name map loaded.")

        model_id = model_path.replace("/", "--").replace("\\", "--")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mscan_filename = f"{model_id}_{timestamp}.mscan"
        mscan_dir = Path("data/scans")
        state["mscan_filepath"] = mscan_dir / mscan_filename

        state["metadata"] = {
            "format_version": "1.1",
            "model_name": model_path,
            "timestamp": timestamp,
            "id_to_param_name_map": state["id_to_param_name_map"],
            "sequences": [],
        }
        create_mscan_file(state["mscan_filepath"], state["metadata"])
        log_message(f"MSCAN file setup at: {state['mscan_filepath']}")

    return state


def store_per_token_data(per_token_data: list[dict], start_idx: int, state: dict, source_text: str = ""):
    if not state["param_name_to_id_map"]:
        log_message("Param map not available.", level="ERROR")
        return

    record_list = []
    for i, token_data in enumerate(per_token_data):
        current_token_idx = start_idx + i
        for param_name, metrics in token_data.items():
            param_id = state["param_name_to_id_map"].get(param_name)
            if param_id is None:
                continue

            activations = metrics.get("activation", [])
            num_blocks = len(activations)
            gradients = metrics.get("gradient", [0.0] * num_blocks)

            for block_idx, (act, grad) in enumerate(
                zip(activations, gradients, strict=False)
            ):
                record_list.append(
                    (
                        current_token_idx,
                        param_id,
                        block_idx,
                        int(act),
                        int(grad),
                    )
                )

    if not record_list:
        return

    new_records = np.array(record_list, dtype=METRICS_DTYPE)

    sequence_info = {
        "sequence_id": len(state["metadata"].get("sequences", [])),
        "num_tokens": len(per_token_data),
        "source": source_text,
    }

    append_records_to_mscan(state["mscan_filepath"], new_records, sequence_info)

    # Reload the state after update
    state["metadata"], state["data"] = load_mscan_file(state["mscan_filepath"])
    state["total_tokens_processed"] = sum(
        seq["num_tokens"] for seq in state["metadata"].get("sequences", [])
    )


def main(args):
    state = initialize_app(args.model, args.mscan_path)

    def process_input(
        user_message,
        history: list,
        view_mode: str,
        vmin: float,
        vmax: float,
        use_fmri: bool,
        no_think_mode: bool,
    ):
        current_history = history or []

        if no_think_mode and not any(d.get("role") == "system" for d in current_history):
            current_history.insert(0, {"role": "system", "content": "/no_think"})

        current_history.append({"role": "user", "content": user_message})

        full_response = ""
        per_token_data: list[dict] = []
        final_plot = None
        slider_update = gr.update()

        if use_fmri:
            full_response, per_token_data, _, _ = run_fmri_scan(
                model=state["model"],
                tokenizer=state["tokenizer"],
                messages=current_history,
                temperature=0.7,
                top_p=0.95,
                max_new_tokens=256,
            )
        else:
            inputs = state["tokenizer"].apply_chat_template(
                current_history,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(state["model"].device)

            with torch.no_grad():
                outputs = state["model"].generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    eos_token_id=state["tokenizer"].eos_token_id,
                )
            full_response = state["tokenizer"].decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # --- Unified Response Handling ---
        import re
        think_match = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)

        if think_match:
            think_content = think_match.group(1).strip()
            visible_response = full_response.replace(think_match.group(0), "").strip()
            # Gradio's Chatbot `metadata` is not a standard feature for creating accordions.
            # We will prepend the thought process in a formatted way.
            formatted_content = f"**Thought Process:**\n```\n{think_content}\n```\n\n**Response:**\n{visible_response}"
            current_history.append({"role": "assistant", "content": formatted_content})
        else:
            current_history.append({"role": "assistant", "content": full_response})

        if use_fmri and per_token_data:
            # The source text for this sequence is the user message and the assistant response
            source_text = f"User: {user_message}\nAssistant: {full_response}"
            store_per_token_data(
                per_token_data, state["total_tokens_processed"], state, source_text
            )
            # total_tokens_processed is now correctly updated inside store_per_token_data
            last_token_idx = state["total_tokens_processed"] - 1
            final_plot = update_plot(
                last_token_idx,
                view_mode,
                state["data"],
                state["total_tokens_processed"],
                state["id_to_param_name_map"],
                vmin=vmin,
                vmax=vmax,
            )
            slider_update = gr.update(
                maximum=max(0, last_token_idx), value=last_token_idx
            )

        return "", current_history, final_plot, slider_update

    def update_plot_wrapper(token_idx, view_mode, vmin, vmax, normalization_scope):
        if token_idx is None or state["data"] is None:
            return None
        return update_plot(
            int(token_idx),
            view_mode,
            state["data"],
            state["total_tokens_processed"],
            state["id_to_param_name_map"],
            normalization_scope=normalization_scope,
            vmin=vmin,
            vmax=vmax,
        )

    demo = create_main_ui(
        process_input_fn=process_input,
        update_plot_fn=update_plot_wrapper,
        get_total_tokens_fn=lambda: state["total_tokens_processed"],
        is_replay_mode=state["is_replay_mode"],
    )
    demo.launch(share=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tiny-ONN fMRI analysis.")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-1.7B", help="Model name for live mode."
    )
    parser.add_argument(
        "--mscan_path",
        type=str,
        default=None,
        help="Path to a .mscan file for replay mode.",
    )
    args = parser.parse_args()
    main(args)
