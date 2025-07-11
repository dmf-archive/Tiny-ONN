import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from threading import Thread
from typing import Any

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scanner.engine import run_fmri_scan
from scanner.utils.plotting import QUANTIZATION_SCALE, update_plot
from scanner.utils.ui import create_main_ui


def log_message(message: str, level: str = "INFO"):
    print(f"{level}: {message}")


def initialize_app(model_path: str, db_path: str | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "model": None,
        "tokenizer": None,
        "db_conn": None,
        "total_tokens_processed": 0,
        "param_name_to_id_map": {},
        "id_to_param_name_map": {},
        "is_replay_mode": db_path is not None,
    }

    if state["is_replay_mode"]:
        log_message(f"--- [REPLAY MODE] Loading database from: {db_path} ---")
        if not db_path or not os.path.exists(db_path):
            log_message(f"Error: Database file not found at {db_path}", level="ERROR")
            sys.exit(1)
        state["db_conn"] = sqlite3.connect(db_path, check_same_thread=False)
        cursor = state["db_conn"].cursor()
        cursor.execute("SELECT MAX(token_idx) FROM block_metrics")
        result = cursor.fetchone()
        state["total_tokens_processed"] = (
            result[0] + 1 if result and result[0] is not None else 0
        )
        log_message(f"Database loaded. Total tokens: {state['total_tokens_processed']}")
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
        state["model"] = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_path,
        )
        log_message("Model loaded successfully.")

        model_id = model_path.replace("/", "--").replace("\\", "--")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_filename = f"{model_id}_{timestamp}.db"
        db_dir = "data/db"
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, db_filename)
        state["db_conn"] = sqlite3.connect(db_path, check_same_thread=False)
        cursor = state["db_conn"].cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS block_metrics (
                token_idx INTEGER, param_name INTEGER, block_idx INTEGER,
                activation INTEGER, grad_norm INTEGER, absmax INTEGER,
                PRIMARY KEY (token_idx, param_name, block_idx)
            )
            """
        )
        state["db_conn"].commit()
        log_message(f"Database setup at: {db_path}")

    map_path = os.path.join("common/data/metadata", "param_name_map.json")
    if os.path.exists(map_path):
        with open(map_path, encoding="utf-8") as f:
            state["id_to_param_name_map"] = json.load(f)
        state["param_name_to_id_map"] = {
            name: int(id_str) for id_str, name in state["id_to_param_name_map"].items()
        }
        log_message("Parameter name map loaded.")

    return state


def store_per_token_data(per_token_data: list[dict], start_idx: int, state: dict):
    if not state["param_name_to_id_map"]:
        log_message("Param map not available.", level="ERROR")
        return

    cursor = state["db_conn"].cursor()
    for i, token_data in enumerate(per_token_data):
        current_token_idx = start_idx + i
        for param_name, metrics in token_data.items():
            param_id = state["param_name_to_id_map"].get(param_name)
            if param_id is None:
                continue

            activations = metrics.get("activation", [])
            gradients = metrics.get("gradient", [0.0] * len(activations))
            weights = metrics.get("weight", [0.0] * len(activations))
            for block_idx, (act, grad, w) in enumerate(
                zip(activations, gradients, weights, strict=False)
            ):
                quantized_act = int(act * QUANTIZATION_SCALE)
                quantized_grad = int(grad * QUANTIZATION_SCALE)
                quantized_w = int(w * QUANTIZATION_SCALE)
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO block_metrics
                    (token_idx, param_name, block_idx, activation, grad_norm, absmax)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        current_token_idx,
                        param_id,
                        block_idx,
                        quantized_act,
                        quantized_grad,
                        quantized_w,
                    ),
                )
    state["db_conn"].commit()


def main(args):
    state = initialize_app(args.model, args.db_path)

    def process_input(
        user_message,
        history: list,
        view_mode: str,
        vmin: float,
        vmax: float,
        use_fmri: bool,
    ):
        history = history or []
        history.append({"role": "user", "content": user_message})

        if use_fmri:
            final_response, per_token_data, _, _ = run_fmri_scan(
                model=state["model"],
                tokenizer=state["tokenizer"],
                user_message=user_message,
                history=history,
                temperature=0.7,
                top_p=0.95,
            )
            history.append({"role": "assistant", "content": final_response})

            num_tokens = len(per_token_data)
            if num_tokens > 0:
                store_per_token_data(
                    per_token_data, state["total_tokens_processed"], state
                )
                state["total_tokens_processed"] += num_tokens

            last_token_idx = state["total_tokens_processed"] - 1
            final_plot = update_plot(
                last_token_idx,
                view_mode,
                state["db_conn"],
                state["total_tokens_processed"],
                vmin=vmin,
                vmax=vmax,
            )
            slider_update = gr.update(
                maximum=max(0, last_token_idx), value=last_token_idx
            )
            return "", history, final_plot, slider_update
        else:
            streamer = TextIteratorStreamer(
                state["tokenizer"], skip_prompt=True, skip_special_tokens=True
            )
            input_ids = state["tokenizer"].apply_chat_template(
                history, return_tensors="pt"
            ).to(state["model"].device)
            generation_kwargs = dict(
                input_ids=input_ids,
                streamer=streamer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )
            thread = Thread(target=state["model"].generate, kwargs=generation_kwargs)
            thread.start()

            history.append({"role": "assistant", "content": ""})
            yield "", history, None, gr.update()

            full_response = ""
            for new_text in streamer:
                full_response += new_text
                history[-1]["content"] = full_response
                yield "", history, None, gr.update()

    def update_plot_wrapper(token_idx, view_mode, vmin, vmax):
        if token_idx is None:
            return None
        return update_plot(
            int(token_idx),
            view_mode,
            state["db_conn"],
            state["total_tokens_processed"],
            state["id_to_param_name_map"],
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
        "--db_path",
        type=str,
        default=None,
        help="Path to a database file for replay mode.",
    )
    args = parser.parse_args()
    main(args)
