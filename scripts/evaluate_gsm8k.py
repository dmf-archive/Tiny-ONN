import argparse
import gc
import json
import os
import re
import sys
from datetime import datetime

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    PreTrainedTokenizer,
)

# 将项目根目录添加到 Python 路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.pruned_layers import PrunedQwen3DecoderLayer
from utils.logging_utils import log_message

# --- 从官方脚本借鉴的常量和函数 ---
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

model = None
tokenizer = None

def load_model_and_tokenizer_for_eval(model_path: str, use_4bit: bool = True):
    global model, tokenizer
    if model is not None and tokenizer is not None:
        return model, tokenizer

    log_message(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    log_message("Tokenizer loaded.")

    quantization_config = None
    if use_4bit:
        log_message("Using 4-bit quantization.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # 确定数据类型
    dtype = torch.bfloat16 if use_4bit else torch.float16

    log_message(f"Loading model from: {model_path} with dtype: {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="cuda:0",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    model.generation_config.do_sample = False
    log_message("Model and generation config loaded successfully.")
    
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    return model, tokenizer

def extract_answer_from_gold(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip().replace(",", "")
        try:
            return float(match_str)
        except ValueError:
            return INVALID_ANS
    return INVALID_ANS

def extract_answer_from_completion(completion):
    try:
        text = completion.replace(",", "")
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if numbers:
            return float(numbers[-1])
    except (ValueError, IndexError):
        pass
    return INVALID_ANS

def is_correct(completion, gold_answer):
    gold = extract_answer_from_gold(gold_answer)
    if gold == INVALID_ANS:
        return False
    pred = extract_answer_from_completion(completion)
    if pred == INVALID_ANS:
        return False
    return abs(gold - pred) < 1e-6

def evaluate_gsm8k(args):
    model, tokenizer = load_model_and_tokenizer_for_eval(args.model_path, use_4bit=args.use_4bit)
    
    dataset = load_dataset(args.dataset_name, "main")
    test_data = dataset["test"]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompts = [f"Question: {q}\nLet's think step by step\n" for q in test_data["question"]]
    answers = test_data["answer"]
    
    results = []
    correct_count = 0
    total_count = 0
    
    with tqdm(total=len(prompts), desc="Evaluating GSM8K Samples") as pbar:
        for i in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[i:i+args.batch_size]
            batch_answers = answers[i:i+args.batch_size]
            
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id
                )

            generated_ids = outputs[:, inputs.input_ids.shape[1]:]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for j in range(len(generated_texts)):
                response_text = generated_texts[j]
                true_answer_full = batch_answers[j]
                
                correct = is_correct(response_text, true_answer_full)
                if correct:
                    correct_count += 1
                total_count += 1

                results.append({
                    "index": i + j,
                    "question": batch_prompts[j],
                    "true_answer_full": true_answer_full,
                    "generated_text": response_text,
                    "is_correct": correct,
                })
            
            pbar.update(len(batch_prompts))
            # Update tqdm postfix with current stats
            accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
            pbar.set_postfix({
                "Correct": correct_count,
                "Total": total_count,
                "Accuracy": f"{accuracy:.2f}%"
            })

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    summary = {
        "model_path": args.model_path,
        "dataset_name": args.dataset_name,
        "total_samples": total_count,
        "correct_predictions": correct_count,
        "accuracy_percent": round(accuracy, 2),
    }
    log_message("Evaluation finished.")
    log_message(json.dumps(summary, indent=2))

    if args.output_path:
        output_dir = os.path.dirname(args.output_path)
        base_name = os.path.basename(args.output_path)
        model_name_slug = os.path.basename(args.model_path).replace('--', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"{os.path.splitext(base_name)[0]}_{model_name_slug}_{timestamp}.json"
        final_path = os.path.join(output_dir, final_filename)
        os.makedirs(output_dir, exist_ok=True)
        full_report = {"summary": summary, "results": results}
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=4, ensure_ascii=False)
        log_message(f"Full report saved to {final_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a pruned model on the GSM8K dataset using Few-shot CoT.")
    parser.add_argument("--model_path", type=str, default="weights/Qwen--Qwen3-1.7B-pruned", help="Path to the pruned model directory.")
    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="Name of the dataset to load from Hugging Face Hub.")
    parser.add_argument("--output_path", type=str, default="results/gsm8k_evaluation.json", help="Path to save the detailed evaluation results.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate for each answer.")
    parser.add_argument("--use_4bit", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable 4-bit quantization.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
    args = parser.parse_args()
    evaluate_gsm8k(args)

if __name__ == "__main__":
    main()
