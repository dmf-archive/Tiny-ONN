import torch
import torch.nn.functional as F
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple
from tqdm import tqdm

from .config import TinyOnnArcConfig
from .model import TinyOnnForArcReconstruction, Block as OriginalBlock
from .data import pad_grid

newline_token = 10
pad_token = 0

class EvalBlock(OriginalBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, dict[Any, Any], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        normed_hs_smha = self.ln1(hidden_states)
        smha_routing_weights, smha_gate_cache = self.smha_layer.forward_gating(normed_hs_smha)
        B, T, C = hidden_states.shape
        smha_routing_weights_flat = smha_routing_weights.view(B * T, -1)

        attn_output, smha_cache, present_key_value = self.smha_layer.forward_main(
            normed_hs_smha, smha_routing_weights_flat, past_key_value
        )
        smha_cache["gate_cache"] = smha_gate_cache
        smha_cache["normed_hs"] = normed_hs_smha
        smha_cache["layer"] = self.smha_layer
        hidden_states = residual + attn_output

        residual = hidden_states
        normed_hs_moe = self.ln2(hidden_states)
        moe_routing_weights, moe_gate_cache = self.moe_layer.forward_gating(normed_hs_moe)
        moe_routing_weights_flat = moe_routing_weights.view(B * T, -1)

        moe_output, moe_cache = self.moe_layer.forward_main(normed_hs_moe, moe_routing_weights_flat, moe_gate_cache)
        moe_cache["normed_hs"] = normed_hs_moe
        hidden_states = residual + moe_output

        block_cache = {("smha", self.layer_index, 0): smha_cache, ("moe", self.layer_index, 0): moe_cache}
        return hidden_states, block_cache, present_key_value if use_cache else None

def preprocess_input_grid(grid: List[List[int]], max_h: int = 30, max_w: int = 30) -> torch.Tensor:
    input_tensor = pad_grid(grid, max_h, max_w)
    input_rows = [torch.cat((row, torch.tensor([newline_token], dtype=torch.long))) for row in input_tensor]
    input_seq = torch.cat(input_rows)
    return input_seq

def to_grid_and_crop(seq: torch.Tensor) -> List[List[int]]:
    pixel_seq = seq[seq != newline_token]
    pixel_seq = pixel_seq[pixel_seq != pad_token]

    if pixel_seq.numel() == 0:
        return [[0]]

    max_dim = 30
    padded_pixel_seq = F.pad(pixel_seq, (0, max_dim * max_dim - pixel_seq.numel()), "constant", pad_token)
    temp_grid = padded_pixel_seq.view(max_dim, max_dim)

    rows = torch.any(temp_grid != pad_token, dim=1)
    cols = torch.any(temp_grid != pad_token, dim=0)

    if not torch.any(rows) or not torch.any(cols):
        return [[0]]

    min_r, max_r = torch.where(rows)[0].min(), torch.where(rows)[0].max()
    min_c, max_c = torch.where(cols)[0].min(), torch.where(cols)[0].max()

    cropped_grid = temp_grid[min_r : max_r + 1, min_c : max_c + 1]

    return cropped_grid.tolist()

def to_grid_for_attempt1(seq: torch.Tensor, h: int = 30, w: int = 30) -> List[List[int]]:
    pixel_seq = seq[seq != newline_token]
    pixel_seq = pixel_seq[pixel_seq != -100]
    
    num_pixels = pixel_seq.numel()
    if num_pixels > h * w:
        pixel_seq = pixel_seq[:h*w]
    elif num_pixels < h * w:
        pixel_seq = F.pad(pixel_seq, (0, h * w - num_pixels), "constant", 0)
    
    return pixel_seq.view(h, w).tolist()

def predict_arc_output(model: TinyOnnForArcReconstruction, input_grid: List[List[int]], config: TinyOnnArcConfig, max_len: int = 1861) -> Tuple[List[List[int]], List[List[int]], torch.Tensor]:
    model.eval()
    with torch.no_grad():
        input_seq = preprocess_input_grid(input_grid).to(next(model.parameters()).device)
        
        generated_sequence = input_seq.clone()
        past_key_values = None
        
        output_start_idx = len(input_seq)

        logits, _, past_key_values = model(input_ids=generated_sequence.unsqueeze(0), use_cache=True)
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()
        
        generated_sequence = torch.cat((generated_sequence, torch.tensor([next_token], device=generated_sequence.device)))

        for _ in range(max_len - len(input_seq) - 1):
            if generated_sequence.numel() >= config.max_position_embeddings:
                break
            
            logits, _, past_key_values = model(
                input_ids=generated_sequence[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            
            generated_sequence = torch.cat((generated_sequence, torch.tensor([next_token], device=generated_sequence.device)))
            
            if next_token == newline_token and generated_sequence.numel() > output_start_idx + 1:
                pass
        
        predicted_output_seq = generated_sequence[output_start_idx:]
        
        attempt1_grid = to_grid_for_attempt1(predicted_output_seq)
        attempt2_grid = to_grid_and_crop(predicted_output_seq)
        
        return attempt1_grid, attempt2_grid, predicted_output_seq

def main():
    config = TinyOnnArcConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyOnnForArcReconstruction(config)
    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = EvalBlock(config, i)
    model = model.to(device)
    
    checkpoint_path = Path("exp/ARC-Killer.pt")
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        sys.exit(1)

    model.eval()

    eval_challenges_path = Path("data/arc-agi_evaluation_challenges.json")
    eval_solutions_path = Path("data/arc-agi_evaluation_solutions.json")

    if not eval_challenges_path.exists() or not eval_solutions_path.exists():
        print("Error: Evaluation challenges or solutions file not found.")
        sys.exit(1)

    with open(eval_challenges_path, "r") as f:
        evaluation_tasks = json.load(f)
    
    with open(eval_solutions_path, "r") as f:
        solutions = json.load(f)

    total_tasks = len(evaluation_tasks)
    correct_predictions = 0
    total_test_pairs = 0

    print(f"Starting inference on {total_tasks} tasks...")

    with tqdm(total=total_tasks, desc="Processing tasks") as pbar:
        total_token_correct = 0
        total_tokens = 0
        
        for task_id, task_data in evaluation_tasks.items():
            for test_pair_idx, test_pair in enumerate(task_data["test"]):
                input_grid = test_pair["input"]
                
                ground_truth_output = None
                if isinstance(solutions[task_id], list) and test_pair_idx < len(solutions[task_id]):
                    if isinstance(solutions[task_id][test_pair_idx], dict) and 'output' in solutions[task_id][test_pair_idx]:
                        ground_truth_output = solutions[task_id][test_pair_idx]["output"]
                    elif isinstance(solutions[task_id][test_pair_idx], list):
                        ground_truth_output = solutions[task_id][test_pair_idx]
                elif isinstance(solutions[task_id], dict) and 'output' in solutions[task_id]:
                    ground_truth_output = solutions[task_id]["output"]
                elif isinstance(solutions[task_id], list) and len(solutions[task_id]) == 1 and 'output' in solutions[task_id][0]:
                    ground_truth_output = solutions[task_id][0]['output']
                elif isinstance(solutions[task_id], list) and len(solutions[task_id]) == 1 and isinstance(solutions[task_id][0], list):
                    ground_truth_output = solutions[task_id][0]
                else:
                    print(f"Warning: Could not parse ground truth for task {task_id}, test pair {test_pair_idx}. Solutions structure: {solutions[task_id]}")
                    continue

                if ground_truth_output is None:
                    print(f"Warning: Ground truth is None for task {task_id}, test pair {test_pair_idx}. Skipping.")
                    continue

                attempt1_output, attempt2_output, predicted_output_seq = predict_arc_output(model, input_grid, config)
                
                # Calculate token accuracy
                ground_truth_tensor = preprocess_input_grid(ground_truth_output).to(predicted_output_seq.device)
                
                min_len = min(predicted_output_seq.numel(), ground_truth_tensor.numel())
                total_token_correct += (predicted_output_seq[:min_len] == ground_truth_tensor[:min_len]).sum().item()
                total_tokens += min_len

                if attempt1_output == ground_truth_output or attempt2_output == ground_truth_output:
                    correct_predictions += 1
                
                total_test_pairs += 1
            
            pbar.update(1)
            current_grid_acc = correct_predictions / total_test_pairs if total_test_pairs > 0 else 0
            current_token_acc = total_token_correct / total_tokens if total_tokens > 0 else 0
            pbar.set_postfix(grid_acc=f"{current_grid_acc:.4f}", token_acc=f"{current_token_acc:.4f}")

    grid_acc = correct_predictions / total_test_pairs if total_test_pairs > 0 else 0
    print(f"\n--- Evaluation Finished ---")
    print(f"Total Test Pairs: {total_test_pairs}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Grid Accuracy: {grid_acc:.4f}")

if __name__ == "__main__":
    main()