import torch
import argparse
from pathlib import Path

def convert_checkpoint_to_bf16(input_path: Path, output_path: Path):
    """
    Loads a checkpoint, converts all model parameters to bfloat16,
    and saves only the model state dict to a new file.
    """
    print(f"Loading checkpoint from: {input_path}")
    
    # Load the checkpoint onto CPU to avoid using GPU memory
    checkpoint = torch.load(input_path, map_location='cpu')
    
    if 'model_state_dict' not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")
        
    model_state_dict = checkpoint['model_state_dict']
    converted_state_dict = {}
    
    print("Converting model parameters to bfloat16...")
    for key, value in model_state_dict.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            converted_state_dict[key] = value.to(torch.bfloat16)
        else:
            converted_state_dict[key] = value # Keep non-float tensors as they are (e.g., int, bool)
            
    print(f"Saving bfloat16 model state dict to: {output_path}")
    torch.save(converted_state_dict, output_path)
    
    original_size = input_path.stat().st_size / (1024 * 1024)
    converted_size = output_path.stat().st_size / (1024 * 1024)
    
    print("\n--- Conversion Summary ---")
    print(f"Original checkpoint size: {original_size:.2f} MB")
    print(f"Converted state dict size: {converted_size:.2f} MB")
    print(f"Size reduction: {original_size - converted_size:.2f} MB")
    print("Conversion complete. The new file contains only the bfloat16 model weights.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch checkpoint to a smaller bfloat16 model state dict."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input checkpoint file (.pt)."
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path for the output bfloat16 state dict file (.pt)."
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.output)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    convert_checkpoint_to_bf16(input_file, output_file)