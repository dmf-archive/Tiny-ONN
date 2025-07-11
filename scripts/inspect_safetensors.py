import argparse

from safetensors import safe_open


def inspect_safetensors_file(filepath: str):
    """
    Opens a .safetensors file and prints all the tensor keys it contains.
    """
    print(f"--- Inspecting: {filepath} ---")
    try:
        with safe_open(filepath, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"Found {len(keys)} tensors.")
            for i, key in enumerate(keys):
                print(f"  {i+1:04d}: {key}")
    except Exception as e:
        print(f"Error opening or reading file: {e}")
    print("--- Inspection Complete ---\n")

def main():
    parser = argparse.ArgumentParser(description="Inspect the keys within a .safetensors file.")
    parser.add_argument("filepath", type=str, help="Path to the .safetensors file to inspect.")
    args = parser.parse_args()
    inspect_safetensors_file(args.filepath)

if __name__ == "__main__":
    main()
