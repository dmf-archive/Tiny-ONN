import torch


class MemoryLogger:
    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def log(self, checkpoint_name: str):
        if torch.cuda.is_available():
            print(f"\n--- Memory Snapshot at: {checkpoint_name} ---")
            print(torch.cuda.memory_summary())
            print(f"--- End Snapshot at: {checkpoint_name} ---\n")
