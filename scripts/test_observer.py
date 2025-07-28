import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from training.observer import UnifiedObserver


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("output/observer_test")
    observer = UnifiedObserver(output_dir, device, max_history_steps=50)
    rng = np.random.default_rng()

    # Mock model config
    model_config = SimpleNamespace(num_hidden_layers=4, num_experts_per_layer=8)
    total_experts = model_config.num_hidden_layers * model_config.num_experts_per_layer

    print("Generating mock data and plotting...")
    for step in range(100):
        # Mock metrics
        metrics = {
            "main_loss": rng.random() + 0.5,
            "smk_loss": rng.random() * 0.5,
            "main_acc": 1 - (rng.random() + 0.1),
            "gating_acc": 1 - (rng.random() + 0.2),
            "pi_score": rng.random(),
            "surprise": rng.random() * 10,
            "tau": rng.random() + 1.0,
            "gating_kld": rng.random() * 0.1,
            "avg_k": rng.uniform(2, 4),
            "global_avg_k": rng.uniform(8, 16),
        }

        # Mock expert data
        num_tokens = rng.integers(50, 100)
        selected_experts = torch.randint(0, total_experts, (num_tokens,), device=device)
        optimal_experts = torch.randint(0, total_experts, (num_tokens,), device=device)

        expert_data = {
            "global_step": torch.tensor(step, device=device),
            "selected_experts": selected_experts,
            "optimal_experts": optimal_experts,
        }

        observer.log_metrics_and_expert_data(metrics, expert_data, step)

        if step % 25 == 0 and step > 0:
            print(f"Plotting dashboard at step {step}...")
            observer.plot_dashboards(step, model_config)

    observer.close()
    print(f"Test complete. Check plots in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
