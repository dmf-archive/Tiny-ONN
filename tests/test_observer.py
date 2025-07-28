import shutil
from pathlib import Path

import numpy as np
import torch

from tiny_onn.config import TinyOnnConfig
from training.observer import UnifiedObserver


def test_unified_observer():
    output_dir = Path("tests/test_output/observer_test")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observer = UnifiedObserver(output_dir, device)
    rng = np.random.default_rng()

    model_config = TinyOnnConfig(num_hidden_layers=2, num_experts_per_layer=4)

    num_steps = 15
    for step in range(num_steps):
        avg_k = rng.random() * 2 + 3
        metrics = {
            "type": "train",
            "step": step,
            "main_loss": rng.random() * 3 + 1,
            "smk_loss": rng.random() * 3 + 1,
            "avg_k_loss": rng.random() * 3 + 1,
            "main_acc": rng.random() * 0.2 + 0.7,
            "gating_acc": rng.random() * 0.3 + 0.1,
            "surprise": rng.random() * 1e-4,
            "tau": rng.random() * 2,
            "pi_score": rng.random(),
            "avg_k": avg_k,
            "global_avg_k": avg_k
            * model_config.num_hidden_layers
            * (0.95 + rng.random() * 0.1),
        }
        observer.log_metrics(metrics, step)

        num_tokens = 100
        optimal_indices = torch.randint(
            0, model_config.num_experts_per_layer, (num_tokens,), device=device
        )

        all_router_logits_list = [
            torch.randn(num_tokens, model_config.num_experts_per_layer, device=device)
            for _ in range(model_config.num_hidden_layers)
        ]

        layer_token_offsets = {
            i: i * num_tokens for i in range(model_config.num_hidden_layers)
        }

        observer.update_expert_data(
            optimal_indices,
            all_router_logits_list,
            layer_token_offsets,
            model_config,
            step,
        )

        if step == num_steps - 1:
            observer.plot_dashboards(step, model_config)

    observer.close()

    assert (output_dir / "img" / "core_metrics_latest.png").exists()
    assert (output_dir / "img" / "expert_dashboard_latest.png").exists()
    assert (output_dir / "logs").exists()
