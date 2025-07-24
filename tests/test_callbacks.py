import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from tiny_onn.modular import TinyOnnForCausalLM
from training.callbacks import MatplotlibVisualizer


def test_matplotlib_visualizer(tiny_test_model_and_tokenizer):
    img_dir = Path("tests/test_output/viz_test")
    if img_dir.exists():
        shutil.rmtree(img_dir)
    
    model, _ = tiny_test_model_and_tokenizer
    visualizer = MatplotlibVisualizer(img_dir=img_dir, model_name="test_model")

    metrics_cache = defaultdict(list)
    metrics_cache["main_loss"] = [1.0, 0.9, 0.8]
    metrics_cache["router_loss"] = [0.5, 0.4, 0.3]
    metrics_cache["main_acc"] = [0.6, 0.7, 0.8]
    metrics_cache["gating_acc"] = [0.5, 0.6, 0.7]
    metrics_cache["pi_score"] = [0.7, 0.75, 0.8]
    metrics_cache["surprise"] = [0.9, 0.8, 0.7]
    metrics_cache["tau"] = [1.2, 1.1, 1.0]

    expert_data_cache = {
        "selected_experts": np.array([[0, 1], [0, 5], [1, 2], [1, 8]]),
        "optimal_experts": np.array([[0, 2], [0, 5], [1, 3], [1, 8]]),
    }

    global_step = 100

    visualizer.update_plots(
        metrics_cache=metrics_cache,
        expert_data_cache=expert_data_cache,
        global_step=global_step,
        model=model
    )

    core_metrics_path = img_dir / "core_metrics_latest.png"
    expert_dashboard_path = img_dir / "expert_dashboard_latest.png"

    assert core_metrics_path.exists(), "Core metrics plot was not generated."
    assert expert_dashboard_path.exists(), "Expert dashboard plot was not generated."

    print("\n✅ PoC successful. MatplotlibVisualizer generated plots correctly.")
