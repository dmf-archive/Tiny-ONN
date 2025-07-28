import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skip(
    reason="Full pipeline test is resource-intensive and tested separately."
)
def test_full_pipeline_runs():
    config_path = "configs/dummy_test_config.yaml"
    output_dir = Path("tests/test_output/full_pipeline_test")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    command = [
        sys.executable,
        "train.py",
        "--config",
        config_path,
        "--max_steps",
        "3",
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    print("--- Subprocess stdout ---")
    print(result.stdout)
    print("--- Subprocess stderr ---")
    print(result.stderr)

    assert result.returncode == 0, "Training script exited with an error."

    log_dir = output_dir / "logs"
    img_dir = output_dir / "img"
    checkpoint_dir = output_dir / "checkpoints"

    assert log_dir.exists(), "Log directory was not created."
    assert len(list(log_dir.glob("*"))) > 0, "No log files were created."

    assert img_dir.exists(), "Image directory was not created."
    assert (img_dir / "core_metrics_latest.png").exists(), (
        "Core metrics plot was not created."
    )

    assert checkpoint_dir.exists(), "Checkpoint directory was not created."
    assert len(list(checkpoint_dir.glob("*.pt"))) > 0, (
        "No checkpoint files were created."
    )

    print("\n✅ PoC successful. Full training pipeline ran and produced artifacts.")
