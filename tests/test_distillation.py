import subprocess
import sys
from pathlib import Path

import pytest

from scripts.perform_surgery import perform_surgery

# Define paths
ROOT_DIR = Path(__file__).parent.parent
CONFIG_PATH = ROOT_DIR / "configs/test_distillation.yaml"
STUDENT_MODEL_PATH = ROOT_DIR / "weights/Tiny-ONN-0.6B-Hyper-SMoE"
BASE_MODEL_NAME = "Qwen/Qwen3-0.6B"
CACHE_DIR = ROOT_DIR / "weights"


@pytest.fixture(scope="module", autouse=True)
def setup_student_model():
    """
    Ensure the surgically modified student model exists before running tests.
    If not, run the surgery script.
    """
    if not STUDENT_MODEL_PATH.exists():
        print(f"Student model not found at {STUDENT_MODEL_PATH}, performing surgery...")
        model, tokenizer = perform_surgery(BASE_MODEL_NAME, cache_dir=str(CACHE_DIR))
        model.save_pretrained(STUDENT_MODEL_PATH)
        tokenizer.save_pretrained(STUDENT_MODEL_PATH)
        print(
            f"Surgically modified model and tokenizer saved to {STUDENT_MODEL_PATH}"
        )
    else:
        print(f"Found existing student model at {STUDENT_MODEL_PATH}.")


@pytest.mark.skip(reason="Distillation test is resource-intensive and currently disabled.")
def test_distillation_training_run():
    """
    Test the full distillation training pipeline by running the main training script
    with the distillation config.
    """
    # We need to limit the training to just a few steps for a quick test.
    train_script_path = ROOT_DIR / "train.py"

    command = [
        sys.executable,
        str(train_script_path),
        "--config",
        str(CONFIG_PATH),
    ]

    try:
        # We increase the timeout because loading models (especially quantized ones) can be slow.
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes timeout
        )
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        pytest.fail(f"Training script failed with exit code {e.returncode}")
    except subprocess.TimeoutExpired as e:
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        pytest.fail("Training script timed out.")
