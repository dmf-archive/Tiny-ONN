import glob
import os
from pathlib import Path

import torch
from torch.optim import Optimizer
from transformers import PreTrainedModel


def save_rolling_checkpoint(
    model: PreTrainedModel,
    optimizer_experts: Optimizer,
    optimizer_router: Optimizer,
    checkpoint_dir: str | Path,
    step: int,
    rolling_checkpoint_count: int,
):
    checkpoint_path_dir = Path(checkpoint_dir)
    os.makedirs(checkpoint_path_dir, exist_ok=True)

    checkpoint_path = checkpoint_path_dir / f"checkpoint-step-{step}"
    model.save_pretrained(str(checkpoint_path))

    torch.save(optimizer_experts.state_dict(), checkpoint_path / "optimizer_experts.pt")
    torch.save(optimizer_router.state_dict(), checkpoint_path / "optimizer_router.pt")

    checkpoints = sorted(
        glob.glob(str(checkpoint_path_dir / "checkpoint-step-*")), key=os.path.getmtime
    )

    if len(checkpoints) > rolling_checkpoint_count:
        for old_checkpoint in checkpoints[:-rolling_checkpoint_count]:
            for f in glob.glob(f"{old_checkpoint}/*"):
                os.remove(f)
            os.rmdir(old_checkpoint)
