from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp.pi_weighted_self_distillation_spike import (
    DistillMode,
    SmallTransformerConfig,
    SpikeConfig,
    build_task_dataset,
    run_spike_experiment,
)


def test_build_task_dataset_add_sub_labels_are_correct() -> None:
    p = 7
    add_ids, add_labels = build_task_dataset(p, "add")
    sub_ids, sub_labels = build_task_dataset(p, "sub")
    assert add_ids.shape == (p * p, 3)
    assert sub_ids.shape == (p * p, 3)
    add_ops = add_ids[:, 2]
    sub_ops = sub_ids[:, 2]
    assert torch.unique(add_ops).tolist() == [p]
    assert torch.unique(sub_ops).tolist() == [p + 1]
    assert torch.equal(add_labels, (add_ids[:, 0] + add_ids[:, 1]) % p)
    assert torch.equal(sub_labels, (sub_ids[:, 0] - sub_ids[:, 1]) % p)


def test_run_spike_experiment_returns_all_modes() -> None:
    cfg = SpikeConfig(
        p=13,
        train_frac=0.5,
        seed=123,
        batch_size=32,
        epochs_a=1,
        epochs_b=1,
        lr=1e-3,
        fixed_lambda=0.5,
        pi_alpha=1.0,
        pi_gamma=0.5,
        model=SmallTransformerConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_mlp=64,
            seq_len=3,
        ),
    )
    results = run_spike_experiment(cfg)
    expected_modes: set[DistillMode] = {
        "ce_only",
        "teacher_only",
        "fixed_kd",
        "confidence_kd",
        "entropy_kd",
        "pi_weighted_kd",
    }
    mode_set = {item["mode"] for item in results}
    assert mode_set == expected_modes
    assert len(results) == 6
    for item in results:
        assert 0.0 <= float(item["task_a_acc_after_b"]) <= 1.0
        assert 0.0 <= float(item["task_b_acc"]) <= 1.0
        assert -1.0 <= float(item["forgetting"]) <= 1.0
        assert 0.0 <= float(item["lambda_mean"]) <= 1.0
