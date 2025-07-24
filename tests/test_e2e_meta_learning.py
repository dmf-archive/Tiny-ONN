import sys
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))

from tiny_onn.modular import TinyOnnForCausalLM
from training.config import load_config
from training.engine import TrainerEngine


@pytest.fixture(scope="module")
def setup_e2e_test_environment():
    config_path = "configs/test_distillation.yaml"
    config = load_config(Path(config_path))

    model = TinyOnnForCausalLM.from_pretrained(
        config.model.model_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model_name,
        model_max_length=config.data.max_seq_length,
        cache_dir="weights",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dummy_input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    dummy_labels = torch.randint(0, model.config.vocab_size, (2, 8))
    dummy_dataset = TensorDataset(dummy_input_ids, dummy_labels)
    train_dataloader = DataLoader(dummy_dataset, batch_size=1)

    expert_params = [
        p
        for n, p in model.named_parameters()
        if "gate" not in n and p.requires_grad
    ]
    router_params = [
        p for n, p in model.named_parameters() if "gate" in n and p.requires_grad
    ]
    optimizer_experts = AdamW(expert_params, lr=1e-4)
    optimizer_router = AdamW(router_params, lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return {
        "config": config,
        "model": model,
        "tokenizer": tokenizer,
        "train_dataloader": train_dataloader,
        "optimizer_experts": optimizer_experts,
        "optimizer_router": optimizer_router,
        "device": device,
    }


def test_e2e_meta_learning_step(setup_e2e_test_environment):
    env = setup_e2e_test_environment
    engine = TrainerEngine(
        config=env["config"],
        model=env["model"],
        optimizer_experts=env["optimizer_experts"],
        optimizer_router=env["optimizer_router"],
        train_dataloader=env["train_dataloader"],
        eval_dataloader=env["train_dataloader"],
        device=env["device"],
    )

    batch = next(iter(env["train_dataloader"]))
    batch_dict = {"input_ids": batch[0], "labels": batch[1]}

    expert_param_before = (
        next(engine.model.model.layers[0].mlp.experts[0].parameters())
        .clone()
        .detach()
    )
    router_param_before = (
        next(engine.model.model.layers[0].mlp.gate.parameters()).clone().detach()
    )

    metrics = engine._hyper_step(batch_dict)

    assert "main_loss" in metrics
    assert metrics["main_loss"] > 0
    assert "surprise" in metrics
    assert metrics["surprise"] >= 0
    assert "router_loss" in metrics
    assert metrics["router_loss"] >= 0
    assert "gating_acc" in metrics
    assert 0 <= metrics["gating_acc"] <= 1
    assert "pi_score" in metrics
    assert 0 <= metrics["pi_score"] <= 1

    if metrics["surprise"] > 0 and "cuda" in env["device"].type:
        assert metrics["router_loss"] > 0

    expert_param_after = (
        next(engine.model.model.layers[0].mlp.experts[0].parameters())
        .clone()
        .detach()
    )
    router_param_after = (
        next(engine.model.model.layers[0].mlp.gate.parameters()).clone().detach()
    )

    assert not torch.equal(expert_param_before, expert_param_after)
    assert not torch.equal(router_param_before, router_param_after)
