import tomllib
from pathlib import Path

from .base import ExperimentConfig


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return ExperimentConfig(**data)
