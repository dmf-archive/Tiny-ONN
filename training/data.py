from typing import Tuple

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .dataset import TextDataset


def get_dataloaders(
    tokenizer: AutoTokenizer,
    train_path: str,
    val_path: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TextDataset(train_path, tokenizer)
    val_dataset = TextDataset(val_path, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader
