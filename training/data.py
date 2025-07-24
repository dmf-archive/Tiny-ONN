
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from .dataset import JSONLDataset


def get_dataloaders(
    tokenizer: PreTrainedTokenizer,
    train_path: str,
    val_path: str,
    batch_size: int,
    num_workers: int,
    max_length: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = JSONLDataset(train_path, tokenizer, max_length)
    val_dataset = JSONLDataset(val_path, tokenizer, max_length)

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
