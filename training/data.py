from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from .config import DataConfig
from .dataset import JSONLDataset


def get_dataloaders(
    data_config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:

    if data_config.mode == "local_json":
        if not data_config.train_path or not data_config.eval_path:
            raise ValueError("train_path and eval_path must be specified for local_json mode")
        train_dataset = JSONLDataset(data_config.train_path, tokenizer, data_config.max_seq_length)
        val_dataset = JSONLDataset(data_config.eval_path, tokenizer, data_config.max_seq_length)
    elif data_config.mode == "transformers":
        if not data_config.dataset_name:
            raise ValueError("dataset_name must be specified for transformers mode")
        dataset = load_dataset(data_config.dataset_name, data_config.dataset_subset)
        split_dataset = dataset["train"].train_test_split(test_size=data_config.validation_split_percentage / 100)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
    else:
        raise ValueError(f"Unknown data mode: {data_config.mode}")

    prefetch_factor = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader
