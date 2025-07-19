from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import DataConfig


def get_dataloaders(
    data_config: DataConfig,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_datasets = load_dataset(
        data_config.dataset_name,
        data_config.dataset_subset,
        streaming=False,
        trust_remote_code=True,
    )

    if "validation" not in raw_datasets:
        raw_datasets["validation"] = load_dataset(
            data_config.dataset_name,
            data_config.dataset_subset,
            split=f"train[:{data_config.validation_split_percentage}%]",
            trust_remote_code=True,
        )
        raw_datasets["train"] = load_dataset(
            data_config.dataset_name,
            data_config.dataset_subset,
            split=f"train[{data_config.validation_split_percentage}%:]",
            trust_remote_code=True,
        )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=data_config.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
        num_proc=num_workers,
    )

    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, eval_dataloader
