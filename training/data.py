from pathlib import Path

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

    is_local_file = Path(data_config.dataset_name).is_file()

    if is_local_file:
        raw_datasets = load_dataset("json", data_files=data_config.dataset_name)
        # Since it's a single file, we split it into train and validation
        split_datasets = raw_datasets["train"].train_test_split(
            test_size=f"{data_config.validation_split_percentage}%"
        )
        raw_datasets["train"] = split_datasets["train"]
        raw_datasets["validation"] = split_datasets["test"]
        remove_columns = ["messages"]
        text_column = "messages"
    else:
        raw_datasets = load_dataset(
            data_config.dataset_name,
            data_config.dataset_subset,
            streaming=False,
            trust_remote_code=True,
        )
        if "validation" not in raw_datasets:
            split_datasets = raw_datasets["train"].train_test_split(
                test_size=f"{data_config.validation_split_percentage}%"
            )
            raw_datasets["train"] = split_datasets["train"]
            raw_datasets["validation"] = split_datasets["test"]
        remove_columns = ["text", "timestamp", "url"]
        text_column = "text"

    def tokenize_function(examples):
        if text_column == "messages":
            tokenized_output = tokenizer(
                [
                    tokenizer.apply_chat_template(
                        conv, tokenize=False, add_generation_prompt=False
                    )
                    for conv in examples["messages"]
                ],
                truncation=True,
                max_length=data_config.max_seq_length,
                padding="max_length",
            )
            # The labels should be the input_ids, shifted.
            # We need to handle padding tokens so they are ignored in loss calculation.
            labels = torch.tensor(tokenized_output["input_ids"])
            labels[labels == tokenizer.pad_token_id] = -100
            tokenized_output["labels"] = labels.tolist()
            return tokenized_output
        else:
            # Original logic for text-based datasets
            return tokenizer(
                examples[text_column],
                truncation=True,
                max_length=data_config.max_seq_length,
                padding="max_length",
            )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_columns,
        num_proc=num_workers,
    )

    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

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
