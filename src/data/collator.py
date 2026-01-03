import torch
from torch.nn.utils.rnn import pad_sequence

from .serializer import VectorizedSerializer
from .tokenizer import ArcColorTokenizer


class ArcCollator:
    def __init__(self, tokenizer: ArcColorTokenizer, max_len: int) -> None:
        self.tokenizer = tokenizer
        self.serializer = VectorizedSerializer(tokenizer)
        self.max_len = max_len

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        all_input_ids: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for task_data in batch:
            input_ids, labels = self.serializer.serialize_task(task_data)

            if len(input_ids) > self.max_len:
                continue

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        if not all_input_ids:
            return {}

        padded_input_ids: torch.Tensor = pad_sequence(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels: torch.Tensor = pad_sequence(all_labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels
        }
