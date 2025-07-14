import os
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from scanner.core.engine import ScannerEngine
from scanner.mscan import MScanWriter
from scanner.utils.normalization import process_and_quantize_data


class Scheduler:
    def __init__(self, model_name: str, device: str = "cuda"):
        os.environ["TRANSFORMERS_CACHE"] = str(Path("./weights").resolve())
        self._model_name = model_name
        self._device = device
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None
        self._param_name_to_id_map: dict[str, int] = {}
        self._id_to_param_name_map: dict[int, str] = {}

    def _load_model_and_tokenizer(self):
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=torch.bfloat16,
                device_map=self._device,
                trust_remote_code=True,
            )
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=True
            )

        if not self._param_name_to_id_map:
            self._param_name_to_id_map = {name: i for i, (name, _) in enumerate(self._model.named_parameters())}
            self._id_to_param_name_map = {i: name for name, i in self._param_name_to_id_map.items()}

    def _prepare_inputs(self, text: str) -> dict[str, torch.Tensor]:
        self._load_model_and_tokenizer()
        assert self._tokenizer is not None
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True)
        labels = inputs.input_ids.clone()
        return {"input_ids": inputs.input_ids.to(self._device), "labels": labels.to(self._device)}

    def scan_text(self, text: str, output_path: Path):
        self._load_model_and_tokenizer()
        inputs = self._prepare_inputs(text)

        metadata = {
            "model_name": self._model_name,
            "scan_mode": "live",
            "id_to_param_name_map": self._id_to_param_name_map,
        }

        assert self._model is not None
        with MScanWriter(output_path, metadata) as writer, ScannerEngine(self._model) as scanner:
            outputs = self._model(**inputs)
            outputs.loss.backward()

            activations, gradients = scanner.get_collected_data()

            records = process_and_quantize_data(activations, gradients, self._param_name_to_id_map)

            sequence_info = {
                "sequence_id": 0,
                "num_tokens": inputs["input_ids"].shape[1],
                "source": text,
            }
            writer.append_records(records, sequence_info)

    def scan_dataset(self, dataset: Dataset, output_path: Path, batch_size: int = 1, progress_callback: Callable | None = None):
        self._load_model_and_tokenizer()
        dataloader = DataLoader(dataset, batch_size=batch_size)

        metadata = {
            "model_name": self._model_name,
            "scan_mode": "batch",
            "id_to_param_name_map": self._id_to_param_name_map,
        }

        seq_id = 0
        with MScanWriter(output_path, metadata) as writer:
            for batch in dataloader:
                inputs = {k: v.to(self._device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                assert self._model is not None
                with ScannerEngine(self._model) as scanner:
                    self._model.zero_grad()
                    outputs = self._model(**inputs)
                    outputs.loss.backward()

                    activations, gradients = scanner.get_collected_data()

                    records = process_and_quantize_data(activations, gradients, self._param_name_to_id_map)

                    sequence_info = {
                        "sequence_id": seq_id,
                        "num_tokens": inputs["input_ids"].shape[1],
                    }
                    writer.append_records(records, sequence_info)
                    seq_id += 1
                    if progress_callback:
                        progress_callback()
