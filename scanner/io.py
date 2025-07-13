import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

METRICS_DTYPE = np.dtype(
    [
        ("token_idx", "u4"),
        ("param_id", "u2"),
        ("block_idx", "u2"),
        ("activation", "u2"),
        ("grad_norm", "u2"),
    ]
)
MAGIC_NUMBER = b"MSCAN"
HEADER_FORMAT = "<5sHII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def create_mscan_file(filepath: Path, metadata: dict[str, Any]):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    json_meta = json.dumps(metadata, indent=2).encode("utf-8")
    header = struct.pack(
        HEADER_FORMAT, MAGIC_NUMBER, 1, HEADER_SIZE + len(json_meta), 0
    )
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(json_meta)


def load_mscan_file(
    filepath: Path,
) -> tuple[dict[str, Any], np.ndarray | None]:
    if not filepath.exists() or filepath.stat().st_size < HEADER_SIZE:
        raise FileNotFoundError(f"Invalid or missing mscan file: {filepath}")

    with open(filepath, "rb") as f:
        magic, version, header_len, num_records = struct.unpack(
            HEADER_FORMAT, f.read(HEADER_SIZE)
        )
        if magic != MAGIC_NUMBER:
            raise ValueError("Invalid magic number. Not a mscan file.")
        if version != 1:
            raise ValueError(f"Unsupported mscan version: {version}")

        json_meta_bytes = f.read(header_len - HEADER_SIZE)
        metadata = json.loads(json_meta_bytes.decode("utf-8"))

    if num_records > 0:
        data = np.memmap(
            filepath,
            dtype=METRICS_DTYPE,
            mode="r",
            offset=header_len,
            shape=(num_records,),
        )
        return metadata, data
    return metadata, None


def append_records_to_mscan(filepath: Path, new_records: np.ndarray, sequence_info: dict[str, Any]):
    if not new_records.size:
        return

    metadata, old_data = load_mscan_file(filepath)

    # Append the new sequence information to the sequences list
    sequences = metadata.get("sequences", [])
    sequences.append(sequence_info)
    metadata["sequences"] = sequences

    num_old_records = old_data.shape[0] if old_data is not None else 0
    num_new_records = new_records.shape[0]
    total_records = num_old_records + num_new_records

    temp_filepath = filepath.with_suffix(".tmp")

    json_meta = json.dumps(metadata, indent=2).encode("utf-8")
    new_header_len = HEADER_SIZE + len(json_meta)
    new_header = struct.pack(
        HEADER_FORMAT, MAGIC_NUMBER, 1, new_header_len, total_records
    )

    with open(temp_filepath, "wb") as f:
        f.write(new_header)
        f.write(json_meta)
        if old_data is not None:
            f.write(old_data.tobytes())
        f.write(new_records.tobytes())

    filepath.unlink()
    temp_filepath.rename(filepath)
