import json
import struct
from pathlib import Path
from typing import IO, Any

import numpy as np

METRICS_DTYPE = np.dtype([
    ('token_idx', 'u4'),
    ('param_id', 'u2'),
    ('block_idx', 'u2'),
    ('activation', 'u2'),
    ('grad_norm', 'u2'),
])

MAGIC_NUMBER = b"MSCAN"
VERSION = 1
HEADER_FORMAT = "<5sHII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class MScanWriter:
    def __init__(self, filepath: Path, metadata: dict[str, Any]):
        self._filepath = filepath.with_suffix(".mscan")
        self._metadata = metadata
        self._file: IO[bytes] | None = None
        self._record_count = 0
        self._header_len = 0

    def __enter__(self):
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._filepath, "wb")

        json_meta = json.dumps(self._metadata, indent=2).encode("utf-8")
        self._header_len = HEADER_SIZE + len(json_meta)

        header = struct.pack(
            HEADER_FORMAT, MAGIC_NUMBER, VERSION, self._header_len, 0
        )
        self._file.write(header)
        self._file.write(json_meta)
        return self

    def append_records(self, records: np.ndarray, sequence_info: dict[str, Any]):
        if self._file is None or self._file.closed:
            raise OSError("MScanWriter is not open. Use 'with' statement.")
        if records.dtype != METRICS_DTYPE:
            raise ValueError(f"Incorrect numpy dtype. Expected {METRICS_DTYPE}, got {records.dtype}")

        if "sequences" not in self._metadata:
            self._metadata["sequences"] = []
        self._metadata["sequences"].append(sequence_info)

        self._file.write(records.tobytes())
        self._record_count += len(records)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None and not self._file.closed:
            self._file.seek(0)

            json_meta = json.dumps(self._metadata, indent=2).encode("utf-8")
            self._header_len = HEADER_SIZE + len(json_meta)

            final_header = struct.pack(
                HEADER_FORMAT,
                MAGIC_NUMBER,
                VERSION,
                self._header_len,
                self._record_count,
            )
            self._file.write(final_header)
            self._file.write(json_meta)

            self._file.close()
        self._file = None

    def close(self):
        self.__exit__(None, None, None)


def load_mscan_file(filepath: Path) -> tuple[dict[str, Any], np.ndarray | None]:
    mscan_filepath = filepath.with_suffix(".mscan")
    if not mscan_filepath.exists() or mscan_filepath.stat().st_size < HEADER_SIZE:
        raise FileNotFoundError(f"Invalid or missing mscan file: {mscan_filepath}")

    with open(mscan_filepath, "rb") as f:
        magic, version, header_len, num_records = struct.unpack(
            HEADER_FORMAT, f.read(HEADER_SIZE)
        )

        if magic != MAGIC_NUMBER:
            raise ValueError("Invalid magic number. Not a mscan file.")
        if version != VERSION:
            raise ValueError(f"Unsupported mscan version: {version}")

        json_meta_bytes = f.read(header_len - HEADER_SIZE)
        metadata = json.loads(json_meta_bytes.decode("utf-8"))

    if num_records > 0:
        data = np.memmap(
            mscan_filepath,
            dtype=METRICS_DTYPE,
            mode="r",
            offset=header_len,
            shape=(num_records,),
        )
        return metadata, data

    return metadata, None
