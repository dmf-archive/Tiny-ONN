# MSCAN (`.mscan`) File Format Specification v1.1

## 1. Overview

The `.mscan` format is a **single-file, binary-native structure** designed for storing and managing data from neural network scanning sessions. It prioritizes storage efficiency, high-performance I/O for batch processing, and data integrity. The format is self-contained and easily archivable.

## 2. File Structure

A scan session is represented by a single file with a `.mscan` extension. The internal layout is as follows:

```
[HEADER] [JSON METADATA] [RAW BINARY RECORDS ...]
```

- **Header**: A fixed-size binary header containing a magic number, version, and offsets.
- **JSON Metadata**: A variable-length, UTF-8 encoded JSON string containing all session metadata.
- **Raw Binary Records**: A tightly packed array of binary records, appended sequentially.

## 3. Header (`struct`)

The header is a small, fixed-size block at the beginning of the file, defined by a Python `struct`.

- **Format**: `<5sHII`
  - `5s`: Magic number (`b"MSCAN"`)
  - `H`: Version number (unsigned short, e.g., 1)
  - `I`: Total header length (unsigned int), including the JSON metadata block.
  - `I`: Total number of binary records in the file (unsigned int).

## 4. Metadata (JSON)

This JSON block serves as the central index and configuration store for the entire scan session.

```json
{
  "format_version": "1.1",
  "model_name": "Qwen/Qwen3-1.7B",
  "id_to_param_name_map": {
    "0": "model.layers.0.self_attn.q_proj.weight",
    "1": "model.layers.0.self_attn.k_proj.weight",
    "...": "..."
  },
  "sequences": [
    {
      "sequence_id": 0,
      "num_tokens": 128,
      "source": "User: Hello\nAssistant: Hi, how can I help?"
    },
    {
      "sequence_id": 1,
      "num_tokens": 96,
      "source": "User: What is the capital of France?\nAssistant: Paris."
    }
  ]
}
```

- **`format_version` (string)**: The version of the `.mscan` specification (e.g., "1.1").
- **`model_name` (string)**: The identifier of the model being scanned.
- **`id_to_param_name_map` (object)**: A dictionary mapping integer IDs to full parameter names for space efficiency.
- **`sequences` (array)**: **The primary source of truth for scan history.** An array of objects, where each object represents a single, complete scan event (e.g., one user-assistant exchange).
  - `sequence_id` (integer): A unique, sequential ID for the scan event within the session.
  - `num_tokens` (integer): The number of tokens generated and analyzed in this specific scan event.
  - `source` (string, optional): The raw text of the conversation for this event.

## 5. Raw Data Records (`numpy.ndarray`)

- These records contain the core metrics, stored in a compact, binary format.
- They are appended to the file after the metadata block.
- The data structure for each record is defined by a NumPy `dtype`.

### Data Type Definition

```python
import numpy as np

METRICS_DTYPE = np.dtype([
    ('token_idx', 'u4'),      # 4 bytes, global token index across all sequences
    ('param_id', 'u2'),       # 2 bytes, foreign key to id_to_param_name_map
    ('block_idx', 'u2'),      # 2 bytes, index of the block within the parameter
    ('activation', 'u2'),     # 2 bytes, quantized activation value (uint16)
    ('grad_norm', 'u2'),      # 2 bytes, quantized gradient norm value (uint16)
])
# Total size per record: 12 bytes
```

This structure ensures high storage efficiency and allows for extremely fast loading and processing using memory-mapping (`np.memmap`).

## 6. Replay Mode Data Loading Logic

To correctly interpret the data in Replay mode, the total number of tokens must be calculated by summing the `num_tokens` field from all entries in the `sequences` array.

**Correct Calculation:**
`total_tokens = sum(seq['num_tokens'] for seq in metadata.get('sequences', []))`

**Incorrect Calculation (to be avoided):**
`total_tokens = np.max(data["token_idx"]) + 1`
