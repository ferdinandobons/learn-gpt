"""
Changes compared with the previous files:
- This module belongs to the cleaned final LearnGPT project.
- It uses the English public project layout and the FineWeb-Edu/BPE pipeline.

File purpose:
- Create training batches from memmapped token arrays.
"""

import json
from pathlib import Path

import numpy as np
import torch


DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "fineweb_edu"
)


def load_token_data(data_dir=DEFAULT_DATA_DIR, split="train"):
    data_dir = Path(data_dir)
    data_path = data_dir / f"{split}.bin"

    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")

    return np.memmap(data_path, dtype=np.uint16, mode="r")


def load_dataset_metadata(data_dir=DEFAULT_DATA_DIR):
    metadata_path = Path(data_dir) / "meta.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")

    return json.loads(metadata_path.read_text(encoding="utf-8"))


def validate_dataset_metadata(metadata, encoding_name=None, require_complete=True):
    if metadata.get("dtype") != "uint16":
        raise ValueError("The prepared dataset dtype must be uint16.")

    if require_complete and metadata.get("complete") is not True:
        raise ValueError(
            "The prepared dataset is incomplete. Finish data preparation before training."
        )

    prepared_encoding = metadata.get("encoding_name")
    if encoding_name is not None and prepared_encoding != encoding_name:
        raise ValueError(
            "Tokenizer mismatch: dataset uses "
            f"{prepared_encoding!r}, but training requested {encoding_name!r}."
        )


def load_training_and_validation_data(
    data_dir=DEFAULT_DATA_DIR,
    encoding_name=None,
    require_complete=True,
):
    metadata = load_dataset_metadata(data_dir)
    validate_dataset_metadata(
        metadata,
        encoding_name=encoding_name,
        require_complete=require_complete,
    )
    counters = metadata.get("counters") or {}
    expected_train_tokens = counters.get("train_tokens")
    expected_val_tokens = counters.get("val_tokens")
    if expected_train_tokens is not None and expected_train_tokens < 1:
        raise ValueError("meta.json reports an empty training split.")
    if expected_val_tokens is not None and expected_val_tokens < 1:
        raise ValueError("meta.json reports an empty validation split.")

    training_data = load_token_data(data_dir=data_dir, split="train")
    validation_data = load_token_data(data_dir=data_dir, split="val")

    if expected_train_tokens is not None and len(training_data) != expected_train_tokens:
        raise ValueError("train.bin length does not match meta.json.")
    if expected_val_tokens is not None and len(validation_data) != expected_val_tokens:
        raise ValueError("val.bin length does not match meta.json.")

    return training_data, validation_data


def create_batch(data, batch_size, context_size, device=None):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if context_size < 1:
        raise ValueError("context_size must be at least 1.")

    if len(data) <= context_size:
        raise ValueError("The dataset must contain more tokens than the context_size.")

    max_start_position = len(data) - context_size - 1
    start_positions = torch.randint(
        low=0,
        high=max_start_position + 1,
        size=(batch_size,),
    )

    input_chunks = []
    target_chunks = []

    for start_position in start_positions.tolist():
        token_chunk = np.asarray(
            data[start_position : start_position + context_size + 1],
            dtype=np.int64,
        )
        input_chunks.append(torch.from_numpy(token_chunk[:-1].copy()))
        target_chunks.append(torch.from_numpy(token_chunk[1:].copy()))

    input_tensor = torch.stack(input_chunks)
    target_tensor = torch.stack(target_chunks)

    if device is not None:
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

    return input_tensor, target_tensor
