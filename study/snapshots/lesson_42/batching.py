"""
Changes compared with the previous files:
- This module belongs to the cleaned final LearnGPT project.
- It uses the English public project layout and the FineWeb-Edu/BPE pipeline.

File purpose:
- Create training batches from memmapped token arrays.
"""

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


def load_training_and_validation_data(data_dir=DEFAULT_DATA_DIR):
    training_data = load_token_data(data_dir=data_dir, split="train")
    validation_data = load_token_data(data_dir=data_dir, split="val")

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
