"""
Changes compared with the previous files:
- This module adds a reproducible experimental subset without replacing the
  canonical FineWeb-Edu dataset.

File purpose:
- Copy randomly selected token chunks from a completed processed dataset into
  a smaller train.bin/val.bin pair for compute-bounded experiments.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .batching import load_dataset_metadata, validate_dataset_metadata


DEFAULT_SOURCE_DATA_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "fineweb_edu"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "processed"
    / "fineweb_edu_experiment_1g"
)
DEFAULT_TARGET_GB = 1.0
DEFAULT_VALIDATION_RATIO = 0.01
DEFAULT_SEED = 1337
DEFAULT_CHUNK_TOKENS = 65_536


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a reproducible randomized subset of processed token data.",
    )
    parser.add_argument("--source-data-dir", type=Path, default=DEFAULT_SOURCE_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-gb", type=float, default=DEFAULT_TARGET_GB)
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def split_target_tokens(total_tokens, validation_ratio):
    if total_tokens < 2:
        raise ValueError("The requested subset must contain at least two tokens.")

    validation_tokens = round(total_tokens * validation_ratio)
    validation_tokens = min(max(validation_tokens, 1), total_tokens - 1)
    training_tokens = total_tokens - validation_tokens

    return training_tokens, validation_tokens


def select_chunk_indices(source_tokens, target_tokens, chunk_tokens, random_generator):
    if target_tokens < 1:
        raise ValueError("target_tokens must be at least 1.")

    available_chunks = source_tokens // chunk_tokens
    required_chunks = math.ceil(target_tokens / chunk_tokens)
    if required_chunks > available_chunks:
        raise ValueError(
            "The source split is too small for this subset size and chunk size."
        )

    return random_generator.choice(
        available_chunks,
        size=required_chunks,
        replace=False,
    )


def copy_random_chunks(
    source_data,
    output_file,
    target_tokens,
    chunk_tokens,
    random_generator,
):
    selected_indices = select_chunk_indices(
        source_tokens=len(source_data),
        target_tokens=target_tokens,
        chunk_tokens=chunk_tokens,
        random_generator=random_generator,
    )
    tokens_remaining = target_tokens

    for chunk_index in selected_indices:
        current_tokens = min(chunk_tokens, tokens_remaining)
        start = int(chunk_index) * chunk_tokens
        token_chunk = np.asarray(
            source_data[start : start + current_tokens],
            dtype=np.uint16,
        )
        output_file.write(token_chunk.tobytes())
        tokens_remaining -= current_tokens

    if tokens_remaining != 0:
        raise RuntimeError("The subset writer did not produce the requested token count.")


def prepare_subset(args):
    if args.target_gb <= 0:
        raise ValueError("--target-gb must be greater than 0.")
    if not 0 < args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be greater than 0 and less than 1.")
    if args.seed < 0:
        raise ValueError("--seed cannot be negative.")
    if args.chunk_tokens < 1:
        raise ValueError("--chunk-tokens must be at least 1.")

    source_data_dir = Path(args.source_data_dir)
    source_metadata = load_dataset_metadata(source_data_dir)
    validate_dataset_metadata(source_metadata, require_complete=True)

    source_train_path = source_data_dir / "train.bin"
    source_validation_path = source_data_dir / "val.bin"
    if not source_train_path.exists() or not source_validation_path.exists():
        raise FileNotFoundError(
            "The source dataset must contain both train.bin and val.bin."
        )

    source_training_data = np.memmap(source_train_path, dtype=np.uint16, mode="r")
    source_validation_data = np.memmap(
        source_validation_path,
        dtype=np.uint16,
        mode="r",
    )
    source_counters = source_metadata.get("counters") or {}
    expected_training_tokens = source_counters.get("train_tokens")
    expected_validation_tokens = source_counters.get("val_tokens")
    if (
        expected_training_tokens is not None
        and len(source_training_data) != expected_training_tokens
    ):
        raise ValueError("The source train.bin length does not match meta.json.")
    if (
        expected_validation_tokens is not None
        and len(source_validation_data) != expected_validation_tokens
    ):
        raise ValueError("The source val.bin length does not match meta.json.")

    target_bytes = int(args.target_gb * 1024**3)
    target_tokens = target_bytes // np.dtype(np.uint16).itemsize
    training_tokens, validation_tokens = split_target_tokens(
        total_tokens=target_tokens,
        validation_ratio=args.validation_ratio,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.bin"
    validation_path = output_dir / "val.bin"
    metadata_path = output_dir / "meta.json"
    existing_paths = [
        path
        for path in (train_path, validation_path, metadata_path)
        if path.exists()
    ]
    if existing_paths and not args.overwrite:
        names = ", ".join(path.name for path in existing_paths)
        raise FileExistsError(
            f"Files already exist in {output_dir}: {names}. Use --overwrite to recreate them."
        )

    metadata = {
        "dataset_name": source_metadata.get("dataset_name"),
        "dataset_config": source_metadata.get("dataset_config"),
        "encoding_name": source_metadata["encoding_name"],
        "dtype": "uint16",
        "complete": False,
        "preparation_mode": "randomized_subset",
        "source_data_dir": str(source_data_dir),
        "source_train_tokens": int(len(source_training_data)),
        "source_val_tokens": int(len(source_validation_data)),
        "target_gb": args.target_gb,
        "target_bytes": target_bytes,
        "validation_ratio": args.validation_ratio,
        "subset_seed": args.seed,
        "subset_chunk_tokens": args.chunk_tokens,
        "train_path": str(train_path),
        "val_path": str(validation_path),
        "counters": {
            "train_tokens": training_tokens,
            "val_tokens": validation_tokens,
            "total_tokens": target_tokens,
            "train_bytes": training_tokens * np.dtype(np.uint16).itemsize,
            "val_bytes": validation_tokens * np.dtype(np.uint16).itemsize,
            "total_bytes": target_tokens * np.dtype(np.uint16).itemsize,
        },
    }
    write_json(metadata_path, metadata)

    random_generator = np.random.default_rng(args.seed)
    try:
        with train_path.open("wb") as training_file:
            copy_random_chunks(
                source_data=source_training_data,
                output_file=training_file,
                target_tokens=training_tokens,
                chunk_tokens=args.chunk_tokens,
                random_generator=random_generator,
            )
        with validation_path.open("wb") as validation_file:
            copy_random_chunks(
                source_data=source_validation_data,
                output_file=validation_file,
                target_tokens=validation_tokens,
                chunk_tokens=args.chunk_tokens,
                random_generator=random_generator,
            )
    except Exception:
        metadata["complete"] = False
        write_json(metadata_path, metadata)
        raise

    metadata["complete"] = (
        train_path.stat().st_size == metadata["counters"]["train_bytes"]
        and validation_path.stat().st_size == metadata["counters"]["val_bytes"]
    )
    write_json(metadata_path, metadata)

    if not metadata["complete"]:
        raise RuntimeError("The subset files do not match their expected size.")

    print("Experimental subset prepared:")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    return metadata


def main():
    args = parse_args()
    prepare_subset(args)


if __name__ == "__main__":
    main()
