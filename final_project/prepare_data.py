"""
Changes compared with the previous files:
- This module belongs to the cleaned final LearnGPT project.
- It uses the English public project layout and the FineWeb-Edu/BPE pipeline.

File purpose:
- Prepare FineWeb-Edu token files for local training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken


DEFAULT_DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DEFAULT_DATASET_CONFIG = "sample-10BT"
DEFAULT_ENCODING_NAME = "gpt2"
DEFAULT_TARGET_GB = 5.0
DEFAULT_VALIDATION_RATIO = 0.01
DEFAULT_HF_HOME = "/private/tmp/learngpt_huggingface"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare FineWeb-Edu as train.bin and val.bin with GPT-2 BPE.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--target-gb", type=float, default=DEFAULT_TARGET_GB)
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO)
    parser.add_argument("--encoding-name", default=DEFAULT_ENCODING_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "processed" / "fineweb_edu",
    )
    parser.add_argument("--progress-mb", type=int, default=64)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def open_output_files(output_dir, overwrite):
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    meta_path = output_dir / "meta.json"

    existing = [path for path in [train_path, val_path, meta_path] if path.exists()]
    if existing and not overwrite:
        existing_names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Files already exist in {output_dir}: {existing_names}. "
            "Use --overwrite to recreate them."
        )

    train_file = train_path.open("wb")
    val_file = val_path.open("wb")

    return train_path, val_path, meta_path, train_file, val_file


def split_name_for_document(document_index, validation_ratio):
    if validation_ratio <= 0:
        return "train"

    validation_every = max(1, round(1 / validation_ratio))
    if document_index % validation_every == 0:
        return "val"

    return "train"


def encode_text(text, encoding):
    token_ids = encoding.encode_ordinary(text)
    token_ids.append(encoding.eot_token)

    return np.array(token_ids, dtype=np.uint16)


def trim_to_remaining_tokens(token_array, remaining_bytes):
    remaining_tokens = remaining_bytes // np.dtype(np.uint16).itemsize

    if remaining_tokens <= 0:
        return token_array[:0]

    return token_array[:remaining_tokens]


def disable_datasets_shared_memory():
    import datasets.iterable_dataset

    datasets.iterable_dataset._maybe_share_with_torch_persistent_workers = (
        lambda value: value
    )


def prepare_dataset(args):
    if args.target_gb <= 0:
        raise ValueError("--target-gb must be greater than 0.")

    if not 0 <= args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1.")

    if args.progress_mb < 1:
        raise ValueError("--progress-mb must be at least 1.")

    if args.max_documents is not None and args.max_documents < 1:
        raise ValueError("--max-documents must be at least 1 when set.")

    target_bytes = int(args.target_gb * 1024**3)
    progress_bytes = args.progress_mb * 1024**2
    next_progress_bytes = progress_bytes
    start_time = time.time()

    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(DEFAULT_HF_HOME) / "datasets"))

    from datasets import load_dataset

    disable_datasets_shared_memory()

    encoding = tiktoken.get_encoding(args.encoding_name)
    train_path, val_path, meta_path, train_file, val_file = open_output_files(
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    counters = {
        "documents_seen": 0,
        "documents_written": 0,
        "train_tokens": 0,
        "val_tokens": 0,
        "train_bytes": 0,
        "val_bytes": 0,
        "total_bytes": 0,
    }

    metadata = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": "train",
        "encoding_name": args.encoding_name,
        "target_gb": args.target_gb,
        "target_bytes": target_bytes,
        "validation_ratio": args.validation_ratio,
        "dtype": "uint16",
        "train_path": str(train_path),
        "val_path": str(val_path),
        "complete": False,
        "counters": counters,
    }

    write_json(meta_path, metadata)

    dataset = load_dataset(
        args.dataset_name,
        name=args.dataset_config,
        split="train",
        streaming=True,
    )

    print("Dataset:", args.dataset_name)
    print("Config:", args.dataset_config)
    print("Encoding:", args.encoding_name)
    print("Output:", args.output_dir)
    print("Target bytes:", target_bytes)
    print()

    try:
        for document in dataset:
            counters["documents_seen"] += 1

            text = document.get("text") or ""
            if not text.strip():
                continue

            token_array = encode_text(text=text, encoding=encoding)
            remaining_bytes = target_bytes - counters["total_bytes"]
            token_array = trim_to_remaining_tokens(
                token_array=token_array,
                remaining_bytes=remaining_bytes,
            )

            if len(token_array) == 0:
                break

            split_name = split_name_for_document(
                document_index=counters["documents_seen"],
                validation_ratio=args.validation_ratio,
            )

            if split_name == "val":
                val_file.write(token_array.tobytes())
                counters["val_tokens"] += int(len(token_array))
                counters["val_bytes"] += int(token_array.nbytes)
            else:
                train_file.write(token_array.tobytes())
                counters["train_tokens"] += int(len(token_array))
                counters["train_bytes"] += int(token_array.nbytes)

            counters["documents_written"] += 1
            counters["total_bytes"] += int(token_array.nbytes)

            if counters["total_bytes"] >= next_progress_bytes:
                elapsed_seconds = max(time.time() - start_time, 0.001)
                mb_written = counters["total_bytes"] / 1024**2
                mb_per_second = mb_written / elapsed_seconds
                print(
                    f"{mb_written:.0f} MiB scritti "
                    f"({mb_per_second:.2f} MiB/s, "
                    f"documenti={counters['documents_written']})",
                    flush=True,
                )
                metadata["elapsed_seconds"] = elapsed_seconds
                write_json(meta_path, metadata)
                next_progress_bytes += progress_bytes

            if counters["total_bytes"] >= target_bytes:
                break

            if args.max_documents is not None:
                if counters["documents_seen"] >= args.max_documents:
                    break
    finally:
        train_file.flush()
        val_file.flush()
        train_file.close()
        val_file.close()

    remaining_bytes = target_bytes - counters["total_bytes"]
    metadata["complete"] = remaining_bytes < np.dtype(np.uint16).itemsize
    metadata["elapsed_seconds"] = time.time() - start_time
    metadata["counters"] = counters
    write_json(meta_path, metadata)

    print()
    print("Preparazione completata:" if metadata["complete"] else "Preparazione interrotta:")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    return metadata


def main():
    args = parse_args()
    prepare_dataset(args)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
