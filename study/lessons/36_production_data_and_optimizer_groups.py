"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 36 of the guided LearnGPT path.

File purpose:
- Run the lesson example in a reproducible way.
- Print the relevant intermediate values, tensor shapes, losses, or generated
  text for inspection.
"""

from pathlib import Path
import sys
import tempfile

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_36.batching import (
    create_batch,
    load_training_and_validation_data,
)
from study.snapshots.lesson_36.model import LanguageModel
from study.snapshots.lesson_36.tokenizer import (
    DEFAULT_ENCODING_NAME,
    decode,
    encode,
    get_vocabulary_size,
)
from study.snapshots.lesson_36.training import configure_optimizer


CONTEXT_SIZE = 8
EMBEDDING_SIZE = 16
NUM_HEADS = 4
HEAD_SIZE = EMBEDDING_SIZE // NUM_HEADS
NUM_TRANSFORMER_BLOCKS = 1
WEIGHT_DECAY = 0.01
LEARNING_RATE = 0.001


def main():
    text = (PROJECT_DIR / "data" / "study_sample.txt").read_text(encoding="utf-8")
    token_ids = np.asarray(encode(text, DEFAULT_ENCODING_NAME), dtype=np.uint16)
    split_index = int(len(token_ids) * 0.9)

    with tempfile.TemporaryDirectory(prefix="learngpt_lesson_36_") as temporary_dir:
        data_dir = Path(temporary_dir)
        token_ids[:split_index].tofile(data_dir / "train.bin")
        token_ids[split_index:].tofile(data_dir / "val.bin")
        training_data, validation_data = load_training_and_validation_data(data_dir)
        input_tensor, target_tensor = create_batch(
            data=training_data,
            batch_size=2,
            context_size=CONTEXT_SIZE,
        )
        training_token_count = len(training_data)
        validation_token_count = len(validation_data)
        del training_data
        del validation_data

    model = LanguageModel(
        vocabulary_size=get_vocabulary_size(DEFAULT_ENCODING_NAME),
        context_size=CONTEXT_SIZE,
        embedding_size=EMBEDDING_SIZE,
        head_size=HEAD_SIZE,
        num_heads=NUM_HEADS,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
    )
    optimizer = configure_optimizer(
        model=model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("Tokenizer:", DEFAULT_ENCODING_NAME)
    print("GPT-2 BPE vocabulary size:", get_vocabulary_size(DEFAULT_ENCODING_NAME))
    print("First decoded tokens:", repr(decode(token_ids[:12], DEFAULT_ENCODING_NAME)))
    print()

    print("Memmapped token splits:")
    print("training tokens:", training_token_count)
    print("validation tokens:", validation_token_count)
    print()

    print("Batch created from the memmapped training split:")
    print("input_tensor shape:", tuple(input_tensor.shape))
    print("target_tensor shape:", tuple(target_tensor.shape))
    print()

    print("Optimizer groups:")
    for index, parameter_group in enumerate(optimizer.param_groups):
        parameter_count = sum(
            parameter.numel()
            for parameter in parameter_group["params"]
        )
        print(
            index,
            "weight_decay:",
            parameter_group["weight_decay"],
            "parameters:",
            parameter_count,
        )


if __name__ == "__main__":
    main()
