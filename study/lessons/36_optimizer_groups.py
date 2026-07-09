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

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_36.batching import create_batch
from study.snapshots.lesson_36.model import LanguageModel
from study.snapshots.lesson_36.training import configure_optimizer


VOCABULARY_SIZE = 100
CONTEXT_SIZE = 8
EMBEDDING_SIZE = 16
NUM_HEADS = 4
HEAD_SIZE = EMBEDDING_SIZE // NUM_HEADS
NUM_TRANSFORMER_BLOCKS = 1
WEIGHT_DECAY = 0.01
LEARNING_RATE = 0.001


def main():
    data = np.arange(CONTEXT_SIZE + 1, dtype=np.uint16)
    input_tensor, target_tensor = create_batch(
        data=data,
        batch_size=2,
        context_size=CONTEXT_SIZE,
    )

    model = LanguageModel(
        vocabulary_size=VOCABULARY_SIZE,
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

    print("Batch creato dall'unica finestra valida:")
    print("input_tensor shape:", tuple(input_tensor.shape))
    print("target_tensor shape:", tuple(target_tensor.shape))
    print()

    print("Gruppi dell'optimizer:")
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
