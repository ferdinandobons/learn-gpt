"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 39 of the guided LearnGPT path.

File purpose:
- Run the lesson example in a reproducible way.
- Print the relevant intermediate values, tensor shapes, losses, or generated
  text for inspection.
"""

from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_39.model import LanguageModel


VOCABULARY_SIZE = 100
CONTEXT_SIZE = 8
EMBEDDING_SIZE = 16
NUM_HEADS = 4
HEAD_SIZE = EMBEDDING_SIZE // NUM_HEADS
NUM_TRANSFORMER_BLOCKS = 1


def main():
    torch.manual_seed(42)
    model = LanguageModel(
        vocabulary_size=VOCABULARY_SIZE,
        context_size=CONTEXT_SIZE,
        embedding_size=EMBEDDING_SIZE,
        head_size=HEAD_SIZE,
        num_heads=NUM_HEADS,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
    )

    input_ids = torch.randint(
        low=0,
        high=VOCABULARY_SIZE,
        size=(2, CONTEXT_SIZE),
    )
    target_ids = torch.randint(
        low=0,
        high=VOCABULARY_SIZE,
        size=(2, CONTEXT_SIZE),
    )

    training_logits, loss = model(input_ids, target_ids)
    generation_logits = model(input_ids)

    print("Training logits shape:")
    print(tuple(training_logits.shape))
    print("Generation logits shape:")
    print(tuple(generation_logits.shape))
    print("Finite loss:")
    print(bool(loss.isfinite().item()))


if __name__ == "__main__":
    main()
