"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 37 of the guided LearnGPT path.

File purpose:
- Run the lesson example in a reproducible way.
- Print the relevant intermediate values, tensor shapes, losses, or generated
  text for inspection.
"""

from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = Path(tempfile.gettempdir()) / "learngpt_lesson_37" / "checkpoint.pt"
sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_37.model import LanguageModel
from study.snapshots.lesson_37.training import configure_optimizer, train_model


VOCABULARY_SIZE = 100
CONTEXT_SIZE = 8
BATCH_SIZE = 2
EMBEDDING_SIZE = 16
NUM_HEADS = 4
HEAD_SIZE = EMBEDDING_SIZE // NUM_HEADS
NUM_TRANSFORMER_BLOCKS = 1
GRADIENT_ACCUMULATION_STEPS = 2


def main():
    torch.manual_seed(42)
    training_data = (np.arange(512, dtype=np.uint16) % VOCABULARY_SIZE).astype(np.uint16)
    validation_data = (np.arange(128, dtype=np.uint16) % VOCABULARY_SIZE).astype(np.uint16)

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
        learning_rate=0.001,
        weight_decay=0.01,
    )

    history, best_checkpoint_path = train_model(
        model=model,
        optimizer=optimizer,
        training_data=training_data,
        validation_data=validation_data,
        batch_size=BATCH_SIZE,
        context_size=CONTEXT_SIZE,
        training_steps=2,
        eval_interval=1,
        eval_batches=1,
        checkpoint_path=CHECKPOINT_PATH,
        model_config={
            "vocabulary_size": VOCABULARY_SIZE,
            "context_size": CONTEXT_SIZE,
            "embedding_size": EMBEDDING_SIZE,
            "head_size": HEAD_SIZE,
            "num_heads": NUM_HEADS,
            "num_transformer_blocks": NUM_TRANSFORMER_BLOCKS,
        },
        tokenizer_config={"encoding_name": "demo"},
        base_learning_rate=0.001,
        min_learning_rate=0.0001,
        warmup_steps=1,
        decay_steps=2,
        gradient_clip=1.0,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        device=torch.device("cpu"),
    )

    print("Gradient accumulation steps:")
    print(GRADIENT_ACCUMULATION_STEPS)
    print("Effective batch size:")
    print(BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    print("Checkpoint:")
    print(best_checkpoint_path)
    print("History:")
    print(history)


if __name__ == "__main__":
    main()
