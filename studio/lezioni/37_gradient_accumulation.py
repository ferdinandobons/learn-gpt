"""
Differenza rispetto al file precedente:
- Prima ogni step usava un solo batch e chiamava subito `optimizer.step()`.
- Qui ogni step può sommare i gradienti di più micro-batch.

Scopo del file:
- Verificare che `gradient_accumulation_steps` funzioni.
- Capire la differenza tra micro-batch e batch effettivo.
- Eseguire un training minimo senza usare tutto FineWeb-Edu.
"""

from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = Path("/private/tmp/learngpt_lesson_38/checkpoint.pt")
sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_37.model import LanguageModel
from studio.snapshot.lezione_37.training import configure_optimizer, train_model


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
    print("Batch effettivo:")
    print(BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    print("Checkpoint:")
    print(best_checkpoint_path)
    print("History:")
    print(history)


if __name__ == "__main__":
    main()
