"""
Differenza rispetto al file precedente:
- Prima, durante la generazione, `output_head` produceva logits per tutte le
  posizioni del contesto.
- Qui, quando non passiamo target, produce logits solo per l'ultimo token.

Scopo del file:
- Vedere la differenza tra forward di training e forward di generazione.
- Verificare che la generazione lavori con shape `[batch_size, 1, vocabulary_size]`.
- Ridurre lavoro inutile sul vocabolario durante l'inference.
"""

from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_39.model import LanguageModel


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
    print("Loss finita:")
    print(bool(loss.isfinite().item()))


if __name__ == "__main__":
    main()
