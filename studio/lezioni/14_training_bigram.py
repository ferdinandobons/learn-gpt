"""
Differenza rispetto al file precedente:
- Prima calcolavamo la loss, ma non cambiavamo i pesi del modello.
- Qui usiamo la loss per aggiornare i pesi con un piccolo training loop.

Scopo del file:
- Capire la sequenza base del training: forward, loss, zero_grad, backward,
  step.
- Verificare che la loss possa scendere dopo alcuni aggiornamenti.
- Preparare il passo successivo: generare testo con il modello addestrato.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_14.batching import create_batch
from studio.snapshot.lezione_14.model import LanguageModel
from studio.snapshot.lezione_14.tokenizer import create_vocabulary, encode

CONTEXT_SIZE = 8
BATCH_SIZE = 32
TRAINING_STEPS = 300
LEARNING_RATE = 0.01


def main():
    random.seed(42)
    torch.manual_seed(42)

    text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, _ = create_vocabulary(text)
    vocabulary_size = len(char_to_id)

    token_ids = encode(text, char_to_id)

    split_index = int(len(token_ids) * 0.9)
    training_data = token_ids[:split_index]

    model = LanguageModel(vocabulary_size=vocabulary_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(TRAINING_STEPS):
        input_tensor, target_tensor = create_batch(
            data=training_data,
            batch_size=BATCH_SIZE,
            context_size=CONTEXT_SIZE,
        )

        logits, loss = model(input_tensor, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print("Step:", step, "Loss:", loss.item())

    print("Step:", TRAINING_STEPS, "Loss finale:", loss.item())


if __name__ == "__main__":
    main()
