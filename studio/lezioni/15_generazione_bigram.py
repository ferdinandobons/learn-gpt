"""
Differenza rispetto al file precedente:
- Prima addestravamo il modello e guardavamo solo la loss.
- Qui usiamo il modello addestrato per generare nuovi caratteri.

Scopo del file:
- Capire come si passa da `logits` a probabilità.
- Campionare il prossimo token con `torch.multinomial`.
- Generare una piccola sequenza di testo partendo da un prompt.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_15.batching import create_batch
from studio.snapshot.lezione_15.model import LanguageModel
from studio.snapshot.lezione_15.tokenizer import create_vocabulary, decode, encode

CONTEXT_SIZE = 8
BATCH_SIZE = 32
TRAINING_STEPS = 500
LEARNING_RATE = 0.01
MAX_NEW_TOKENS = 300


def main():
    random.seed(42)
    torch.manual_seed(42)

    text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, id_to_char = create_vocabulary(text)
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

        if step % 100 == 0:
            print("Step:", step, "Loss:", loss.item())

    print("Step:", TRAINING_STEPS, "Loss finale:", loss.item())
    print()

    prompt = "N"
    prompt_ids = encode(prompt, char_to_id)
    input_ids = torch.tensor([prompt_ids])

    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    generated_text = decode(generated_ids[0].tolist(), id_to_char)

    print("Testo generato:")
    print(generated_text)


if __name__ == "__main__":
    main()
