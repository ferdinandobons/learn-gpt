"""
Differenza rispetto al file precedente:
- Prima creavamo un singolo esempio casuale alla volta.
- Qui raccogliamo più esempi insieme in un batch.

Scopo del file:
- Introdurre l'idea di batch: un gruppo di esempi input/target usati insieme
  durante il training.
"""

from pathlib import Path
import random
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_08.tokenizer import create_vocabulary, encode, decode

CONTEXT_SIZE = 32
BATCH_SIZE = 4


def create_example(data, context_size):
    start_position = random.randint(0, len(data) - context_size - 1)

    input_tokens = data[start_position:start_position + context_size]
    target_tokens = data[start_position + 1:start_position + context_size + 1]

    return input_tokens, target_tokens


def create_batch(data, batch_size, context_size):
    batch_inputs = []
    batch_targets = []

    for _ in range(batch_size):
        input_tokens, target_tokens = create_example(data, context_size)

        batch_inputs.append(input_tokens)
        batch_targets.append(target_tokens)

    return batch_inputs, batch_targets


def main():
    random.seed(42)

    text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, id_to_char = create_vocabulary(text)

    token_ids = encode(text, char_to_id)

    split_index = int(len(token_ids) * 0.9)
    training_data = token_ids[:split_index]

    batch_inputs, batch_targets = create_batch(
        data=training_data,
        batch_size=BATCH_SIZE,
        context_size=CONTEXT_SIZE,
    )

    print("Numero esempi nel batch:", len(batch_inputs))
    print("Lunghezza di ogni esempio:", len(batch_inputs[0]))
    print()

    for indice in range(BATCH_SIZE):
        print("Esempio", indice + 1)
        print("Input:")
        print(repr(decode(batch_inputs[indice], id_to_char)))
        print("Target:")
        print(repr(decode(batch_targets[indice], id_to_char)))
        print()


if __name__ == "__main__":
    main()
