"""
Differenza rispetto al file precedente:
- Prima usavamo solo l'inizio del testo.
- Qui prendiamo esempi casuali da punti diversi del training set.

Scopo del file:
- Far vedere che il modello deve allenarsi su molte zone del testo, non sempre
  sugli stessi primi caratteri.
"""

from pathlib import Path
import random
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_07.tokenizer import create_vocabulary, encode, decode

CONTEXT_SIZE = 32


def create_example(data, context_size):
    start_position = random.randint(0, len(data) - context_size - 1)

    input_tokens = data[start_position:start_position + context_size]
    target_tokens = data[start_position + 1:start_position + context_size + 1]

    return input_tokens, target_tokens


def main():
    random.seed(42)

    text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, id_to_char = create_vocabulary(text)

    token_ids = encode(text, char_to_id)

    split_index = int(len(token_ids) * 0.9)
    training_data = token_ids[:split_index]
    validation_data = token_ids[split_index:]

    print("Token training:", len(training_data))
    print("Token validation:", len(validation_data))
    print()

    for example_number in range(5):
        input_tokens, target_tokens = create_example(training_data, CONTEXT_SIZE)

        input_text = decode(input_tokens, id_to_char)
        target_text = decode(target_tokens, id_to_char)

        print("Esempio", example_number + 1)
        print("Input:")
        print(repr(input_text))
        print("Target:")
        print(repr(target_text))
        print()


if __name__ == "__main__":
    main()
