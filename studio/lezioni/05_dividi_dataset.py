"""
Differenza rispetto al file precedente:
- Prima testavamo solo il tokenizer su un piccolo esempio.
- Qui codifichiamo tutto il corpus e lo dividiamo in training e validation.

Scopo del file:
- Preparare due parti del dataset: una per allenare il modello e una per
  controllare se sta imparando.
"""

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_05.tokenizer import create_vocabulary, encode, decode


def main():
    text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, id_to_char = create_vocabulary(text)

    token_ids = encode(text, char_to_id)

    split_index = int(len(token_ids) * 0.9)

    training_data = token_ids[:split_index]
    validation_data = token_ids[split_index:]

    print("Caratteri totali nel testo:", len(text))
    print("Token numerici totali:", len(token_ids))
    print("Token training:", len(training_data))
    print("Token validation:", len(validation_data))
    print()

    print("Primi 80 token del training:")
    print(training_data[:80])
    print()

    print("Gli stessi token riconvertiti in testo:")
    print(decode(training_data[:80], id_to_char))


if __name__ == "__main__":
    main()
