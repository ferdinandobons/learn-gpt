"""
Differenza rispetto al file precedente:
- Prima le funzioni erano definite nello stesso script.
- Qui le importiamo da `tokenizer.py`.

Scopo del file:
- Verificare che il modulo `tokenizer.py` funzioni correttamente quando viene
  usato da un altro file.
"""

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_04.tokenizer import create_vocabulary, encode, decode


def main():
    full_text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, id_to_char = create_vocabulary(full_text)

    sample = full_text[:200]

    token_ids = encode(sample, char_to_id)
    reconstructed_text = decode(token_ids, id_to_char)

    print("Numero caratteri diversi:", len(char_to_id))
    print("Primi 30 numeri:")
    print(token_ids[:30])
    print()
    print("Il testo ricostruito è uguale all'originale?")
    print(reconstructed_text == sample)


if __name__ == "__main__":
    main()
