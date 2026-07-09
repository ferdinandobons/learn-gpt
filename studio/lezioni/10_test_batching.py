"""
Differenza rispetto al file precedente:
- Prima creavamo il batch direttamente nello script.
- Qui usiamo il nuovo modulo `batching.py`.

Scopo del file:
- Verificare che `batching.py` produca ancora input e target corretti.
- Controllare che i tensori abbiano forma batch_size x context_size.
"""

from pathlib import Path
import random
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_10.tokenizer import create_vocabulary, encode, decode
from studio.snapshot.lezione_10.batching import create_batch

CONTEXT_SIZE = 32
BATCH_SIZE = 4


def main():
    random.seed(42)

    text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, id_to_char = create_vocabulary(text)

    token_ids = encode(text, char_to_id)

    split_index = int(len(token_ids) * 0.9)
    training_data = token_ids[:split_index]

    input_tensor, target_tensor = create_batch(
        data=training_data,
        batch_size=BATCH_SIZE,
        context_size=CONTEXT_SIZE,
    )

    print("Forma input:")
    print(input_tensor.shape)
    print()

    print("Forma target:")
    print(target_tensor.shape)
    print()

    print("Primo input come testo:")
    print(repr(decode(input_tensor[0].tolist(), id_to_char)))
    print()

    print("Primo target come testo:")
    print(repr(decode(target_tensor[0].tolist(), id_to_char)))


if __name__ == "__main__":
    main()
