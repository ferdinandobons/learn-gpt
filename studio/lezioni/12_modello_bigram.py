"""
Differenza rispetto al file precedente:
- Prima controllavamo soltanto che PyTorch funzionasse.
- Qui creiamo il primo modello neurale e gli passiamo un batch reale della
  campione FineWeb-Edu.

Scopo del file:
- Capire che un modello riceve token numerici.
- Vedere che il modello produce punteggi chiamati `logits`.
- Osservare la forma dell'output: batch_size x context_size x vocabulary_size.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_12.batching import create_batch
from studio.snapshot.lezione_12.model import LanguageModel
from studio.snapshot.lezione_12.tokenizer import create_vocabulary, decode, encode

CONTEXT_SIZE = 8
BATCH_SIZE = 4


def main():
    random.seed(42)
    torch.manual_seed(42)

    text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, id_to_char = create_vocabulary(text)
    vocabulary_size = len(char_to_id)

    token_ids = encode(text, char_to_id)

    split_index = int(len(token_ids) * 0.9)
    training_data = token_ids[:split_index]

    input_tensor, target_tensor = create_batch(
        data=training_data,
        batch_size=BATCH_SIZE,
        context_size=CONTEXT_SIZE,
    )

    model = LanguageModel(vocabulary_size=vocabulary_size)

    logits = model(input_tensor)

    print("Forma input:")
    print(input_tensor.shape)
    print()

    print("Forma target:")
    print(target_tensor.shape)
    print()

    print("Grandezza vocabolario:")
    print(vocabulary_size)
    print()

    print("Forma logits:")
    print(logits.shape)
    print()

    first_token = input_tensor[0, 0].item()
    first_token_scores = logits[0, 0]
    predicted_token = torch.argmax(first_token_scores).item()

    print("Primo token del primo esempio:")
    print(first_token, repr(decode([first_token], id_to_char)))
    print()

    print("Punteggi prodotti per quel token:")
    print(first_token_scores)
    print()

    print("Token con punteggio più alto secondo il modello non addestrato:")
    print(predicted_token, repr(decode([predicted_token], id_to_char)))


if __name__ == "__main__":
    main()
