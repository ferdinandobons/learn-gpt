"""
Differenza rispetto al file precedente:
- Prima abbiamo visto che il bigram guarda solo l'ultimo token.
- Qui trasformiamo ogni token in un vettore più piccolo e più ricco prima di
  produrre i logits.

Scopo del file:
- Capire la differenza tra ID numerico del token e embedding del token.
- Vedere la forma degli embeddings: batch_size x context_size x embedding_size.
- Preparare il passo successivo: aggiungere informazioni di posizione.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_17.batching import create_batch
from studio.snapshot.lezione_17.model import LanguageModel
from studio.snapshot.lezione_17.tokenizer import create_vocabulary, decode, encode

CONTEXT_SIZE = 8
BATCH_SIZE = 4
EMBEDDING_SIZE = 16


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

    model = LanguageModel(
        vocabulary_size=vocabulary_size,
        embedding_size=EMBEDDING_SIZE,
    )

    token_embeddings = model.token_embedding_table(input_tensor)
    logits, loss = model(input_tensor, target_tensor)

    print("Forma input:")
    print(input_tensor.shape)
    print()

    print("Primo esempio come testo:")
    print(repr(decode(input_tensor[0].tolist(), id_to_char)))
    print()

    print("Grandezza vocabolario:")
    print(vocabulary_size)
    print()

    print("Embedding size:")
    print(EMBEDDING_SIZE)
    print()

    print("Forma token embeddings:")
    print(token_embeddings.shape)
    print()

    print("Forma logits:")
    print(logits.shape)
    print()

    print("Loss iniziale:")
    print(loss.item())
    print()

    first_token_id = input_tensor[0, 0].item()
    first_token_embedding = token_embeddings[0, 0]

    print("Primo token:")
    print(first_token_id, repr(decode([first_token_id], id_to_char)))
    print()

    print("Embedding del primo token:")
    print(first_token_embedding)


if __name__ == "__main__":
    main()
