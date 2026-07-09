"""
Differenza rispetto al file precedente:
- Prima ogni token diventava un token embedding.
- Qui aggiungiamo anche un position embedding, cioè un vettore legato alla
  posizione del token dentro il contesto.

Scopo del file:
- Capire perché il modello deve conoscere anche la posizione dei token.
- Vedere la forma dei position embeddings: context_size x embedding_size.
- Vedere che token embeddings e position embeddings si possono sommare perché
  hanno la stessa dimensione finale.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_18.batching import create_batch
from studio.snapshot.lezione_18.model import LanguageModel
from studio.snapshot.lezione_18.tokenizer import create_vocabulary, decode, encode

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
        context_size=CONTEXT_SIZE,
        embedding_size=EMBEDDING_SIZE,
    )

    positions = torch.arange(CONTEXT_SIZE)
    token_embeddings = model.token_embedding_table(input_tensor)
    position_embeddings = model.position_embedding_table(positions)
    embeddings = token_embeddings + position_embeddings
    logits, loss = model(input_tensor, target_tensor)

    print("Forma input:")
    print(input_tensor.shape)
    print()

    print("Primo esempio come testo:")
    print(repr(decode(input_tensor[0].tolist(), id_to_char)))
    print()

    print("Position IDs:")
    print(positions)
    print()

    print("Forma token embeddings:")
    print(token_embeddings.shape)
    print()

    print("Forma position embeddings:")
    print(position_embeddings.shape)
    print()

    print("Forma embeddings sommati:")
    print(embeddings.shape)
    print()

    print("Forma logits:")
    print(logits.shape)
    print()

    print("Loss iniziale:")
    print(loss.item())
    print()

    first_token_id = input_tensor[0, 0].item()
    first_token_embedding = token_embeddings[0, 0]
    prima_position_embedding = position_embeddings[0]
    first_summed_embedding = embeddings[0, 0]

    print("Primo token:")
    print(first_token_id, repr(decode([first_token_id], id_to_char)))
    print()

    print("Token embedding del primo token:")
    print(first_token_embedding)
    print()

    print("Position embedding della posizione 0:")
    print(prima_position_embedding)
    print()

    print("Somma usata dal modello per il primo token:")
    print(first_summed_embedding)


if __name__ == "__main__":
    main()
