"""
Differenza rispetto al file precedente:
- Prima sommavamo token embeddings e position embeddings e li mandavamo
  direttamente alla testa di output.
- Qui inseriamo una prima head di self-attention causale tra gli embeddings e
  la testa di output.

Scopo del file:
- Vedere le forme di query, key e value.
- Vedere che la self-attention crea una matrice context_size x context_size per
  ogni esempio del batch.
- Vedere che la maschera causale impedisce a una posizione di usare token
  futuri.
"""

from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_19.batching import create_batch
from studio.snapshot.lezione_19.model import LanguageModel
from studio.snapshot.lezione_19.tokenizer import create_vocabulary, decode, encode


CONTEXT_SIZE = 8
BATCH_SIZE = 4
EMBEDDING_SIZE = 16
HEAD_SIZE = 16


def main():
    random.seed(42)
    torch.manual_seed(42)
    torch.set_printoptions(precision=3, sci_mode=False)

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
        head_size=HEAD_SIZE,
    )

    positions = torch.arange(CONTEXT_SIZE)
    token_embeddings = model.token_embedding_table(input_tensor)
    position_embeddings = model.position_embedding_table(positions)
    embeddings = token_embeddings + position_embeddings

    keys = model.attention_head.key(embeddings)
    queries = model.attention_head.query(embeddings)
    values = model.attention_head.value(embeddings)

    attention_scores = queries @ keys.transpose(-2, -1)
    attention_scores = attention_scores / (HEAD_SIZE ** 0.5)

    causal_mask = model.attention_head.causal_mask[:CONTEXT_SIZE, :CONTEXT_SIZE]
    masked_attention_scores = attention_scores.masked_fill(
        causal_mask == 0,
        float("-inf"),
    )
    attention_weights = F.softmax(masked_attention_scores, dim=-1)
    attended_embeddings = attention_weights @ values

    logits, loss = model(input_tensor, target_tensor)

    print("Forma input:")
    print(input_tensor.shape)
    print()

    print("Primo esempio come testo:")
    print(repr(decode(input_tensor[0].tolist(), id_to_char)))
    print()

    print("Forma embeddings prima della attention:")
    print(embeddings.shape)
    print()

    print("Forma keys:")
    print(keys.shape)
    print()

    print("Forma queries:")
    print(queries.shape)
    print()

    print("Forma values:")
    print(values.shape)
    print()

    print("Forma attention scores:")
    print(attention_scores.shape)
    print()

    print("Maschera causale:")
    print(causal_mask)
    print()

    print("Forma attention weights:")
    print(attention_weights.shape)
    print()

    print("Attention weights del primo esempio:")
    print(attention_weights[0])
    print()

    print("Somma di ogni riga degli attention weights del primo esempio:")
    print(attention_weights[0].sum(dim=-1))
    print()

    print("Forma embeddings dopo la attention:")
    print(attended_embeddings.shape)
    print()

    print("Forma logits:")
    print(logits.shape)
    print()

    print("Loss iniziale:")
    print(loss.item())


if __name__ == "__main__":
    main()
