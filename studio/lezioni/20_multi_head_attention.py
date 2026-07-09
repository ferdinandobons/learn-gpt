"""
Differenza rispetto al file precedente:
- Prima usavamo una sola head di self-attention causale.
- Qui usiamo più head in parallelo e concateniamo i loro risultati.

Scopo del file:
- Vedere che ogni head produce una rappresentazione più piccola.
- Vedere che la concatenazione delle head ricostruisce la dimensione interna
  del modello.
- Vedere che `output_head` trasforma la rappresentazione multi-head in logits
  sul vocabolario.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_20.batching import create_batch
from studio.snapshot.lezione_20.model import LanguageModel
from studio.snapshot.lezione_20.tokenizer import create_vocabulary, decode, encode


CONTEXT_SIZE = 8
BATCH_SIZE = 4
EMBEDDING_SIZE = 16
NUM_HEADS = 4
HEAD_SIZE = EMBEDDING_SIZE // NUM_HEADS


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
        num_heads=NUM_HEADS,
    )

    positions = torch.arange(CONTEXT_SIZE)
    token_embeddings = model.token_embedding_table(input_tensor)
    position_embeddings = model.position_embedding_table(positions)
    embeddings = token_embeddings + position_embeddings

    head_outputs = []
    attention_weights_by_head = []

    for head in model.multi_head_attention.heads:
        head_output, attention_weights = head(embeddings)
        head_outputs.append(head_output)
        attention_weights_by_head.append(attention_weights)

    multi_head_embeddings, _ = model.multi_head_attention(embeddings)
    logits, loss = model(input_tensor, target_tensor)

    print("Forma input:")
    print(input_tensor.shape)
    print()

    print("Primo esempio come testo:")
    print(repr(decode(input_tensor[0].tolist(), id_to_char)))
    print()

    print("Numero di head:")
    print(NUM_HEADS)
    print()

    print("Head size:")
    print(HEAD_SIZE)
    print()

    print("Forma embeddings prima della multi-head attention:")
    print(embeddings.shape)
    print()

    print("Forma output di ogni head:")
    for index, head_output in enumerate(head_outputs):
        print(f"head {index}: {head_output.shape}")
    print()

    print("Forma attention weights di ogni head:")
    for index, attention_weights in enumerate(attention_weights_by_head):
        print(f"head {index}: {attention_weights.shape}")
    print()

    print("Attention weights della head 0, primo esempio:")
    print(attention_weights_by_head[0][0])
    print()

    print("Somma di ogni riga degli attention weights della head 0:")
    print(attention_weights_by_head[0][0].sum(dim=-1))
    print()

    print("Forma dopo concatenazione delle head:")
    print(multi_head_embeddings.shape)
    print()

    print("Forma logits:")
    print(logits.shape)
    print()

    print("Loss iniziale:")
    print(loss.item())


if __name__ == "__main__":
    main()
