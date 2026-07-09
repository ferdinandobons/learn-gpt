"""
Differenza rispetto al file precedente:
- Prima concatenavamo le head e mandavamo il risultato direttamente a
  `output_head`.
- Qui aggiungiamo una proiezione finale interna alla multi-head attention.

Scopo del file:
- Vedere che la concatenazione delle head produce ancora una rappresentazione
  interna.
- Vedere che `output_projection` trasforma quella rappresentazione senza
  cambiare la shape.
- Separare il ruolo di `output_projection` dal ruolo di `output_head`.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_21.batching import create_batch
from studio.snapshot.lezione_21.model import LanguageModel
from studio.snapshot.lezione_21.tokenizer import create_vocabulary, decode, encode


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

    for head in model.multi_head_attention.heads:
        head_output, _ = head(embeddings)
        head_outputs.append(head_output)

    concatenated_embeddings = torch.cat(head_outputs, dim=-1)
    projected_embeddings = model.multi_head_attention.output_projection(
        concatenated_embeddings
    )
    model_attention_output, _ = model.multi_head_attention(embeddings)

    logits, loss = model(input_tensor, target_tensor)

    max_difference = (projected_embeddings - model_attention_output).abs().max()

    print("Forma input:")
    print(input_tensor.shape)
    print()

    print("Primo esempio come testo:")
    print(repr(decode(input_tensor[0].tolist(), id_to_char)))
    print()

    print("Forma embeddings prima della multi-head attention:")
    print(embeddings.shape)
    print()

    print("Forma output di ogni head:")
    for index, head_output in enumerate(head_outputs):
        print(f"head {index}: {head_output.shape}")
    print()

    print("Forma dopo concatenazione delle head:")
    print(concatenated_embeddings.shape)
    print()

    print("Forma dopo output_projection:")
    print(projected_embeddings.shape)
    print()

    print("Forma logits dopo output_head:")
    print(logits.shape)
    print()

    print("Primo token dopo concatenazione:")
    print(concatenated_embeddings[0, 0])
    print()

    print("Primo token dopo output_projection:")
    print(projected_embeddings[0, 0])
    print()

    print("Differenza massima tra calcolo manuale e forward del modulo:")
    print(max_difference.item())
    print()

    print("Loss iniziale:")
    print(loss.item())


if __name__ == "__main__":
    main()
