"""
Differenza rispetto al file precedente:
- Prima osservavamo la proiezione finale interna alla multi-head attention.
- Qui aggiungiamo una residual connection attorno alla multi-head attention.

Scopo del file:
- Vedere che input e output della attention hanno la stessa shape.
- Vedere che la residual connection somma `embeddings` e `attention_output`.
- Preparare la struttura dei blocchi Transformer, dove i sottoblocchi vengono
  collegati con residual connections.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_22.batching import create_batch
from studio.snapshot.lezione_22.model import LanguageModel
from studio.snapshot.lezione_22.tokenizer import create_vocabulary, decode, encode


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

    attention_output, _ = model.multi_head_attention(embeddings)
    residual_embeddings = embeddings + attention_output

    manual_logits = model.output_head(residual_embeddings)
    logits, loss = model(input_tensor, target_tensor)

    max_difference = (manual_logits - logits).abs().max()

    print("Forma input:")
    print(input_tensor.shape)
    print()

    print("Primo esempio come testo:")
    print(repr(decode(input_tensor[0].tolist(), id_to_char)))
    print()

    print("Forma embeddings:")
    print(embeddings.shape)
    print()

    print("Forma attention_output:")
    print(attention_output.shape)
    print()

    print("Forma dopo residual connection:")
    print(residual_embeddings.shape)
    print()

    print("Forma logits:")
    print(logits.shape)
    print()

    print("Primo token prima della residual connection:")
    print(embeddings[0, 0])
    print()

    print("Output attention del primo token:")
    print(attention_output[0, 0])
    print()

    print("Primo token dopo residual connection:")
    print(residual_embeddings[0, 0])
    print()

    print("Differenza massima tra calcolo manuale e forward del modello:")
    print(max_difference.item())
    print()

    print("Loss iniziale:")
    print(loss.item())


if __name__ == "__main__":
    main()
