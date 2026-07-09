"""
Differenza rispetto al file precedente:
- Prima il modello aveva attention, LayerNorm e residual connection.
- Qui aggiungiamo un feed-forward network dopo la attention.

Scopo del file:
- Vedere che il feed-forward network lavora su ogni token dopo la attention.
- Vedere il passaggio interno `embedding_size -> 4 * embedding_size`.
- Vedere che il risultato finale torna alla shape `[batch_size, context_size,
  embedding_size]`, così può essere sommato con una residual connection.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_24.batching import create_batch
from studio.snapshot.lezione_24.model import LanguageModel
from studio.snapshot.lezione_24.tokenizer import create_vocabulary, decode, encode


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

    positions = torch.arange(input_tensor.shape[1])
    token_embeddings = model.token_embedding_table(input_tensor)
    position_embeddings = model.position_embedding_table(positions)
    embeddings = token_embeddings + position_embeddings

    attention_input = model.attention_layer_norm(embeddings)
    attention_output, _ = model.multi_head_attention(attention_input)
    residual_after_attention = embeddings + attention_output

    feed_forward_input = model.feed_forward_layer_norm(residual_after_attention)
    feed_forward_hidden = model.feed_forward.expand(feed_forward_input)
    feed_forward_activated = model.feed_forward.activation(feed_forward_hidden)
    feed_forward_output = model.feed_forward.project(feed_forward_activated)
    residual_after_feed_forward = residual_after_attention + feed_forward_output

    manual_logits = model.output_head(residual_after_feed_forward)
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

    print("Forma dopo attention residuale:")
    print(residual_after_attention.shape)
    print()

    print("Forma input del feed-forward:")
    print(feed_forward_input.shape)
    print()

    print("Forma dopo prima Linear del feed-forward:")
    print(feed_forward_hidden.shape)
    print()

    print("Forma dopo GELU:")
    print(feed_forward_activated.shape)
    print()

    print("Forma output del feed-forward:")
    print(feed_forward_output.shape)
    print()

    print("Forma dopo residual connection del feed-forward:")
    print(residual_after_feed_forward.shape)
    print()

    print("Forma logits:")
    print(logits.shape)
    print()

    print("Primo token prima del feed-forward:")
    print(feed_forward_input[0, 0])
    print()

    print("Primi 8 valori del primo token dopo prima Linear:")
    print(feed_forward_hidden[0, 0, :8])
    print()

    print("Primi 8 valori del primo token dopo GELU:")
    print(feed_forward_activated[0, 0, :8])
    print()

    print("Primo token in output dal feed-forward:")
    print(feed_forward_output[0, 0])
    print()

    print("Differenza massima tra calcolo manuale e forward del modello:")
    print(max_difference.item())
    print()

    print("Loss iniziale:")
    print(loss.item())


if __name__ == "__main__":
    main()
