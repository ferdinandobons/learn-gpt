"""
Differenza rispetto al file precedente:
- Prima sommavamo direttamente `embeddings` e `attention_output`.
- Qui applichiamo `LayerNorm` agli embeddings prima di passarli alla
  multi-head attention.

Scopo del file:
- Vedere che `LayerNorm` cambia i valori interni degli embeddings.
- Vedere che `LayerNorm` non cambia la shape del tensore.
- Preparare la struttura pre-norm dei blocchi Transformer:
  `x = x + attention(layer_norm(x))`.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_23.batching import create_batch
from studio.snapshot.lezione_23.model import LanguageModel
from studio.snapshot.lezione_23.tokenizer import create_vocabulary, decode, encode


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

    normalized_embeddings = model.attention_layer_norm(embeddings)
    attention_output, _ = model.multi_head_attention(normalized_embeddings)
    residual_embeddings = embeddings + attention_output

    manual_logits = model.output_head(residual_embeddings)
    logits, loss = model(input_tensor, target_tensor)

    max_difference = (manual_logits - logits).abs().max()

    first_token_before = embeddings[0, 0]
    first_token_after = normalized_embeddings[0, 0]

    print("Forma input:")
    print(input_tensor.shape)
    print()

    print("Primo esempio come testo:")
    print(repr(decode(input_tensor[0].tolist(), id_to_char)))
    print()

    print("Forma embeddings prima di LayerNorm:")
    print(embeddings.shape)
    print()

    print("Forma embeddings dopo LayerNorm:")
    print(normalized_embeddings.shape)
    print()

    print("Primo token prima di LayerNorm:")
    print(first_token_before)
    print()

    print("Primo token dopo LayerNorm:")
    print(first_token_after)
    print()

    print("Media del primo token prima e dopo LayerNorm:")
    print(first_token_before.mean().item())
    print(first_token_after.mean().item())
    print()

    print("Deviazione standard del primo token prima e dopo LayerNorm:")
    print(first_token_before.std(unbiased=False).item())
    print(first_token_after.std(unbiased=False).item())
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

    print("Differenza massima tra calcolo manuale e forward del modello:")
    print(max_difference.item())
    print()

    print("Loss iniziale:")
    print(loss.item())


if __name__ == "__main__":
    main()
