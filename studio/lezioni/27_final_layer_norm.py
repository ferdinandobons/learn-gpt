"""
Differenza rispetto al file precedente:
- Prima il modello passava l'output dell'ultimo `TransformerBlock` direttamente
  a `output_head`.
- Qui aggiungiamo una `final_layer_norm` tra l'ultimo blocco e `output_head`.

Scopo del file:
- Vedere dove si trova la LayerNorm finale nel flusso del modello.
- Verificare che la LayerNorm finale non cambia la shape `[4, 8, 16]`.
- Avvicinare la struttura didattica alla struttura di nanoGPT, dove dopo tutti
  i blocchi viene applicata una normalizzazione finale.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_27.batching import create_batch
from studio.snapshot.lezione_27.model import LanguageModel
from studio.snapshot.lezione_27.tokenizer import create_vocabulary, decode, encode


CONTEXT_SIZE = 8
BATCH_SIZE = 4
EMBEDDING_SIZE = 16
NUM_HEADS = 4
HEAD_SIZE = EMBEDDING_SIZE // NUM_HEADS
NUM_TRANSFORMER_BLOCKS = 3


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
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
    )

    positions = torch.arange(input_tensor.shape[1], device=input_tensor.device)
    token_embeddings = model.token_embedding_table(input_tensor)
    position_embeddings = model.position_embedding_table(positions)
    embeddings = token_embeddings + position_embeddings

    block_output = embeddings

    for transformer_block in model.transformer_blocks:
        block_output = transformer_block(block_output)

    normalized_output = model.final_layer_norm(block_output)
    manual_logits = model.output_head(normalized_output)
    logits, loss = model(input_tensor, target_tensor)

    logits_difference = (manual_logits - logits).abs().max()

    first_token_before_norm = block_output[0, 0]
    first_token_after_norm = normalized_output[0, 0]

    print("Forma input:")
    print(input_tensor.shape)
    print()

    print("Primo esempio come testo:")
    print(repr(decode(input_tensor[0].tolist(), id_to_char)))
    print()

    print("Forma embeddings iniziali:")
    print(embeddings.shape)
    print()

    print("Forma output dopo tutti i TransformerBlock:")
    print(block_output.shape)
    print()

    print("Forma output dopo final_layer_norm:")
    print(normalized_output.shape)
    print()

    print("Media del primo token prima e dopo final_layer_norm:")
    print(first_token_before_norm.mean().item())
    print(first_token_after_norm.mean().item())
    print()

    print("Deviazione standard del primo token prima e dopo final_layer_norm:")
    print(first_token_before_norm.std(unbiased=False).item())
    print(first_token_after_norm.std(unbiased=False).item())
    print()

    print("Forma logits:")
    print(logits.shape)
    print()

    print("Differenza massima tra logits manuali e forward del modello:")
    print(logits_difference.item())
    print()

    print("Loss iniziale:")
    print(loss.item())


if __name__ == "__main__":
    main()
