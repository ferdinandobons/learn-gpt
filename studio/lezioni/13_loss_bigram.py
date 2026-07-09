"""
Differenza rispetto al file precedente:
- Prima il modello produceva solo `logits`, cioè punteggi per il prossimo token.
- Qui confrontiamo quei punteggi con i target corretti e calcoliamo una `loss`.

Scopo del file:
- Capire che la loss misura quanto il modello sta sbagliando.
- Vedere perché dobbiamo trasformare `batch_size x context_size x vocabulary_size`
  in una tabella piatta prima di usare la cross entropy.
- Preparare il passo successivo: aggiornare i pesi del modello per ridurre la
  loss.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_13.batching import create_batch
from studio.snapshot.lezione_13.model import LanguageModel
from studio.snapshot.lezione_13.tokenizer import create_vocabulary, decode, encode

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

    logits, loss = model(input_tensor, target_tensor)

    print("Forma logits originale:")
    print(logits.shape)
    print()

    batch_size, context_size, vocabulary_size = logits.shape

    logits_flat = logits.reshape(batch_size * context_size, vocabulary_size)
    target_flat = target_tensor.reshape(batch_size * context_size)

    print("Forma logits dopo reshape:")
    print(logits_flat.shape)
    print()

    print("Forma target dopo reshape:")
    print(target_flat.shape)
    print()

    print("Loss del modello non addestrato:")
    print(loss.item())
    print()

    first_input = input_tensor[0, 0].item()
    first_target = target_tensor[0, 0].item()
    first_predicted_token = torch.argmax(logits[0, 0]).item()

    print("Primo token letto dal modello:")
    print(first_input, repr(decode([first_input], id_to_char)))
    print()

    print("Target corretto per quella posizione:")
    print(first_target, repr(decode([first_target], id_to_char)))
    print()

    print("Token scelto dal modello non addestrato:")
    print(first_predicted_token, repr(decode([first_predicted_token], id_to_char)))


if __name__ == "__main__":
    main()
