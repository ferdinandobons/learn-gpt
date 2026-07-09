"""
Differenza rispetto al file precedente:
- Prima leggevamo la loss di training da un batch corrente e da un solo batch
  di controllo.
- Qui stimiamo una media su più batch, separando training loss e validation
  loss.

Scopo del file:
- Usare `estimate_loss` per valutare il modello senza aggiornare i pesi.
- Confrontare la loss su dati di training e dati di validation.
- Avvicinare il training loop alla struttura usata da nanoGPT.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_29.batching import create_batch
from studio.snapshot.lezione_29.model import LanguageModel
from studio.snapshot.lezione_29.tokenizer import create_vocabulary, decode, encode
from studio.snapshot.lezione_29.training import estimate_loss


CONTEXT_SIZE = 8
BATCH_SIZE = 4
EMBEDDING_SIZE = 16
NUM_HEADS = 4
HEAD_SIZE = EMBEDDING_SIZE // NUM_HEADS
NUM_TRANSFORMER_BLOCKS = 3
LEARNING_RATE = 0.003
TRAINING_STEPS = 40
PRINT_EVERY = 10
EVAL_BATCHES = 5
GENERATED_TOKENS = 80


def print_losses(label, losses):
    print(label)
    print(f"training loss:   {losses['training']:.4f}")
    print(f"validation loss: {losses['validation']:.4f}")
    print()


def main():
    random.seed(42)
    torch.manual_seed(42)

    text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, id_to_char = create_vocabulary(text)
    vocabulary_size = len(char_to_id)

    token_ids = encode(text, char_to_id)

    split_index = int(len(token_ids) * 0.9)
    training_data = token_ids[:split_index]
    validation_data = token_ids[split_index:]

    model = LanguageModel(
        vocabulary_size=vocabulary_size,
        context_size=CONTEXT_SIZE,
        embedding_size=EMBEDDING_SIZE,
        head_size=HEAD_SIZE,
        num_heads=NUM_HEADS,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    initial_losses = estimate_loss(
        model=model,
        training_data=training_data,
        validation_data=validation_data,
        batch_size=BATCH_SIZE,
        context_size=CONTEXT_SIZE,
        eval_batches=EVAL_BATCHES,
    )
    print_losses("Loss stimate prima del training:", initial_losses)

    for step in range(1, TRAINING_STEPS + 1):
        input_tensor, target_tensor = create_batch(
            data=training_data,
            batch_size=BATCH_SIZE,
            context_size=CONTEXT_SIZE,
        )

        _, loss = model(input_tensor, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % PRINT_EVERY == 0:
            losses = estimate_loss(
                model=model,
                training_data=training_data,
                validation_data=validation_data,
                batch_size=BATCH_SIZE,
                context_size=CONTEXT_SIZE,
                eval_batches=EVAL_BATCHES,
            )
            print_losses(f"Loss stimate dopo step {step:02d}:", losses)

    prompt_ids = torch.zeros((1, 1), dtype=torch.long)
    generated_ids = model.generate(prompt_ids, max_new_tokens=GENERATED_TOKENS)
    generated_text = decode(generated_ids[0].tolist(), id_to_char)

    print("Testo generato dopo il training:")
    print(repr(generated_text))


if __name__ == "__main__":
    main()
