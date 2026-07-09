"""
Differenza rispetto al file precedente:
- Prima osservavamo il flusso del modello con più `TransformerBlock` e
  `final_layer_norm`.
- Qui usiamo quel modello in un vero training loop.

Scopo del file:
- Addestrare per pochi step il `LanguageModel` Transformer.
- Vedere la sequenza pratica: batch, forward, loss, zero_grad, backward,
  optimizer.step.
- Verificare che i pesi cambiano dopo il training.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_28.batching import create_batch
from studio.snapshot.lezione_28.model import LanguageModel
from studio.snapshot.lezione_28.tokenizer import create_vocabulary, decode, encode


CONTEXT_SIZE = 8
BATCH_SIZE = 4
EMBEDDING_SIZE = 16
NUM_HEADS = 4
HEAD_SIZE = EMBEDDING_SIZE // NUM_HEADS
NUM_TRANSFORMER_BLOCKS = 3
LEARNING_RATE = 0.003
TRAINING_STEPS = 30
PRINT_EVERY = 10
GENERATED_TOKENS = 80


def main():
    random.seed(42)
    torch.manual_seed(42)

    text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, id_to_char = create_vocabulary(text)
    vocabulary_size = len(char_to_id)

    token_ids = encode(text, char_to_id)

    split_index = int(len(token_ids) * 0.9)
    training_data = token_ids[:split_index]

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

    check_input, check_target = create_batch(
        data=training_data,
        batch_size=BATCH_SIZE,
        context_size=CONTEXT_SIZE,
    )
    _, initial_loss = model(check_input, check_target)

    first_parameter_before = next(model.parameters()).detach().clone()

    print("Loss sul batch di controllo prima del training:")
    print(initial_loss.item())
    print()

    for step in range(1, TRAINING_STEPS + 1):
        input_tensor, target_tensor = create_batch(
            data=training_data,
            batch_size=BATCH_SIZE,
            context_size=CONTEXT_SIZE,
        )

        logits, loss = model(input_tensor, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % PRINT_EVERY == 0:
            print(f"Step {step:02d} - loss batch corrente: {loss.item():.4f}")

    _, final_loss = model(check_input, check_target)
    first_parameter_after = next(model.parameters()).detach()
    parameter_difference = (first_parameter_after - first_parameter_before).abs().max()

    prompt_ids = torch.zeros((1, 1), dtype=torch.long)
    generated_ids = model.generate(prompt_ids, max_new_tokens=GENERATED_TOKENS)
    generated_text = decode(generated_ids[0].tolist(), id_to_char)

    print()
    print("Loss sullo stesso batch di controllo dopo il training:")
    print(final_loss.item())
    print()

    print("Differenza massima nel primo parametro dopo il training:")
    print(parameter_difference.item())
    print()

    print("Testo generato dopo il breve training:")
    print(repr(generated_text))


if __name__ == "__main__":
    main()
