"""
Differenza rispetto al file precedente:
- Prima il batch era composto da liste Python.
- Qui trasformiamo quelle liste in tensori PyTorch.

Scopo del file:
- Preparare input e target nel formato che potrà essere usato da un modello
  neurale.
- Osservare la forma del batch: batch_size x context_size.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_09.tokenizer import create_vocabulary, encode, decode

CONTEXT_SIZE = 32
BATCH_SIZE = 4


def create_example(data, context_size):
    start_position = random.randint(0, len(data) - context_size - 1)

    input_tokens = data[start_position:start_position + context_size]
    target_tokens = data[start_position + 1:start_position + context_size + 1]

    return input_tokens, target_tokens


def create_batch(data, batch_size, context_size):
    batch_inputs = []
    batch_targets = []

    for _ in range(batch_size):
        input_tokens, target_tokens = create_example(data, context_size)

        batch_inputs.append(input_tokens)
        batch_targets.append(target_tokens)

    input_tensor = torch.tensor(batch_inputs)
    target_tensor = torch.tensor(batch_targets)

    return input_tensor, target_tensor


def main():
    random.seed(42)

    text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, id_to_char = create_vocabulary(text)

    token_ids = encode(text, char_to_id)

    split_index = int(len(token_ids) * 0.9)
    training_data = token_ids[:split_index]

    input_tensor, target_tensor = create_batch(
        data=training_data,
        batch_size=BATCH_SIZE,
        context_size=CONTEXT_SIZE,
    )

    print("Input tensor:")
    print(input_tensor)
    print()

    print("Target tensor:")
    print(target_tensor)
    print()

    print("Forma input tensor:")
    print(input_tensor.shape)
    print()

    print("Forma target tensor:")
    print(target_tensor.shape)
    print()

    first_input = input_tensor[0].tolist()
    first_target = target_tensor[0].tolist()

    print("Primo input come testo:")
    print(repr(decode(first_input, id_to_char)))
    print()

    print("Primo target come testo:")
    print(repr(decode(first_target, id_to_char)))


if __name__ == "__main__":
    main()
