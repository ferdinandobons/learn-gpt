"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 13 of the guided LearnGPT path.

File purpose:
- Run the lesson example in a reproducible way.
- Print the relevant intermediate values, tensor shapes, losses, or generated
  text for inspection.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "study_sample.txt"

sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_13.batching import create_batch
from study.snapshots.lesson_13.model import LanguageModel
from study.snapshots.lesson_13.tokenizer import create_vocabulary, decode, encode

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

    print("Original logits shape:")
    print(logits.shape)
    print()

    batch_size, context_size, vocabulary_size = logits.shape

    logits_flat = logits.reshape(batch_size * context_size, vocabulary_size)
    target_flat = target_tensor.reshape(batch_size * context_size)

    print("Logits shape after reshape:")
    print(logits_flat.shape)
    print()

    print("Target shape after reshape:")
    print(target_flat.shape)
    print()

    print("Loss of the untrained model:")
    print(loss.item())
    print()

    first_input = input_tensor[0, 0].item()
    first_target = target_tensor[0, 0].item()
    first_predicted_token = torch.argmax(logits[0, 0]).item()

    print("First token read by the model:")
    print(first_input, repr(decode([first_input], id_to_char)))
    print()

    print("Correct target for that position:")
    print(first_target, repr(decode([first_target], id_to_char)))
    print()

    print("Token chosen by the untrained model:")
    print(first_predicted_token, repr(decode([first_predicted_token], id_to_char)))


if __name__ == "__main__":
    main()
