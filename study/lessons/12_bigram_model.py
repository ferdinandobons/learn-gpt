"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 12 of the guided LearnGPT path.

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
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_12.batching import create_batch
from study.snapshots.lesson_12.model import LanguageModel
from study.snapshots.lesson_12.tokenizer import create_vocabulary, decode, encode

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

    logits = model(input_tensor)

    print("Input form:")
    print(input_tensor.shape)
    print()

    print("Target shape:")
    print(target_tensor.shape)
    print()

    print("Vocabulary size:")
    print(vocabulary_size)
    print()

    print("Logits form:")
    print(logits.shape)
    print()

    first_token = input_tensor[0, 0].item()
    first_token_scores = logits[0, 0]
    predicted_token = torch.argmax(first_token_scores).item()

    print("First token of the first example:")
    print(first_token, repr(decode([first_token], id_to_char)))
    print()

    print("Scores produced for that token:")
    print(first_token_scores)
    print()

    print("Highest scoring token according to the untrained model:")
    print(predicted_token, repr(decode([predicted_token], id_to_char)))


if __name__ == "__main__":
    main()
