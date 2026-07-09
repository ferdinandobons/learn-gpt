"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 10 of the guided LearnGPT path.

File purpose:
- Run the lesson example in a reproducible way.
- Print the relevant intermediate values, tensor shapes, losses, or generated
  text for inspection.
"""

from pathlib import Path
import random
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_10.tokenizer import create_vocabulary, encode, decode
from study.snapshots.lesson_10.batching import create_batch

CONTEXT_SIZE = 32
BATCH_SIZE = 4


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

    print("Input form:")
    print(input_tensor.shape)
    print()

    print("Target shape:")
    print(target_tensor.shape)
    print()

    print("First input as text:")
    print(repr(decode(input_tensor[0].tolist(), id_to_char)))
    print()

    print("First target as text:")
    print(repr(decode(target_tensor[0].tolist(), id_to_char)))


if __name__ == "__main__":
    main()
