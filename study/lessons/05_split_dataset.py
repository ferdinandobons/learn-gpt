"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 05 of the guided LearnGPT path.

File purpose:
- Run the lesson example in a reproducible way.
- Print the relevant intermediate values, tensor shapes, losses, or generated
  text for inspection.
"""

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "study_sample.txt"

sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_05.tokenizer import create_vocabulary, encode, decode


def main():
    text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, id_to_char = create_vocabulary(text)

    token_ids = encode(text, char_to_id)

    split_index = int(len(token_ids) * 0.9)

    training_data = token_ids[:split_index]
    validation_data = token_ids[split_index:]

    print("Total characters in text:", len(text))
    print("Total numeric tokens:", len(token_ids))
    print("Token training:", len(training_data))
    print("Token validation:", len(validation_data))
    print()

    print("First 80 training tokens:")
    print(training_data[:80])
    print()

    print("The same tokens decoded back to text:")
    print(decode(training_data[:80], id_to_char))


if __name__ == "__main__":
    main()
