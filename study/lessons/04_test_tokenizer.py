"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 04 of the guided LearnGPT path.

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

from study.snapshots.lesson_04.tokenizer import create_vocabulary, encode, decode


def main():
    full_text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, id_to_char = create_vocabulary(full_text)

    sample = full_text[:200]

    token_ids = encode(sample, char_to_id)
    reconstructed_text = decode(token_ids, id_to_char)

    print("Number of different characters:", len(char_to_id))
    print("First 30 token IDs:")
    print(token_ids[:30])
    print()
    print("Is the reconstructed text the same as the original?")
    print(reconstructed_text == sample)


if __name__ == "__main__":
    main()
