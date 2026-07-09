"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 02 of the guided LearnGPT path.

File purpose:
- Run the lesson example in a reproducible way.
- Print the relevant intermediate values, tensor shapes, losses, or generated
  text for inspection.
"""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"


def main():
    text = DATASET_PATH.read_text(encoding="utf-8")

    unique_chars = sorted(set(text))

    char_to_id = {}
    id_to_char = {}

    for token_id, char in enumerate(unique_chars):
        char_to_id[char] = token_id
        id_to_char[token_id] = char
    print(char_to_id)
    print(id_to_char)

    sample = "The quick brown fox"

    token_ids = []
    for char in sample:
        token_id = char_to_id[char]
        token_ids.append(token_id)

    reconstructed_text = ""
    for token_id in token_ids:
        char = id_to_char[token_id]
        reconstructed_text += char

    print("Total number of characters in the text:", len(text))
    print("Number of different characters:", len(unique_chars))
    print()
    print("Original example:")
    print(sample)
    print()
    print("Example converted to numbers:")
    print(token_ids)
    print()
    print("Example reconstructed from numbers:")
    print(reconstructed_text)


if __name__ == "__main__":
    main()
