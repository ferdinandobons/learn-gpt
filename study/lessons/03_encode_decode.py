"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 03 of the guided LearnGPT path.

File purpose:
- Run the lesson example in a reproducible way.
- Print the relevant intermediate values, tensor shapes, losses, or generated
  text for inspection.
"""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"


def create_vocabulary(text):
    unique_chars = sorted(set(text))

    char_to_id = {}
    id_to_char = {}

    for token_id, char in enumerate(unique_chars):
        char_to_id[char] = token_id
        id_to_char[token_id] = char
    
    # print(f"Carattere a ID: {char_to_id}")
    # print("")
    # print(f"ID a Carattere: {id_to_char}")

    return char_to_id, id_to_char


def encode(text, char_to_id):
    token_ids = []

    for char in text:
        token_id = char_to_id[char]
        token_ids.append(token_id)
    print(token_ids)

    return token_ids


def decode(token_ids, id_to_char):
    text = ""

    for token_id in token_ids:
        char = id_to_char[token_id]
        text += char

    return text


def main():
    full_text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, id_to_char = create_vocabulary(full_text)

    sample = full_text[:100]

    token_ids = encode(sample, char_to_id)
    reconstructed_text = decode(token_ids, id_to_char)

    print("Vocabolario:", len(char_to_id), "different characters")
    print()
    print("Original text:")
    print(sample)
    print()
    print("Numeri:")
    print(token_ids)
    print()
    print("Reconstructed text:")
    print(reconstructed_text)
    print()
    print("Is the reconstructed text the same as the original?")
    print(reconstructed_text == sample)


if __name__ == "__main__":
    main()
