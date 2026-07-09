"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 06 of the guided LearnGPT path.

File purpose:
- Run the lesson example in a reproducible way.
- Print the relevant intermediate values, tensor shapes, losses, or generated
  text for inspection.
"""

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_06.tokenizer import create_vocabulary, encode, decode

CONTEXT_SIZE = 24


def main():
    text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, id_to_char = create_vocabulary(text)

    token_ids = encode(text, char_to_id)

    input_tokens = token_ids[:CONTEXT_SIZE]
    target_tokens = token_ids[1:CONTEXT_SIZE + 1]

    print("Input tokens:")
    print(input_tokens)
    print()

    print("Target tokens:")
    print(target_tokens)
    print()

    print("Input as text:")
    print(repr(decode(input_tokens, id_to_char)))
    print()

    print("Target as text:")
    print(repr(decode(target_tokens, id_to_char)))
    print()

    print("Esempi di previsione:")
    for position in range(CONTEXT_SIZE):
        context = input_tokens[:position + 1]
        next_token = target_tokens[position]

        context_text = decode(context, id_to_char)
        next_char = decode([next_token], id_to_char)

        print(repr(context_text), "->", repr(next_char))


if __name__ == "__main__":
    main()
