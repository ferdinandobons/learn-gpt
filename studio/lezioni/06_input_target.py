"""
Differenza rispetto al file precedente:
- Prima dividevamo il dataset in training e validation.
- Qui creiamo una coppia input/target spostata di un token.

Scopo del file:
- Mostrare il problema che un GPT deve imparare: dato un contesto, prevedere il
  token successivo.
"""

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_06.tokenizer import create_vocabulary, encode, decode

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

    print("Input come testo:")
    print(repr(decode(input_tokens, id_to_char)))
    print()

    print("Target come testo:")
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
