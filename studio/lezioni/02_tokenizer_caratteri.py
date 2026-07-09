"""
Differenza rispetto al file precedente:
- Prima leggevamo soltanto il testo.
- Qui costruiamo un vocabolario di caratteri e trasformiamo un esempio in
  numeri.

Scopo del file:
- Mostrare che un testo deve diventare una sequenza di ID numerici.
- Verificare che i numeri possano essere riconvertiti nel testo originale.
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

    sample = "Nel mezzo del cammin"

    token_ids = []
    for char in sample:
        token_id = char_to_id[char]
        token_ids.append(token_id)

    reconstructed_text = ""
    for token_id in token_ids:
        char = id_to_char[token_id]
        reconstructed_text += char

    print("Numero caratteri totali nel testo:", len(text))
    print("Numero caratteri diversi:", len(unique_chars))
    print()
    print("Esempio originale:")
    print(sample)
    print()
    print("Esempio trasformato in numeri:")
    print(token_ids)
    print()
    print("Esempio ricostruito dai numeri:")
    print(reconstructed_text)


if __name__ == "__main__":
    main()
