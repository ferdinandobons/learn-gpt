"""
Differenza rispetto agli script precedenti:
- Prima le funzioni del tokenizer vivevano dentro un file di esercizio.
- Qui diventano un modulo riutilizzabile da altri script.

Scopo del file:
- Contenere le funzioni comuni per creare il vocabolario, codificare testo in
  numeri e decodificare numeri in testo.
"""

def create_vocabulary(text):
    unique_chars = sorted(set(text))

    char_to_id = {}
    id_to_char = {}

    for token_id, char in enumerate(unique_chars):
        char_to_id[char] = token_id
        id_to_char[token_id] = char

    return char_to_id, id_to_char


def encode(text, char_to_id):
    token_ids = []

    for char in text:
        token_id = char_to_id[char]
        token_ids.append(token_id)

    return token_ids


def decode(token_ids, id_to_char):
    text = ""

    for token_id in token_ids:
        char = id_to_char[token_id]
        text += char

    return text
