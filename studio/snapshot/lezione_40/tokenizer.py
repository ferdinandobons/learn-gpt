"""
Differenza rispetto ai file precedenti:
- Prima il progetto finale usava un tokenizer a caratteri costruito dal testo.
- Qui usiamo il tokenizer BPE GPT-2 tramite `tiktoken`, più vicino a un GPT
  reale e coerente con i file FineWeb-Edu già processati.

Scopo del file:
- Fornire funzioni comuni per codificare e decodificare testo con GPT-2 BPE.
- Esporre la dimensione del vocabolario usata dal modello finale.
- Evitare vocabolari creati a mano come `char_to_id` e `id_to_char`.
"""

import tiktoken


DEFAULT_ENCODING_NAME = "gpt2"


def get_tokenizer(encoding_name=DEFAULT_ENCODING_NAME):
    return tiktoken.get_encoding(encoding_name)


def get_vocabulary_size(encoding_name=DEFAULT_ENCODING_NAME):
    tokenizer = get_tokenizer(encoding_name)

    return tokenizer.n_vocab


def encode(text, encoding_name=DEFAULT_ENCODING_NAME):
    tokenizer = get_tokenizer(encoding_name)

    return tokenizer.encode(text, allowed_special={"<|endoftext|>"})


def decode(token_ids, encoding_name=DEFAULT_ENCODING_NAME):
    tokenizer = get_tokenizer(encoding_name)

    return tokenizer.decode(list(token_ids))
