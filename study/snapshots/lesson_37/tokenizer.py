"""
Changes compared with the previous files:
- This module is part of the project-code snapshot used by lesson 37.
- It keeps the lesson independent from future changes in `final_project`.

File purpose:
- Provide the code needed by the matching lesson script.
- Preserve a stable reference point for the course examples.
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
