"""
Changes compared with the previous files:
- This module belongs to the cleaned final LearnGPT project.
- It uses the English public project layout and the FineWeb-Edu/BPE pipeline.

File purpose:
- Wrap the GPT-2 BPE tokenizer used by the final project.
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
