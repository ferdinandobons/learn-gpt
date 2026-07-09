"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 18 of the guided LearnGPT path.

File purpose:
- Run the lesson example in a reproducible way.
- Print the relevant intermediate values, tensor shapes, losses, or generated
  text for inspection.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_18.batching import create_batch
from study.snapshots.lesson_18.model import LanguageModel
from study.snapshots.lesson_18.tokenizer import create_vocabulary, decode, encode

CONTEXT_SIZE = 8
BATCH_SIZE = 4
EMBEDDING_SIZE = 16


def main():
    random.seed(42)
    torch.manual_seed(42)

    text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, id_to_char = create_vocabulary(text)
    vocabulary_size = len(char_to_id)

    token_ids = encode(text, char_to_id)

    split_index = int(len(token_ids) * 0.9)
    training_data = token_ids[:split_index]

    input_tensor, target_tensor = create_batch(
        data=training_data,
        batch_size=BATCH_SIZE,
        context_size=CONTEXT_SIZE,
    )

    model = LanguageModel(
        vocabulary_size=vocabulary_size,
        context_size=CONTEXT_SIZE,
        embedding_size=EMBEDDING_SIZE,
    )

    positions = torch.arange(CONTEXT_SIZE)
    token_embeddings = model.token_embedding_table(input_tensor)
    position_embeddings = model.position_embedding_table(positions)
    embeddings = token_embeddings + position_embeddings
    logits, loss = model(input_tensor, target_tensor)

    print("Input form:")
    print(input_tensor.shape)
    print()

    print("First example as text:")
    print(repr(decode(input_tensor[0].tolist(), id_to_char)))
    print()

    print("Position IDs:")
    print(positions)
    print()

    print("Token embeddings form:")
    print(token_embeddings.shape)
    print()

    print("Form position embeddings:")
    print(position_embeddings.shape)
    print()

    print("Form summed embeddings:")
    print(embeddings.shape)
    print()

    print("Logits form:")
    print(logits.shape)
    print()

    print("Initial loss:")
    print(loss.item())
    print()

    first_token_id = input_tensor[0, 0].item()
    first_token_embedding = token_embeddings[0, 0]
    prima_position_embedding = position_embeddings[0]
    first_summed_embedding = embeddings[0, 0]

    print("First token:")
    print(first_token_id, repr(decode([first_token_id], id_to_char)))
    print()

    print("Token embedding of the first token:")
    print(first_token_embedding)
    print()

    print("Position embedding at position 0:")
    print(prima_position_embedding)
    print()

    print("Amount used by the model for the first token:")
    print(first_summed_embedding)


if __name__ == "__main__":
    main()
