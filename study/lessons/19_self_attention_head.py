"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 19 of the guided LearnGPT path.

File purpose:
- Run the lesson example in a reproducible way.
- Print the relevant intermediate values, tensor shapes, losses, or generated
  text for inspection.
"""

from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_19.batching import create_batch
from study.snapshots.lesson_19.model import LanguageModel
from study.snapshots.lesson_19.tokenizer import create_vocabulary, decode, encode


CONTEXT_SIZE = 8
BATCH_SIZE = 4
EMBEDDING_SIZE = 16
HEAD_SIZE = 16


def main():
    random.seed(42)
    torch.manual_seed(42)
    torch.set_printoptions(precision=3, sci_mode=False)

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
        head_size=HEAD_SIZE,
    )

    positions = torch.arange(CONTEXT_SIZE)
    token_embeddings = model.token_embedding_table(input_tensor)
    position_embeddings = model.position_embedding_table(positions)
    embeddings = token_embeddings + position_embeddings

    keys = model.attention_head.key(embeddings)
    queries = model.attention_head.query(embeddings)
    values = model.attention_head.value(embeddings)

    attention_scores = queries @ keys.transpose(-2, -1)
    attention_scores = attention_scores / (HEAD_SIZE ** 0.5)

    causal_mask = model.attention_head.causal_mask[:CONTEXT_SIZE, :CONTEXT_SIZE]
    masked_attention_scores = attention_scores.masked_fill(
        causal_mask == 0,
        float("-inf"),
    )
    attention_weights = F.softmax(masked_attention_scores, dim=-1)
    attended_embeddings = attention_weights @ values

    logits, loss = model(input_tensor, target_tensor)

    print("Input form:")
    print(input_tensor.shape)
    print()

    print("First example as text:")
    print(repr(decode(input_tensor[0].tolist(), id_to_char)))
    print()

    print("Form embeddings before attention:")
    print(embeddings.shape)
    print()

    print("Key form:")
    print(keys.shape)
    print()

    print("Form queries:")
    print(queries.shape)
    print()

    print("Form values:")
    print(values.shape)
    print()

    print("Attention scores form:")
    print(attention_scores.shape)
    print()

    print("Maschera causale:")
    print(causal_mask)
    print()

    print("Attention weights form:")
    print(attention_weights.shape)
    print()

    print("Attention weights of the first example:")
    print(attention_weights[0])
    print()

    print("Sum of each attention-weight row for the first example:")
    print(attention_weights[0].sum(dim=-1))
    print()

    print("Form embeddings after attention:")
    print(attended_embeddings.shape)
    print()

    print("Logits form:")
    print(logits.shape)
    print()

    print("Initial loss:")
    print(loss.item())


if __name__ == "__main__":
    main()
