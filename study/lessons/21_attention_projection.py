"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 21 of the guided LearnGPT path.

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

from study.snapshots.lesson_21.batching import create_batch
from study.snapshots.lesson_21.model import LanguageModel
from study.snapshots.lesson_21.tokenizer import create_vocabulary, decode, encode


CONTEXT_SIZE = 8
BATCH_SIZE = 4
EMBEDDING_SIZE = 16
NUM_HEADS = 4
HEAD_SIZE = EMBEDDING_SIZE // NUM_HEADS


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
        num_heads=NUM_HEADS,
    )

    positions = torch.arange(CONTEXT_SIZE)
    token_embeddings = model.token_embedding_table(input_tensor)
    position_embeddings = model.position_embedding_table(positions)
    embeddings = token_embeddings + position_embeddings

    head_outputs = []

    for head in model.multi_head_attention.heads:
        head_output, _ = head(embeddings)
        head_outputs.append(head_output)

    concatenated_embeddings = torch.cat(head_outputs, dim=-1)
    projected_embeddings = model.multi_head_attention.output_projection(
        concatenated_embeddings
    )
    model_attention_output, _ = model.multi_head_attention(embeddings)

    logits, loss = model(input_tensor, target_tensor)

    max_difference = (projected_embeddings - model_attention_output).abs().max()

    print("Input form:")
    print(input_tensor.shape)
    print()

    print("First example as text:")
    print(repr(decode(input_tensor[0].tolist(), id_to_char)))
    print()

    print("Form embeddings before multi-head attention:")
    print(embeddings.shape)
    print()

    print("Output form of each head:")
    for index, head_output in enumerate(head_outputs):
        print(f"head {index}: {head_output.shape}")
    print()

    print("Form after head concatenation:")
    print(concatenated_embeddings.shape)
    print()

    print("Form after output_projection:")
    print(projected_embeddings.shape)
    print()

    print("Form logits after output_head:")
    print(logits.shape)
    print()

    print("First token after concatenation:")
    print(concatenated_embeddings[0, 0])
    print()

    print("First token after output_projection:")
    print(projected_embeddings[0, 0])
    print()

    print("Maximum difference between manual and forward module calculation:")
    print(max_difference.item())
    print()

    print("Initial loss:")
    print(loss.item())


if __name__ == "__main__":
    main()
