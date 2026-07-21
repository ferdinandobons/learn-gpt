"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 25 of the guided LearnGPT path.

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
DATASET_PATH = PROJECT_DIR / "data" / "study_sample.txt"

sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_25.batching import create_batch
from study.snapshots.lesson_25.model import LanguageModel
from study.snapshots.lesson_25.tokenizer import create_vocabulary, decode, encode


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

    positions = torch.arange(input_tensor.shape[1])
    token_embeddings = model.token_embedding_table(input_tensor)
    position_embeddings = model.position_embedding_table(positions)
    embeddings = token_embeddings + position_embeddings

    block = model.transformer_block

    attention_input = block.attention_layer_norm(embeddings)
    attention_output, _ = block.multi_head_attention(attention_input)
    residual_after_attention = embeddings + attention_output

    feed_forward_input = block.feed_forward_layer_norm(residual_after_attention)
    feed_forward_output = block.feed_forward(feed_forward_input)
    manual_block_output = residual_after_attention + feed_forward_output

    block_output = block(embeddings)
    manual_logits = model.output_head(manual_block_output)
    logits, loss = model(input_tensor, target_tensor)

    block_difference = (manual_block_output - block_output).abs().max()
    logits_difference = (manual_logits - logits).abs().max()

    print("Input shape:")
    print(input_tensor.shape)
    print()

    print("First example as text:")
    print(repr(decode(input_tensor[0].tolist(), id_to_char)))
    print()

    print("Shape embeddings as input to the TransformerBlock:")
    print(embeddings.shape)
    print()

    print("attention_input shape:")
    print(attention_input.shape)
    print()

    print("attention_output shape:")
    print(attention_output.shape)
    print()

    print("Shape after residual attention:")
    print(residual_after_attention.shape)
    print()

    print("feed_forward_input shape:")
    print(feed_forward_input.shape)
    print()

    print("feed_forward_output shape:")
    print(feed_forward_output.shape)
    print()

    print("Output shape of the TransformerBlock:")
    print(block_output.shape)
    print()

    print("Logits shape:")
    print(logits.shape)
    print()

    print("Maximum difference between manual calculation and TransformerBlock:")
    print(block_difference.item())
    print()

    print("Maximum difference between manual and forward logits of the model:")
    print(logits_difference.item())
    print()

    print("Initial loss:")
    print(loss.item())


if __name__ == "__main__":
    main()
