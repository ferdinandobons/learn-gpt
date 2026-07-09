"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 24 of the guided LearnGPT path.

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

from study.snapshots.lesson_24.batching import create_batch
from study.snapshots.lesson_24.model import LanguageModel
from study.snapshots.lesson_24.tokenizer import create_vocabulary, decode, encode


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

    attention_input = model.attention_layer_norm(embeddings)
    attention_output, _ = model.multi_head_attention(attention_input)
    residual_after_attention = embeddings + attention_output

    feed_forward_input = model.feed_forward_layer_norm(residual_after_attention)
    feed_forward_hidden = model.feed_forward.expand(feed_forward_input)
    feed_forward_activated = model.feed_forward.activation(feed_forward_hidden)
    feed_forward_output = model.feed_forward.project(feed_forward_activated)
    residual_after_feed_forward = residual_after_attention + feed_forward_output

    manual_logits = model.output_head(residual_after_feed_forward)
    logits, loss = model(input_tensor, target_tensor)

    max_difference = (manual_logits - logits).abs().max()

    print("Input form:")
    print(input_tensor.shape)
    print()

    print("First example as text:")
    print(repr(decode(input_tensor[0].tolist(), id_to_char)))
    print()

    print("Form embeddings:")
    print(embeddings.shape)
    print()

    print("Form after residual attention:")
    print(residual_after_attention.shape)
    print()

    print("Feedforward input form:")
    print(feed_forward_input.shape)
    print()

    print("Form after before Linear of the feed-forward:")
    print(feed_forward_hidden.shape)
    print()

    print("Form after GELU:")
    print(feed_forward_activated.shape)
    print()

    print("Feed-forward output form:")
    print(feed_forward_output.shape)
    print()

    print("Form after residual connection of the feed-forward:")
    print(residual_after_feed_forward.shape)
    print()

    print("Logits form:")
    print(logits.shape)
    print()

    print("First token before the feed-forward layer:")
    print(feed_forward_input[0, 0])
    print()

    print("First 8 values ​​of the first token after first Linear:")
    print(feed_forward_hidden[0, 0, :8])
    print()

    print("Top 8 values ​​of the first token after GELU:")
    print(feed_forward_activated[0, 0, :8])
    print()

    print("First token output from the feed-forward layer:")
    print(feed_forward_output[0, 0])
    print()

    print("Maximum difference between manual and forward model calculation:")
    print(max_difference.item())
    print()

    print("Initial loss:")
    print(loss.item())


if __name__ == "__main__":
    main()
