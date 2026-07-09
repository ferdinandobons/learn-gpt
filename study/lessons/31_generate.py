"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 31 of the guided LearnGPT path.

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
CHECKPOINT_PATH = Path("/private/tmp/learngpt_lesson_31/checkpoint.pt")

sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_31.batching import create_batch
from study.snapshots.lesson_31.checkpoint import save_checkpoint
from study.snapshots.lesson_31.generate import generate_text_from_checkpoint
from study.snapshots.lesson_31.model import LanguageModel
from study.snapshots.lesson_31.tokenizer import create_vocabulary, encode
from study.snapshots.lesson_31.training import estimate_loss


CONTEXT_SIZE = 8
BATCH_SIZE = 4
EMBEDDING_SIZE = 16
NUM_HEADS = 4
HEAD_SIZE = EMBEDDING_SIZE // NUM_HEADS
NUM_TRANSFORMER_BLOCKS = 3
LEARNING_RATE = 0.003
TRAINING_STEPS = 20
EVAL_BATCHES = 3
PROMPT_TEXT = "\n"
GENERATED_TOKENS = 120


def create_model_config(vocabulary_size):
    return {
        "vocabulary_size": vocabulary_size,
        "context_size": CONTEXT_SIZE,
        "embedding_size": EMBEDDING_SIZE,
        "head_size": HEAD_SIZE,
        "num_heads": NUM_HEADS,
        "num_transformer_blocks": NUM_TRANSFORMER_BLOCKS,
    }


def create_demo_checkpoint():
    text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, id_to_char = create_vocabulary(text)
    vocabulary_size = len(char_to_id)
    token_ids = encode(text, char_to_id)

    split_index = int(len(token_ids) * 0.9)
    training_data = token_ids[:split_index]
    validation_data = token_ids[split_index:]

    model_config = create_model_config(vocabulary_size)
    model = LanguageModel(**model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    for _ in range(TRAINING_STEPS):
        input_tensor, target_tensor = create_batch(
            data=training_data,
            batch_size=BATCH_SIZE,
            context_size=CONTEXT_SIZE,
        )
        _, loss = model(input_tensor, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses = estimate_loss(
        model=model,
        training_data=training_data,
        validation_data=validation_data,
        batch_size=BATCH_SIZE,
        context_size=CONTEXT_SIZE,
        eval_batches=EVAL_BATCHES,
    )

    return save_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        step=TRAINING_STEPS,
        losses=losses,
        char_to_id=char_to_id,
        id_to_char=id_to_char,
    )


def main():
    random.seed(42)
    torch.manual_seed(42)

    checkpoint_path = create_demo_checkpoint()

    torch.manual_seed(123)
    generated_text, checkpoint = generate_text_from_checkpoint(
        checkpoint_path=checkpoint_path,
        prompt_text=PROMPT_TEXT,
        max_new_tokens=GENERATED_TOKENS,
    )

    print("Checkpoint usato per generare:")
    print(checkpoint_path)
    print()

    print("Step saved in the checkpoint:")
    print(checkpoint["step"])
    print()

    print("Prompt:")
    print(repr(PROMPT_TEXT))
    print()

    print("Generated text from checkpoint:")
    print(repr(generated_text))


if __name__ == "__main__":
    main()
