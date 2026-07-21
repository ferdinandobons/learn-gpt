"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 33 of the guided LearnGPT path.

File purpose:
- Run the lesson example in a reproducible way.
- Print the relevant intermediate values, tensor shapes, losses, or generated
  text for inspection.
"""

from pathlib import Path
import random
import sys
import tempfile

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "study_sample.txt"
CHECKPOINT_PATH = Path(tempfile.gettempdir()) / "learngpt_lesson_33" / "best_checkpoint.pt"

sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_33.generate import generate_text_from_checkpoint
from study.snapshots.lesson_33.model import LanguageModel
from study.snapshots.lesson_33.tokenizer import create_vocabulary, encode
from study.snapshots.lesson_33.training import train_model


CONTEXT_SIZE = 8
BATCH_SIZE = 4
EMBEDDING_SIZE = 16
NUM_HEADS = 4
HEAD_SIZE = EMBEDDING_SIZE // NUM_HEADS
NUM_TRANSFORMER_BLOCKS = 3
LEARNING_RATE = 0.003
TRAINING_STEPS = 30
EVAL_INTERVAL = 10
EVAL_BATCHES = 3
PROMPT_TEXT = "\n"
GENERATED_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 20


def create_model_config(vocabulary_size):
    return {
        "vocabulary_size": vocabulary_size,
        "context_size": CONTEXT_SIZE,
        "embedding_size": EMBEDDING_SIZE,
        "head_size": HEAD_SIZE,
        "num_heads": NUM_HEADS,
        "num_transformer_blocks": NUM_TRANSFORMER_BLOCKS,
    }


def main():
    random.seed(42)
    torch.manual_seed(42)

    text = DATASET_PATH.read_text(encoding="utf-8")
    char_to_id, id_to_char = create_vocabulary(text)
    token_ids = encode(text, char_to_id)

    split_index = int(len(token_ids) * 0.9)
    training_data = token_ids[:split_index]
    validation_data = token_ids[split_index:]

    model_config = create_model_config(vocabulary_size=len(char_to_id))
    model = LanguageModel(**model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    history, best_checkpoint_path = train_model(
        model=model,
        optimizer=optimizer,
        training_data=training_data,
        validation_data=validation_data,
        batch_size=BATCH_SIZE,
        context_size=CONTEXT_SIZE,
        training_steps=TRAINING_STEPS,
        eval_interval=EVAL_INTERVAL,
        eval_batches=EVAL_BATCHES,
        checkpoint_path=CHECKPOINT_PATH,
        model_config=model_config,
        char_to_id=char_to_id,
        id_to_char=id_to_char,
    )

    print("Evaluations during training:")
    for item in history:
        print(
            f"step={item['step']} "
            f"training={item['training']:.4f} "
            f"validation={item['validation']:.4f}"
        )
    print()

    print("Best checkpoint:")
    print(best_checkpoint_path)
    print()

    torch.manual_seed(123)
    generated_text, checkpoint = generate_text_from_checkpoint(
        checkpoint_path=best_checkpoint_path,
        prompt_text=PROMPT_TEXT,
        max_new_tokens=GENERATED_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )

    print("Loaded checkpoint step:")
    print(checkpoint["step"])
    print()

    print("Generated text:")
    print(repr(generated_text))


if __name__ == "__main__":
    main()
