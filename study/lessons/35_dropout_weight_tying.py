"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 35 of the guided LearnGPT path.

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
CHECKPOINT_PATH = Path("/private/tmp/learngpt_lesson_35/best_checkpoint.pt")

sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_35.batching import create_batch
from study.snapshots.lesson_35.generate import generate_text_from_checkpoint
from study.snapshots.lesson_35.model import LanguageModel
from study.snapshots.lesson_35.tokenizer import create_vocabulary, encode
from study.snapshots.lesson_35.training import configure_optimizer, train_model


CONTEXT_SIZE = 8
BATCH_SIZE = 4
EMBEDDING_SIZE = 16
NUM_HEADS = 4
HEAD_SIZE = EMBEDDING_SIZE // NUM_HEADS
NUM_TRANSFORMER_BLOCKS = 3
DROPOUT = 0.2
TIE_WEIGHTS = True
BASE_LEARNING_RATE = 0.003
MIN_LEARNING_RATE = 0.0003
WARMUP_STEPS = 5
DECAY_STEPS = 30
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
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
        "dropout": DROPOUT,
        "tie_weights": TIE_WEIGHTS,
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
    optimizer = configure_optimizer(
        model=model,
        learning_rate=BASE_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    input_tensor, _ = create_batch(
        data=training_data,
        batch_size=BATCH_SIZE,
        context_size=CONTEXT_SIZE,
    )

    model.train()
    torch.manual_seed(100)
    train_logits_a = model(input_tensor)
    torch.manual_seed(101)
    train_logits_b = model(input_tensor)
    train_difference = torch.sum(torch.abs(train_logits_a - train_logits_b)).item()

    model.eval()
    torch.manual_seed(100)
    eval_logits_a = model(input_tensor)
    torch.manual_seed(101)
    eval_logits_b = model(input_tensor)
    eval_difference = torch.sum(torch.abs(eval_logits_a - eval_logits_b)).item()

    print("Weight tying enabled:")
    print(model.output_head.weight is model.token_embedding_table.weight)
    print()

    print("Difference between two forward passes in training mode:")
    print(round(train_difference, 6))
    print()

    print("Difference between two forward passes in eval mode:")
    print(round(eval_difference, 6))
    print()

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
        base_learning_rate=BASE_LEARNING_RATE,
        min_learning_rate=MIN_LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=DECAY_STEPS,
        gradient_clip=GRADIENT_CLIP,
    )

    print("Final evaluation:")
    print(history[-1])
    print()

    torch.manual_seed(123)
    generated_text, _ = generate_text_from_checkpoint(
        checkpoint_path=best_checkpoint_path,
        prompt_text=PROMPT_TEXT,
        max_new_tokens=GENERATED_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )

    print("Generated text:")
    print(repr(generated_text))


if __name__ == "__main__":
    main()
