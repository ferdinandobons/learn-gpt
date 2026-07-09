"""
Differenza rispetto al file precedente:
- Prima creavamo un checkpoint dimostrativo dentro lo script di generazione.
- Qui usiamo una funzione `train_model` dedicata e salviamo il checkpoint solo
  quando la validation loss migliora.

Scopo del file:
- Eseguire un training breve ma completo.
- Vedere la storia delle valutazioni training/validation.
- Generare testo partendo dal miglior checkpoint salvato.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"
CHECKPOINT_PATH = Path("/private/tmp/learngpt_lesson_33/best_checkpoint.pt")

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_33.generate import generate_text_from_checkpoint
from studio.snapshot.lezione_33.model import LanguageModel
from studio.snapshot.lezione_33.tokenizer import create_vocabulary, encode
from studio.snapshot.lezione_33.training import train_model


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

    print("Valutazioni durante il training:")
    for item in history:
        print(
            f"step={item['step']} "
            f"training={item['training']:.4f} "
            f"validation={item['validation']:.4f}"
        )
    print()

    print("Miglior checkpoint:")
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

    print("Step del checkpoint caricato:")
    print(checkpoint["step"])
    print()

    print("Testo generato:")
    print(repr(generated_text))


if __name__ == "__main__":
    main()
