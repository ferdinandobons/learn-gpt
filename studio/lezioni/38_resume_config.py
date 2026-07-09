"""
Differenza rispetto al file precedente:
- Prima i parametri erano passati uno per uno allo script.
- Qui usiamo dataclass di configurazione e facciamo resume da checkpoint.

Scopo del file:
- Mostrare una configurazione di training più ordinata.
- Salvare nel checkpoint anche configurazione e migliore validation loss.
- Verificare che un secondo training possa ripartire dallo step salvato.
"""

from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = Path("/private/tmp/learngpt_lesson_39/checkpoint.pt")
sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_38.checkpoint import load_checkpoint
from studio.snapshot.lezione_38.config import ModelConfig, TrainingConfig
from studio.snapshot.lezione_38.model import LanguageModel
from studio.snapshot.lezione_38.training import configure_optimizer, train_model


VOCABULARY_SIZE = 100


def build_data():
    training_data = (np.arange(512, dtype=np.uint16) % VOCABULARY_SIZE).astype(np.uint16)
    validation_data = (np.arange(128, dtype=np.uint16) % VOCABULARY_SIZE).astype(np.uint16)

    return training_data, validation_data


def run_training(model_config, training_config, resume_checkpoint_path=None):
    training_data, validation_data = build_data()
    model = LanguageModel(**model_config.to_model_kwargs())
    optimizer = configure_optimizer(
        model=model,
        learning_rate=training_config.base_learning_rate,
        weight_decay=training_config.weight_decay,
    )

    return train_model(
        model=model,
        optimizer=optimizer,
        training_data=training_data,
        validation_data=validation_data,
        batch_size=training_config.batch_size,
        context_size=model_config.context_size,
        training_steps=training_config.training_steps,
        eval_interval=training_config.eval_interval,
        eval_batches=training_config.eval_batches,
        checkpoint_path=CHECKPOINT_PATH,
        model_config=model_config.to_checkpoint_dict(),
        tokenizer_config={"encoding_name": "demo"},
        base_learning_rate=training_config.base_learning_rate,
        min_learning_rate=training_config.min_learning_rate,
        warmup_steps=training_config.warmup_steps,
        decay_steps=training_config.decay_steps,
        gradient_clip=training_config.gradient_clip,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        resume_checkpoint_path=resume_checkpoint_path,
        training_config=training_config.to_checkpoint_dict(),
        device=torch.device("cpu"),
    )


def main():
    torch.manual_seed(42)
    model_config = ModelConfig(vocabulary_size=VOCABULARY_SIZE)
    first_training_config = TrainingConfig(training_steps=1)
    _, first_checkpoint_path = run_training(
        model_config=model_config,
        training_config=first_training_config,
    )

    resumed_training_config = TrainingConfig(training_steps=2, resume_from_checkpoint=True)
    history, resumed_checkpoint_path = run_training(
        model_config=model_config,
        training_config=resumed_training_config,
        resume_checkpoint_path=first_checkpoint_path,
    )

    resumed_model = LanguageModel(**model_config.to_model_kwargs())
    checkpoint = load_checkpoint(
        checkpoint_path=resumed_checkpoint_path,
        model=resumed_model,
        device=torch.device("cpu"),
    )

    print("Primo checkpoint:")
    print(first_checkpoint_path)
    print("Checkpoint dopo resume:")
    print(resumed_checkpoint_path)
    print("Step salvato:")
    print(checkpoint["step"])
    print("History del secondo run:")
    print(history)


if __name__ == "__main__":
    main()
