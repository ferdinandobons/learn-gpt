"""
Differenza rispetto al file precedente:
- Prima abbiamo aggiunto le ultime ottimizzazioni in lezioni separate.
- Qui ricomponiamo tutto in una versione finale pulita, allineata a
  `progetto_finale/`.

Scopo del file:
- Verificare che i moduli finali funzionino insieme sul dataset grande.
- Confermare che anche `prepare_data.py` fa parte del progetto finale pulito.
- Produrre un checkpoint dimostrativo usando batch letti da memmap.
- Generare testo dal checkpoint con lo stesso tokenizer BPE del training.
- Essere il corrispettivo didattico pulito del progetto finale.
"""

from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed" / "fineweb_edu"
CHECKPOINT_PATH = Path("/private/tmp/learngpt_final_project/best_checkpoint.pt")

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_42.batching import load_training_and_validation_data
from studio.snapshot.lezione_42.config import GenerationConfig, ModelConfig, TrainingConfig
from studio.snapshot.lezione_42.device import get_default_device
from studio.snapshot.lezione_42.generate import generate_text_from_checkpoint
from studio.snapshot.lezione_42.model import LanguageModel
from studio.snapshot.lezione_42.tokenizer import DEFAULT_ENCODING_NAME, get_vocabulary_size
from studio.snapshot.lezione_42.training import (
    configure_optimizer,
    maybe_compile_model,
    train_model,
)


def main():
    torch.manual_seed(42)

    device = get_default_device()
    training_data, validation_data = load_training_and_validation_data(DATA_DIR)

    tokenizer_config = {"encoding_name": DEFAULT_ENCODING_NAME}
    model_config = ModelConfig(
        vocabulary_size=get_vocabulary_size(DEFAULT_ENCODING_NAME),
    )
    training_config = TrainingConfig()
    generation_config = GenerationConfig()

    model = LanguageModel(**model_config.to_model_kwargs()).to(device)
    model = maybe_compile_model(
        model=model,
        compile_model=training_config.compile_model,
    )
    optimizer = configure_optimizer(
        model=model,
        learning_rate=training_config.base_learning_rate,
        weight_decay=training_config.weight_decay,
        device=device,
    )

    history, best_checkpoint_path = train_model(
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
        tokenizer_config=tokenizer_config,
        base_learning_rate=training_config.base_learning_rate,
        min_learning_rate=training_config.min_learning_rate,
        warmup_steps=training_config.warmup_steps,
        decay_steps=training_config.decay_steps,
        gradient_clip=training_config.gradient_clip,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        training_config=training_config.to_checkpoint_dict(),
        mixed_precision=training_config.mixed_precision,
        precision_dtype=training_config.precision_dtype,
        device=device,
    )

    best_validation = min(item["validation"] for item in history)

    print("Device usato:")
    print(device)
    print()

    print("Token disponibili:")
    print("training:", len(training_data))
    print("validation:", len(validation_data))
    print()

    print("Moduli finali verificati:")
    print(
        "config.py, prepare_data.py, tokenizer.py, batching.py, device.py, "
        "model.py, training.py, checkpoint.py, generate.py"
    )
    print()

    print("Miglior checkpoint:")
    print(best_checkpoint_path)
    print()

    print("Migliore validation loss osservata:")
    print(round(best_validation, 4))
    print()

    torch.manual_seed(123)
    generated_text, checkpoint = generate_text_from_checkpoint(
        checkpoint_path=best_checkpoint_path,
        prompt_text=generation_config.prompt_text,
        max_new_tokens=generation_config.generated_tokens,
        temperature=generation_config.temperature,
        top_k=generation_config.top_k,
        device=device,
        compile_model=training_config.compile_model,
    )

    print("Step del checkpoint caricato:")
    print(checkpoint["step"])
    print()

    print("Testo generato:")
    print(repr(generated_text))


if __name__ == "__main__":
    main()
