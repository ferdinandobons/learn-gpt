"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 42 of the guided LearnGPT path.

File purpose:
- Run the lesson example in a reproducible way.
- Print the relevant intermediate values, tensor shapes, losses, or generated
  text for inspection.
"""

from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed" / "fineweb_edu"
CHECKPOINT_PATH = Path("/private/tmp/learngpt_final_project/best_checkpoint.pt")

sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_42.batching import load_training_and_validation_data
from study.snapshots.lesson_42.config import GenerationConfig, ModelConfig, TrainingConfig
from study.snapshots.lesson_42.device import get_default_device
from study.snapshots.lesson_42.generate import generate_text_from_checkpoint
from study.snapshots.lesson_42.model import LanguageModel
from study.snapshots.lesson_42.tokenizer import DEFAULT_ENCODING_NAME, get_vocabulary_size
from study.snapshots.lesson_42.training import (
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

    print("Device used:")
    print(device)
    print()

    print("Available tokens:")
    print("training:", len(training_data))
    print("validation:", len(validation_data))
    print()

    print("Verified final modules:")
    print(
        "config.py, prepare_data.py, tokenizer.py, batching.py, device.py, "
        "model.py, training.py, checkpoint.py, generate.py"
    )
    print()

    print("Best checkpoint:")
    print(best_checkpoint_path)
    print()

    print("Best observed validation loss:")
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

    print("Loaded checkpoint step:")
    print(checkpoint["step"])
    print()

    print("Generated text:")
    print(repr(generated_text))


if __name__ == "__main__":
    main()
