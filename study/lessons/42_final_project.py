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
import tempfile

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
STUDY_DATA_PATH = PROJECT_DIR / "data" / "study_sample.txt"
CHECKPOINT_PATH = (
    Path(tempfile.gettempdir())
    / "learngpt_final_project"
    / "best_checkpoint.pt"
)

sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_42.config import GenerationConfig, ModelConfig, TrainingConfig
from study.snapshots.lesson_42.device import get_default_device
from study.snapshots.lesson_42.generate import generate_text_from_checkpoint
from study.snapshots.lesson_42.model import LanguageModel
from study.snapshots.lesson_42.tokenizer import (
    DEFAULT_ENCODING_NAME,
    encode,
    get_vocabulary_size,
)
from study.snapshots.lesson_42.training import (
    configure_optimizer,
    maybe_compile_model,
    train_model,
)


def main():
    torch.manual_seed(42)

    device = get_default_device()
    study_text = STUDY_DATA_PATH.read_text(encoding="utf-8")
    token_ids = np.asarray(
        encode(study_text, encoding_name=DEFAULT_ENCODING_NAME),
        dtype=np.uint16,
    )
    split_index = int(len(token_ids) * 0.9)
    training_data = token_ids[:split_index]
    validation_data = token_ids[split_index:]

    tokenizer_config = {"encoding_name": DEFAULT_ENCODING_NAME}
    model_config = ModelConfig(
        vocabulary_size=get_vocabulary_size(DEFAULT_ENCODING_NAME),
        output_chunk_size=32768,
    )
    training_config = TrainingConfig(
        max_grad_norm_before_clip=100.0,
        gradient_retry_attempts=3,
        context_sensitivity_contexts=2,
    )
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
        max_grad_norm_before_clip=training_config.max_grad_norm_before_clip,
        gradient_retry_attempts=training_config.gradient_retry_attempts,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        context_sensitivity_contexts=training_config.context_sensitivity_contexts,
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
        "config.py, prepare_data.py, prepare_subset.py, tokenizer.py, "
        "batching.py, device.py, model.py, training.py, checkpoint.py, "
        "quality.py, generate.py"
    )
    print()

    print("Best checkpoint:")
    print(best_checkpoint_path)
    print()

    print("Best observed validation loss:")
    print(round(best_validation, 4))
    print()

    print("Output vocabulary chunk size:")
    print(model_config.output_chunk_size)
    print()

    print("Last raw gradient norm and retries:")
    print(round(history[-1]["grad_norm"], 4), history[-1]["gradient_retries"])
    print()

    print("Last context JS divergence:")
    print(f"{history[-1]['context_js_divergence']:.2e}")
    print()

    print("Last target-aware context loss gain:")
    print(f"{history[-1]['context_loss_gain']:+.4f}")
    print()

    generated_text, checkpoint = generate_text_from_checkpoint(
        checkpoint_path=best_checkpoint_path,
        prompt_text=generation_config.prompt_text,
        max_new_tokens=generation_config.generated_tokens,
        temperature=generation_config.temperature,
        top_k=generation_config.top_k,
        device=device,
        compile_model=training_config.compile_model,
        seed=generation_config.seed,
    )

    print("Loaded checkpoint step:")
    print(checkpoint["step"])
    print()

    print("Generated text:")
    print(repr(generated_text))


if __name__ == "__main__":
    main()
