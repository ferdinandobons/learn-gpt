"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 40 of the guided LearnGPT path.

File purpose:
- Run the lesson example in a reproducible way.
- Print the relevant intermediate values, tensor shapes, losses, or generated
  text for inspection.
"""

from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR))

from study.snapshots.lesson_40.config import ModelConfig
from study.snapshots.lesson_40.model import LanguageModel


VOCABULARY_SIZE = 100


def main():
    torch.manual_seed(42)
    manual_config = ModelConfig(
        vocabulary_size=VOCABULARY_SIZE,
        use_scaled_dot_product_attention=False,
    )
    optimized_config = ModelConfig(
        vocabulary_size=VOCABULARY_SIZE,
        use_scaled_dot_product_attention=True,
    )

    manual_model = LanguageModel(**manual_config.to_model_kwargs())
    optimized_model = LanguageModel(**optimized_config.to_model_kwargs())

    input_ids = torch.randint(
        low=0,
        high=VOCABULARY_SIZE,
        size=(2, manual_config.context_size),
    )
    target_ids = torch.randint(
        low=0,
        high=VOCABULARY_SIZE,
        size=(2, manual_config.context_size),
    )

    manual_logits, manual_loss = manual_model(input_ids, target_ids)
    optimized_logits, optimized_loss = optimized_model(input_ids, target_ids)

    print("Manual logits shape:")
    print(tuple(manual_logits.shape))
    print("Optimized logits shape:")
    print(tuple(optimized_logits.shape))
    print("Manual loss is finite:")
    print(bool(manual_loss.isfinite().item()))
    print("Optimized loss is finite:")
    print(bool(optimized_loss.isfinite().item()))


if __name__ == "__main__":
    main()
