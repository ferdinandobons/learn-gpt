"""
Differenza rispetto al file precedente:
- Prima avevamo solo l'ottimizzazione interna della attention.
- Qui aggiungiamo i flag opzionali per `torch.compile` e mixed precision, e
  trattiamo DDP come concetto avanzato non obbligatorio.

Scopo del file:
- Verificare il report del device disponibile.
- Mostrare che compile e mixed precision sono spenti di default.
- Chiarire che DDP serve per training distribuito, non per lo smoke test locale.
"""

from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_41.config import ModelConfig, TrainingConfig
from studio.snapshot.lezione_41.device import get_default_device, get_device_report
from studio.snapshot.lezione_41.model import LanguageModel
from studio.snapshot.lezione_41.training import (
    configure_optimizer,
    get_autocast_context,
    maybe_compile_model,
)


VOCABULARY_SIZE = 100


def main():
    torch.manual_seed(42)
    device = get_default_device()
    model_config = ModelConfig(
        vocabulary_size=VOCABULARY_SIZE,
        use_scaled_dot_product_attention=True,
    )
    training_config = TrainingConfig(
        compile_model=False,
        mixed_precision=False,
    )

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

    input_ids = torch.randint(
        low=0,
        high=VOCABULARY_SIZE,
        size=(2, model_config.context_size),
        device=device,
    )
    target_ids = torch.randint(
        low=0,
        high=VOCABULARY_SIZE,
        size=(2, model_config.context_size),
        device=device,
    )

    with get_autocast_context(
        device=device,
        mixed_precision=training_config.mixed_precision,
        precision_dtype=training_config.precision_dtype,
    ):
        logits, loss = model(input_ids, target_ids)

    print("Device report:")
    print(get_device_report())
    print()

    print("Flag performance:")
    print("compile_model:", training_config.compile_model)
    print("mixed_precision:", training_config.mixed_precision)
    print("precision_dtype:", training_config.precision_dtype)
    print("optimizer groups:", len(optimizer.param_groups))
    print()

    print("Forward verificato:")
    print("logits shape:", tuple(logits.shape))
    print("loss finita:", bool(loss.isfinite().item()))
    print()

    print("DDP:")
    print("concetto avanzato per training multi-processo; non viene avviato qui.")


if __name__ == "__main__":
    main()
