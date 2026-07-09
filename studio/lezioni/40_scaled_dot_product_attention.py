"""
Differenza rispetto al file precedente:
- Prima la self-attention era sempre calcolata manualmente con matrice dei
  punteggi, maschera causale e softmax.
- Qui possiamo attivare `scaled_dot_product_attention` di PyTorch.

Scopo del file:
- Tenere la versione manuale come default didattico.
- Verificare che l'attention ottimizzata mantenga la stessa shape dei logits.
- Capire che questa è un'ottimizzazione interna, non un cambio di obiettivo.
"""

from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_40.config import ModelConfig
from studio.snapshot.lezione_40.model import LanguageModel


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
    print("Loss manuale finita:")
    print(bool(manual_loss.isfinite().item()))
    print("Loss ottimizzata finita:")
    print(bool(optimized_loss.isfinite().item()))


if __name__ == "__main__":
    main()
