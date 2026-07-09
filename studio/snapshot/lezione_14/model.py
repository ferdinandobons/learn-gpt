"""
Differenza rispetto ai file precedenti:
- Qui manteniamo solo il modello bigram necessario al primo training loop.
- Il nome pubblico stabile è `LanguageModel`.

Scopo del file:
- Restituire logits e loss.
- Esporre i parametri che l'optimizer aggiorna durante il training.
"""

import torch.nn.functional as F
from torch import nn


class LanguageModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=vocabulary_size,
        )

    def forward(self, input_ids, target_ids=None):
        logits = self.token_embedding_table(input_ids)

        if target_ids is None:
            return logits

        batch_size, context_size, vocabulary_size = logits.shape

        logits_flat = logits.reshape(batch_size * context_size, vocabulary_size)
        target_ids_flat = target_ids.reshape(batch_size * context_size)

        loss = F.cross_entropy(logits_flat, target_ids_flat)

        return logits, loss
