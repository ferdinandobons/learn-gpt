"""
Differenza rispetto ai file precedenti:
- Qui definiamo solo il modello usato dalla lezione 12.
- Il nome pubblico stabile è `LanguageModel`.

Scopo del file:
- Restituire logits partendo da token numerici.
- Non calcolare ancora la loss e non generare testo.
"""

from torch import nn


class LanguageModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=vocabulary_size,
        )

    def forward(self, input_ids):
        logits = self.token_embedding_table(input_ids)

        return logits
