"""
Differenza rispetto ai file precedenti:
- Qui manteniamo solo la singola head di self-attention e il modello che la usa.
- `LanguageModel` indica il modello della lezione corrente.

Scopo del file:
- Calcolare query, key, value e attention causale.
- Produrre logits dopo una singola head di attention.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn



class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_size, head_size, context_size):
        super().__init__()

        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_size, context_size)),
        )

    def forward(self, embeddings):
        current_context_size = embeddings.shape[1]

        keys = self.key(embeddings)
        queries = self.query(embeddings)
        values = self.value(embeddings)

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(keys.shape[-1])

        causal_mask = self.causal_mask[:current_context_size, :current_context_size]
        attention_scores = attention_scores.masked_fill(
            causal_mask == 0,
            float("-inf"),
        )

        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_embeddings = attention_weights @ values

        return attended_embeddings, attention_weights



class LanguageModel(nn.Module):
    def __init__(self, vocabulary_size, context_size, embedding_size, head_size):
        super().__init__()

        self.context_size = context_size
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
        )
        self.position_embedding_table = nn.Embedding(
            num_embeddings=context_size,
            embedding_dim=embedding_size,
        )
        self.attention_head = SelfAttentionHead(
            embedding_size=embedding_size,
            head_size=head_size,
            context_size=context_size,
        )
        self.output_head = nn.Linear(
            in_features=head_size,
            out_features=vocabulary_size,
        )

    def forward(self, input_ids, target_ids=None):
        current_context_size = input_ids.shape[1]

        if current_context_size > self.context_size:
            raise ValueError(
                f"Il contesto ricevuto contiene {current_context_size} token, "
                f"ma il modello supporta al massimo {self.context_size} token."
            )

        positions = torch.arange(current_context_size, device=input_ids.device)

        token_embeddings = self.token_embedding_table(input_ids)
        position_embeddings = self.position_embedding_table(positions)
        embeddings = token_embeddings + position_embeddings
        attended_embeddings, _ = self.attention_head(embeddings)
        logits = self.output_head(attended_embeddings)

        if target_ids is None:
            return logits

        batch_size, context_size, vocabulary_size = logits.shape

        logits_flat = logits.reshape(batch_size * context_size, vocabulary_size)
        target_ids_flat = target_ids.reshape(batch_size * context_size)

        loss = F.cross_entropy(logits_flat, target_ids_flat)

        return logits, loss

    def generate(self, input_ids, max_new_tokens):
        generated_ids = input_ids

        for _ in range(max_new_tokens):
            input_ids_limited = generated_ids[:, -self.context_size :]
            logits = self(input_ids_limited)
            last_token_logits = logits[:, -1, :]
            probabilities = F.softmax(last_token_logits, dim=-1)
            next_token_ids = torch.multinomial(probabilities, num_samples=1)
            generated_ids = torch.cat((generated_ids, next_token_ids), dim=1)

        return generated_ids
