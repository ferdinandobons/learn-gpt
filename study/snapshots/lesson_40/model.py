"""
Changes compared with the previous files:
- This module is part of the project-code snapshot used by lesson 40.
- It keeps the lesson independent from future changes in `final_project`.

File purpose:
- Provide the code needed by the matching lesson script.
- Preserve a stable reference point for the course examples.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class SelfAttentionHead(nn.Module):
    def __init__(
        self,
        embedding_size,
        head_size,
        context_size,
        dropout,
        use_scaled_dot_product_attention=False,
    ):
        super().__init__()

        self.dropout = dropout
        self.use_scaled_dot_product_attention = use_scaled_dot_product_attention
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_size, context_size)),
        )

    def forward(self, embeddings):
        current_context_size = embeddings.shape[1]

        keys = self.key(embeddings)
        queries = self.query(embeddings)
        values = self.value(embeddings)

        if self.use_scaled_dot_product_attention:
            attended_embeddings = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )

            return attended_embeddings, None

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(keys.shape[-1])

        causal_mask = self.causal_mask[:current_context_size, :current_context_size]
        attention_scores = attention_scores.masked_fill(
            causal_mask == 0,
            float("-inf"),
        )

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        attended_embeddings = attention_weights @ values

        return attended_embeddings, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_size,
        head_size,
        context_size,
        num_heads,
        dropout,
        use_scaled_dot_product_attention=False,
    ):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    embedding_size=embedding_size,
                    head_size=head_size,
                    context_size=context_size,
                    dropout=dropout,
                    use_scaled_dot_product_attention=use_scaled_dot_product_attention,
                )
                for _ in range(num_heads)
            ]
        )
        self.output_projection = nn.Linear(
            in_features=num_heads * head_size,
            out_features=embedding_size,
        )
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        attended_outputs = []
        attention_weights_by_head = []

        for head in self.heads:
            attended_embeddings, attention_weights = head(embeddings)
            attended_outputs.append(attended_embeddings)
            attention_weights_by_head.append(attention_weights)

        concatenated_embeddings = torch.cat(attended_outputs, dim=-1)
        projected_embeddings = self.output_projection(concatenated_embeddings)
        projected_embeddings = self.output_dropout(projected_embeddings)

        return projected_embeddings, attention_weights_by_head


class FeedForward(nn.Module):
    def __init__(self, embedding_size, dropout):
        super().__init__()

        self.expand = nn.Linear(
            in_features=embedding_size,
            out_features=4 * embedding_size,
        )
        self.activation = nn.GELU()
        self.project = nn.Linear(
            in_features=4 * embedding_size,
            out_features=embedding_size,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        hidden = self.expand(embeddings)
        activated = self.activation(hidden)
        projected = self.project(activated)
        output = self.dropout(projected)

        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_size,
        head_size,
        context_size,
        num_heads,
        dropout,
        use_scaled_dot_product_attention=False,
    ):
        super().__init__()

        self.attention_layer_norm = nn.LayerNorm(
            normalized_shape=embedding_size,
        )
        self.multi_head_attention = MultiHeadAttention(
            embedding_size=embedding_size,
            head_size=head_size,
            context_size=context_size,
            num_heads=num_heads,
            dropout=dropout,
            use_scaled_dot_product_attention=use_scaled_dot_product_attention,
        )
        self.feed_forward_layer_norm = nn.LayerNorm(
            normalized_shape=embedding_size,
        )
        self.feed_forward = FeedForward(
            embedding_size=embedding_size,
            dropout=dropout,
        )

    def forward(self, embeddings):
        attention_input = self.attention_layer_norm(embeddings)
        attention_output, _ = self.multi_head_attention(attention_input)
        residual_after_attention = embeddings + attention_output

        feed_forward_input = self.feed_forward_layer_norm(residual_after_attention)
        feed_forward_output = self.feed_forward(feed_forward_input)
        residual_after_feed_forward = residual_after_attention + feed_forward_output

        return residual_after_feed_forward


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        context_size,
        embedding_size,
        head_size,
        num_heads,
        num_transformer_blocks,
        dropout=0.0,
        tie_weights=True,
        use_scaled_dot_product_attention=False,
    ):
        super().__init__()

        if num_heads * head_size != embedding_size:
            raise ValueError(
                "In this educational version, num_heads * head_size must "
                "be equal to embedding_size."
            )

        if num_transformer_blocks < 1:
            raise ValueError("num_transformer_blocks must be at least 1.")

        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0.")

        self.context_size = context_size
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
        )
        self.position_embedding_table = nn.Embedding(
            num_embeddings=context_size,
            embedding_dim=embedding_size,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_size=embedding_size,
                    head_size=head_size,
                    context_size=context_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_scaled_dot_product_attention=use_scaled_dot_product_attention,
                )
                for _ in range(num_transformer_blocks)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            normalized_shape=embedding_size,
        )
        self.output_head = nn.Linear(
            in_features=embedding_size,
            out_features=vocabulary_size,
        )

        if tie_weights:
            self.output_head.weight = self.token_embedding_table.weight

    def forward(self, input_ids, target_ids=None):
        current_context_size = input_ids.shape[1]

        if current_context_size > self.context_size:
            raise ValueError(
                f"The received context contains {current_context_size} token, "
                f"but the model supports at most {self.context_size} token."
            )

        positions = torch.arange(current_context_size, device=input_ids.device)

        token_embeddings = self.token_embedding_table(input_ids)
        position_embeddings = self.position_embedding_table(positions)

        block_output = token_embeddings + position_embeddings
        block_output = self.embedding_dropout(block_output)

        for transformer_block in self.transformer_blocks:
            block_output = transformer_block(block_output)

        if target_ids is None:
            block_output = self.final_layer_norm(block_output[:, [-1], :])
            logits = self.output_head(block_output)

            return logits

        block_output = self.final_layer_norm(block_output)
        logits = self.output_head(block_output)

        batch_size, context_size, vocabulary_size = logits.shape

        logits_flat = logits.reshape(batch_size * context_size, vocabulary_size)
        target_ids_flat = target_ids.reshape(batch_size * context_size)

        loss = F.cross_entropy(logits_flat, target_ids_flat)

        return logits, loss

    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0.")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than 0 when set.")

        generated_ids = input_ids

        for _ in range(max_new_tokens):
            input_ids_limited = generated_ids[:, -self.context_size :]
            logits = self(input_ids_limited)
            last_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_k = min(top_k, last_token_logits.shape[-1])
                top_values, _ = torch.topk(last_token_logits, top_k)
                minimum_top_value = top_values[:, [-1]]
                last_token_logits = last_token_logits.masked_fill(
                    last_token_logits < minimum_top_value,
                    float("-inf"),
                )

            probabilities = F.softmax(last_token_logits, dim=-1)
            next_token_ids = torch.multinomial(probabilities, num_samples=1)
            generated_ids = torch.cat((generated_ids, next_token_ids), dim=1)

        return generated_ids
