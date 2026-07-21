"""
Changes compared with the previous files:
- This module belongs to the cleaned final LearnGPT project.
- It uses the English public project layout and the FineWeb-Edu/BPE pipeline.

File purpose:
- Define the final decoder-only Transformer language model.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


GPT_INITIALIZATION_STD = 0.02


class LayerNorm(nn.Module):
    """LayerNorm with nanoGPT-style optional bias support."""

    def __init__(self, embedding_size, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embedding_size))
        self.bias = nn.Parameter(torch.zeros(embedding_size)) if bias else None

    def forward(self, embeddings):
        return F.layer_norm(
            embeddings,
            self.weight.shape,
            self.weight,
            self.bias,
            1e-5,
        )


class SelfAttentionHead(nn.Module):
    def __init__(
        self,
        embedding_size,
        head_size,
        context_size,
        dropout,
        bias=False,
        use_scaled_dot_product_attention=False,
    ):
        super().__init__()

        self.dropout = dropout
        self.use_scaled_dot_product_attention = use_scaled_dot_product_attention
        # GPT-style Q/K/V projections stay bias-free. The broader bias option
        # controls modules whose bias-free MPS kernels must pass the startup
        # CPU-parity check before they are used for real training.
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
        bias=False,
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
                    bias=bias,
                    use_scaled_dot_product_attention=use_scaled_dot_product_attention,
                )
                for _ in range(num_heads)
            ]
        )
        self.output_projection = nn.Linear(
            in_features=num_heads * head_size,
            out_features=embedding_size,
            bias=bias,
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


class FusedMultiHeadAttention(nn.Module):
    """Compute all attention heads with one QKV projection and one SDPA call."""

    def __init__(
        self,
        embedding_size,
        head_size,
        context_size,
        num_heads,
        dropout,
        bias=False,
        use_scaled_dot_product_attention=False,
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scaled_dot_product_attention = use_scaled_dot_product_attention
        self.query_key_value = nn.Linear(
            embedding_size,
            3 * embedding_size,
            bias=False,
        )
        self.output_projection = nn.Linear(
            embedding_size,
            embedding_size,
            bias=bias,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_size, context_size)),
        )

    def _split_heads(self, projection):
        batch_size, context_size, _ = projection.shape
        return projection.view(
            batch_size,
            context_size,
            self.num_heads,
            self.head_size,
        ).transpose(1, 2)

    def forward(self, embeddings):
        current_context_size = embeddings.shape[1]
        query_key_value = self.query_key_value(embeddings)
        queries, keys, values = query_key_value.split(
            self.embedding_size,
            dim=-1,
        )
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        if self.use_scaled_dot_product_attention:
            attended_embeddings = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            attention_weights = None
        else:
            attention_scores = queries @ keys.transpose(-2, -1)
            attention_scores = attention_scores / math.sqrt(self.head_size)
            causal_mask = self.causal_mask[
                :current_context_size,
                :current_context_size,
            ]
            attention_scores = attention_scores.masked_fill(
                causal_mask == 0,
                float("-inf"),
            )
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.attention_dropout(attention_weights)
            attended_embeddings = attention_weights @ values

        batch_size = embeddings.shape[0]
        attended_embeddings = (
            attended_embeddings.transpose(1, 2)
            .contiguous()
            .view(batch_size, current_context_size, self.embedding_size)
        )
        projected_embeddings = self.output_projection(attended_embeddings)
        projected_embeddings = self.output_dropout(projected_embeddings)

        return projected_embeddings, attention_weights


class FeedForward(nn.Module):
    def __init__(self, embedding_size, dropout, bias=False):
        super().__init__()

        self.expand = nn.Linear(
            in_features=embedding_size,
            out_features=4 * embedding_size,
            bias=bias,
        )
        self.activation = nn.GELU()
        self.project = nn.Linear(
            in_features=4 * embedding_size,
            out_features=embedding_size,
            bias=bias,
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
        bias=False,
        use_scaled_dot_product_attention=False,
        fused_attention=False,
    ):
        super().__init__()

        self.attention_layer_norm = LayerNorm(embedding_size, bias=bias)
        attention_class = (
            FusedMultiHeadAttention if fused_attention else MultiHeadAttention
        )
        self.multi_head_attention = attention_class(
            embedding_size=embedding_size,
            head_size=head_size,
            context_size=context_size,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            use_scaled_dot_product_attention=use_scaled_dot_product_attention,
        )
        self.feed_forward_layer_norm = LayerNorm(embedding_size, bias=bias)
        self.feed_forward = FeedForward(
            embedding_size=embedding_size,
            dropout=dropout,
            bias=bias,
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
        bias=False,
        tie_weights=True,
        use_scaled_dot_product_attention=False,
        fused_attention=False,
        output_chunk_size=32768,
    ):
        super().__init__()

        if num_heads * head_size != embedding_size:
            raise ValueError(
                "In this educational version, num_heads * head_size must"
                "be equal to embedding_size."
            )

        if num_transformer_blocks < 1:
            raise ValueError("num_transformer_blocks must be at least 1.")

        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0.")

        if output_chunk_size < 0:
            raise ValueError("output_chunk_size cannot be negative.")

        self.context_size = context_size
        self.output_chunk_size = output_chunk_size
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
                    bias=bias,
                    use_scaled_dot_product_attention=use_scaled_dot_product_attention,
                    fused_attention=fused_attention,
                )
                for _ in range(num_transformer_blocks)
            ]
        )

        self.final_layer_norm = LayerNorm(embedding_size, bias=bias)
        self.output_head = nn.Linear(
            in_features=embedding_size,
            out_features=vocabulary_size,
            bias=bias,
        )

        self.apply(self._initialize_weights)
        residual_projection_std = GPT_INITIALIZATION_STD / math.sqrt(
            2 * num_transformer_blocks
        )
        for transformer_block in self.transformer_blocks:
            nn.init.normal_(
                transformer_block.multi_head_attention.output_projection.weight,
                mean=0.0,
                std=residual_projection_std,
            )
            nn.init.normal_(
                transformer_block.feed_forward.project.weight,
                mean=0.0,
                std=residual_projection_std,
            )

        if tie_weights:
            self.output_head.weight = self.token_embedding_table.weight

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=GPT_INITIALIZATION_STD,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=GPT_INITIALIZATION_STD,
            )

    def project_to_vocabulary(self, embeddings):
        vocabulary_size = self.output_head.weight.shape[0]
        if (
            self.output_chunk_size == 0
            or self.output_chunk_size >= vocabulary_size
        ):
            return self.output_head(embeddings)

        projected_chunks = []
        for start in range(0, vocabulary_size, self.output_chunk_size):
            end = min(start + self.output_chunk_size, vocabulary_size)
            bias = (
                None
                if self.output_head.bias is None
                else self.output_head.bias[start:end]
            )
            projected_chunks.append(
                F.linear(
                    embeddings,
                    self.output_head.weight[start:end],
                    bias,
                )
            )

        return torch.cat(projected_chunks, dim=-1)

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
            logits = self.project_to_vocabulary(block_output)

            return logits

        block_output = self.final_layer_norm(block_output)
        logits = self.project_to_vocabulary(block_output)

        batch_size, context_size, vocabulary_size = logits.shape

        logits_flat = logits.reshape(batch_size * context_size, vocabulary_size)
        target_ids_flat = target_ids.reshape(batch_size * context_size)

        loss = F.cross_entropy(logits_flat, target_ids_flat)

        return logits, loss

    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have the form [batch_size, context_size].")

        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one token.")

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")

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
