"""
Changes compared with the previous files:
- This module adds a checkpoint-independent signal for detecting a model that
  gives nearly the same next-token distribution to unrelated contexts.

File purpose:
- Measure whether next-token logits vary across fixed validation contexts.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional


@torch.no_grad()
def estimate_context_sensitivity(
    model,
    validation_data,
    context_size,
    num_contexts,
    device,
):
    if context_size < 1:
        raise ValueError("context_size must be at least 1.")
    if num_contexts < 2:
        raise ValueError("num_contexts must be at least 2.")
    if len(validation_data) <= context_size:
        raise ValueError("validation_data is too small for the requested context_size.")

    maximum_start = len(validation_data) - context_size - 1
    start_positions = np.linspace(
        0,
        maximum_start,
        num=num_contexts,
        dtype=np.int64,
    )
    input_tensor = torch.stack(
        [
            torch.from_numpy(
                np.asarray(
                    validation_data[start : start + context_size],
                    dtype=np.int64,
                ).copy()
            )
            for start in start_positions
        ]
    ).to(device)

    was_training = model.training
    model.eval()
    logits = model(input_tensor)[:, -1, :]
    log_probabilities = functional.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    mean_probability = probabilities.mean(dim=0)
    context_js_divergence = (
        probabilities
        * (log_probabilities - mean_probability.clamp_min(torch.finfo(probabilities.dtype).tiny).log())
    ).sum(dim=-1).mean()
    context_logit_std = logits.float().std(dim=0, correction=0).mean()

    if was_training:
        model.train()

    return {
        "context_js_divergence": float(context_js_divergence.cpu().item()),
        "context_logit_std": float(context_logit_std.cpu().item()),
    }
