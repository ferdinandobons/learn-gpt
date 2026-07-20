"""
Changes compared with the previous files:
- This module adds target-aware context diagnostics alongside distributional
  context-variation signals.

File purpose:
- Compare correct and shuffled validation contexts against real next tokens.
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
    target_tensor = torch.tensor(
        [
            int(validation_data[start + context_size])
            for start in start_positions
        ],
        dtype=torch.long,
        device=device,
    )

    was_training = model.training
    model.eval()
    logits = model(input_tensor)[:, -1, :].float()
    shuffled_logits = model(input_tensor.roll(shifts=1, dims=0))[:, -1, :].float()
    log_probabilities = functional.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    mean_probability = probabilities.mean(dim=0)
    context_js_divergence = (
        probabilities
        * (log_probabilities - mean_probability.clamp_min(torch.finfo(probabilities.dtype).tiny).log())
    ).sum(dim=-1).mean()
    context_logit_std = logits.float().std(dim=0, correction=0).mean()
    context_true_loss = functional.cross_entropy(logits, target_tensor)
    context_shuffled_loss = functional.cross_entropy(
        shuffled_logits,
        target_tensor,
    )
    context_loss_gain = context_shuffled_loss - context_true_loss

    if was_training:
        model.train()

    return {
        "context_js_divergence": float(context_js_divergence.cpu().item()),
        "context_logit_std": float(context_logit_std.cpu().item()),
        "context_true_loss": float(context_true_loss.cpu().item()),
        "context_shuffled_loss": float(context_shuffled_loss.cpu().item()),
        "context_loss_gain": float(context_loss_gain.cpu().item()),
    }
