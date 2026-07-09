"""
Changes compared with the previous files:
- This module is part of the project-code snapshot used by lesson 38.
- It keeps the lesson independent from future changes in `final_project`.

File purpose:
- Provide the code needed by the matching lesson script.
- Preserve a stable reference point for the course examples.
"""

from pathlib import Path

import torch


def save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    model_config,
    step,
    losses,
    tokenizer_config,
    training_config=None,
    best_validation_loss=None,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "step": step,
        "losses": losses,
        "tokenizer_config": tokenizer_config,
        "training_config": training_config,
        "best_validation_loss": best_validation_loss,
    }

    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, device=None):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
