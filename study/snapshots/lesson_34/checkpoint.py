"""
Changes compared with the previous files:
- This module is part of the project-code snapshot used by lesson 34.
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
    char_to_id,
    id_to_char,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "step": step,
        "losses": losses,
        "char_to_id": char_to_id,
        "id_to_char": id_to_char,
    }

    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
