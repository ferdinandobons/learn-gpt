"""
Differenza rispetto ai file precedenti:
- Prima il training esisteva solo nella memoria del processo Python.
- Qui aggiungiamo funzioni per salvare e ricaricare un checkpoint.

Scopo del file:
- Salvare pesi del modello, stato dell'optimizer, configurazione e tokenizer.
- Ricaricare il checkpoint dentro un modello nuovo.
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
