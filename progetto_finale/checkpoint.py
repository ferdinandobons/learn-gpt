"""
Differenza rispetto ai file precedenti:
- Prima il checkpoint salvava il vocabolario a caratteri `char_to_id` e
  `id_to_char`.
- Qui salva la configurazione del tokenizer BPE, perché il vocabolario GPT-2 è
  fisso e non va ricostruito dal testo.
- Il checkpoint ora può contenere anche configurazione di training e migliore
  validation loss, così il training può ripartire in modo più completo.

Scopo del file:
- Salvare pesi del modello, stato dell'optimizer, configurazione e tokenizer.
- Ricaricare un checkpoint su CPU, CUDA o MPS.
- Conservare abbastanza stato per fare resume del training.
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
