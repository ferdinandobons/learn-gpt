"""
Differenza rispetto ai file precedenti:
- Prima la loss veniva letta su un solo batch dentro lo script di training.
- Qui aggiungiamo una funzione riutilizzabile per stimare la loss media su più
  batch di training e validation.

Scopo del file:
- Separare il training vero dalla valutazione della loss.
- Misurare train loss e validation loss senza aggiornare i pesi del modello.
"""

import torch

from .batching import create_batch


@torch.no_grad()
def estimate_loss(
    model,
    training_data,
    validation_data,
    batch_size,
    context_size,
    eval_batches,
):
    was_training = model.training
    model.eval()

    losses_by_split = {}
    data_by_split = {
        "training": training_data,
        "validation": validation_data,
    }

    for split_name, split_data in data_by_split.items():
        split_losses = []

        for _ in range(eval_batches):
            input_tensor, target_tensor = create_batch(
                data=split_data,
                batch_size=batch_size,
                context_size=context_size,
            )
            _, loss = model(input_tensor, target_tensor)
            split_losses.append(loss.item())

        losses_by_split[split_name] = sum(split_losses) / len(split_losses)

    if was_training:
        model.train()

    return losses_by_split
