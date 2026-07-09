"""
Changes compared with the previous files:
- This module is part of the project-code snapshot used by lesson 33.
- It keeps the lesson independent from future changes in `final_project`.

File purpose:
- Provide the code needed by the matching lesson script.
- Preserve a stable reference point for the course examples.
"""

import math

import torch

from .batching import create_batch
from .checkpoint import save_checkpoint


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


def train_model(
    model,
    optimizer,
    training_data,
    validation_data,
    batch_size,
    context_size,
    training_steps,
    eval_interval,
    eval_batches,
    checkpoint_path,
    model_config,
    char_to_id,
    id_to_char,
):
    history = []
    best_validation_loss = math.inf
    best_checkpoint_path = None
    model.train()

    for step in range(1, training_steps + 1):
        input_tensor, target_tensor = create_batch(
            data=training_data,
            batch_size=batch_size,
            context_size=context_size,
        )
        _, loss = model(input_tensor, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        should_evaluate = step == 1 or step % eval_interval == 0

        if should_evaluate:
            losses = estimate_loss(
                model=model,
                training_data=training_data,
                validation_data=validation_data,
                batch_size=batch_size,
                context_size=context_size,
                eval_batches=eval_batches,
            )
            history.append(
                {
                    "step": step,
                    "training": losses["training"],
                    "validation": losses["validation"],
                }
            )

            if losses["validation"] < best_validation_loss:
                best_validation_loss = losses["validation"]
                best_checkpoint_path = save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    model_config=model_config,
                    step=step,
                    losses=losses,
                    char_to_id=char_to_id,
                    id_to_char=id_to_char,
                )

    return history, best_checkpoint_path
