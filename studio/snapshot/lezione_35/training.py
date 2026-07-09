"""
Differenza rispetto ai file precedenti:
- Prima `train_model` usava un learning rate fisso deciso nello script.
- Qui aggiungiamo configurazione dell'optimizer, learning rate schedule e
  gradient clipping.

Scopo del file:
- Centralizzare le ottimizzazioni principali del training.
- Rendere espliciti warmup, decay e clipping prima del progetto finale.
- Mantenere il ciclo di training leggibile e riutilizzabile.
"""

import math

import torch

from .batching import create_batch
from .checkpoint import save_checkpoint


def configure_optimizer(model, learning_rate, weight_decay):
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def get_learning_rate(
    step,
    base_learning_rate,
    min_learning_rate,
    warmup_steps,
    decay_steps,
):
    if step < warmup_steps:
        return base_learning_rate * step / warmup_steps

    if step > decay_steps:
        return min_learning_rate

    decay_ratio = (step - warmup_steps) / (decay_steps - warmup_steps)
    cosine_coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_learning_rate + cosine_coefficient * (
        base_learning_rate - min_learning_rate
    )


def apply_learning_rate(optimizer, learning_rate):
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


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
    base_learning_rate,
    min_learning_rate,
    warmup_steps,
    decay_steps,
    gradient_clip,
):
    history = []
    best_validation_loss = math.inf
    best_checkpoint_path = None
    model.train()

    for step in range(1, training_steps + 1):
        learning_rate = get_learning_rate(
            step=step,
            base_learning_rate=base_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
        )
        apply_learning_rate(optimizer=optimizer, learning_rate=learning_rate)

        input_tensor, target_tensor = create_batch(
            data=training_data,
            batch_size=batch_size,
            context_size=context_size,
        )
        _, loss = model(input_tensor, target_tensor)

        optimizer.zero_grad()
        loss.backward()

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

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
                    "learning_rate": learning_rate,
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
