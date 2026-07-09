"""
Differenza rispetto ai file precedenti:
- Prima il training lavorava su liste di token a caratteri già caricate in RAM.
- Qui il training lavora su FineWeb-Edu processato in `train.bin` e `val.bin`,
  usando batch da memmap e device CPU/CUDA/MPS.
- L'optimizer ora separa i parametri con weight decay dai parametri senza
  weight decay.
- Il training loop ora supporta gradient accumulation.
- Il training loop ora può ripartire da un checkpoint completo.

Scopo del file:
- Centralizzare optimizer, learning rate schedule, stima loss e training loop.
- Gestire dataset grandi senza caricarli interamente in memoria.
- Salvare il miglior checkpoint con configurazione modello e tokenizer BPE.
- Applicare AdamW in modo più simile a nanoGPT.
- Simulare batch effettivi più grandi sommando gradienti di più micro-batch.
- Riprendere `model_state_dict`, `optimizer_state_dict`, step e migliore
  validation loss quando serve.
"""

import inspect
import math

import torch

from .batching import create_batch
from .checkpoint import load_checkpoint, save_checkpoint
from .device import get_default_device


def configure_optimizer(
    model,
    learning_rate,
    weight_decay,
    betas=(0.9, 0.95),
    device=None,
):
    parameter_dict = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    decay_parameters = [
        parameter
        for parameter in parameter_dict.values()
        if parameter.dim() >= 2
    ]
    no_decay_parameters = [
        parameter
        for parameter in parameter_dict.values()
        if parameter.dim() < 2
    ]
    optimizer_groups = [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]

    device = device or get_default_device()
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and torch.device(device).type == "cuda"
    extra_args = {"fused": True} if use_fused else {}

    return torch.optim.AdamW(
        optimizer_groups,
        lr=learning_rate,
        betas=betas,
        **extra_args,
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
    device=None,
):
    device = device or get_default_device()
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
                device=device,
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
    tokenizer_config,
    base_learning_rate,
    min_learning_rate,
    warmup_steps,
    decay_steps,
    gradient_clip,
    gradient_accumulation_steps=1,
    resume_checkpoint_path=None,
    training_config=None,
    device=None,
):
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps deve essere almeno 1.")

    device = device or get_default_device()
    model.to(device)
    model.train()

    history = []
    best_validation_loss = math.inf
    best_checkpoint_path = None
    start_step = 1

    if resume_checkpoint_path is not None:
        checkpoint = load_checkpoint(
            checkpoint_path=resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        start_step = int(checkpoint.get("step", 0)) + 1
        best_validation_loss = checkpoint.get("best_validation_loss") or math.inf
        best_checkpoint_path = resume_checkpoint_path

    for step in range(start_step, training_steps + 1):
        learning_rate = get_learning_rate(
            step=step,
            base_learning_rate=base_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
        )
        apply_learning_rate(optimizer=optimizer, learning_rate=learning_rate)

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(gradient_accumulation_steps):
            input_tensor, target_tensor = create_batch(
                data=training_data,
                batch_size=batch_size,
                context_size=context_size,
                device=device,
            )
            _, loss = model(input_tensor, target_tensor)
            loss = loss / gradient_accumulation_steps
            total_loss += loss.item()
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
                device=device,
            )
            history.append(
                {
                    "step": step,
                    "learning_rate": learning_rate,
                    "loss": total_loss,
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
                    tokenizer_config=tokenizer_config,
                    training_config=training_config,
                    best_validation_loss=best_validation_loss,
                )

    return history, best_checkpoint_path
