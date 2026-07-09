"""
Changes compared with the previous files:
- This module belongs to the cleaned final LearnGPT project.
- It uses the English public project layout and the FineWeb-Edu/BPE pipeline.

File purpose:
- Run optimizer setup, loss estimation, and training.
"""

from contextlib import nullcontext
import argparse
import inspect
import json
import math
from pathlib import Path
import time

import torch

from .batching import DEFAULT_DATA_DIR, create_batch, load_training_and_validation_data
from .checkpoint import load_checkpoint, save_checkpoint
from .config import ModelConfig, TrainingConfig
from .device import (
    get_default_device,
    get_device_type,
    get_precision_dtype,
    resolve_device,
    supports_mixed_precision,
)
from .model import LanguageModel
from .tokenizer import DEFAULT_ENCODING_NAME, get_vocabulary_size


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
    device_type = get_device_type(device)
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = {"fused": True} if use_fused else {}

    return torch.optim.AdamW(
        optimizer_groups,
        lr=learning_rate,
        betas=betas,
        **extra_args,
    )


def maybe_compile_model(model, compile_model):
    if not compile_model:
        return model

    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this version of PyTorch.")

    return torch.compile(model)


def get_autocast_context(device, mixed_precision, precision_dtype):
    device_type = get_device_type(device)

    if (
        not mixed_precision
        or not supports_mixed_precision(device)
        or precision_dtype == "float32"
    ):
        return nullcontext()

    return torch.amp.autocast(
        device_type=device_type,
        dtype=get_precision_dtype(precision_dtype),
    )


def create_gradient_scaler(device, mixed_precision, precision_dtype):
    use_scaler = (
        mixed_precision
        and get_device_type(device) == "cuda"
        and precision_dtype == "float16"
    )

    return torch.amp.GradScaler("cuda", enabled=use_scaler)


def get_learning_rate(
    step,
    base_learning_rate,
    min_learning_rate,
    warmup_steps,
    decay_steps,
):
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative.")

    if decay_steps <= warmup_steps:
        raise ValueError("decay_steps must be greater than warmup_steps.")

    if warmup_steps > 0 and step < warmup_steps:
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
    if eval_batches < 1:
        raise ValueError("eval_batches must be at least 1.")

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
    mixed_precision=False,
    precision_dtype="float16",
    device=None,
    print_progress=False,
):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if context_size < 1:
        raise ValueError("context_size must be at least 1.")

    if training_steps < 1:
        raise ValueError("training_steps must be at least 1.")

    if eval_interval < 1:
        raise ValueError("eval_interval must be at least 1.")

    if eval_batches < 1:
        raise ValueError("eval_batches must be at least 1.")

    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative.")

    if decay_steps <= warmup_steps:
        raise ValueError("decay_steps must be greater than warmup_steps.")

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1.")

    device = device or get_default_device()
    model.to(device)
    model.train()

    history = []
    best_validation_loss = math.inf
    best_checkpoint_path = None
    start_step = 1
    training_start_time = time.time()

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

    scaler = create_gradient_scaler(
        device=device,
        mixed_precision=mixed_precision,
        precision_dtype=precision_dtype,
    )

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
        grad_norm = None

        for _ in range(gradient_accumulation_steps):
            input_tensor, target_tensor = create_batch(
                data=training_data,
                batch_size=batch_size,
                context_size=context_size,
                device=device,
            )

            with get_autocast_context(
                device=device,
                mixed_precision=mixed_precision,
                precision_dtype=precision_dtype,
            ):
                _, loss = model(input_tensor, target_tensor)
                loss = loss / gradient_accumulation_steps

            total_loss += loss.item()
            scaler.scale(loss).backward()

        if gradient_clip is not None:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip,
            )
            grad_norm = float(grad_norm.detach().cpu().item())

        scaler.step(optimizer)
        scaler.update()

        should_evaluate = step == 1 or step % eval_interval == 0

        if should_evaluate:
            elapsed_seconds = max(time.time() - training_start_time, 0.001)
            completed_steps = max(step - start_step + 1, 1)
            tokens_per_step = batch_size * context_size * gradient_accumulation_steps
            tokens_seen = completed_steps * tokens_per_step
            tokens_per_second = tokens_seen / elapsed_seconds
            remaining_steps = max(training_steps - step, 0)
            eta_seconds = remaining_steps * elapsed_seconds / completed_steps
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
                    "grad_norm": grad_norm,
                    "tokens_per_second": tokens_per_second,
                    "eta_seconds": eta_seconds,
                }
            )

            if print_progress:
                grad_norm_text = "n/a" if grad_norm is None else f"{grad_norm:.4f}"
                print(
                    "step={step}/{training_steps} "
                    "train={training:.4f} "
                    "val={validation:.4f} "
                    "loss={loss:.4f} "
                    "lr={learning_rate:.2e} "
                    "grad_norm={grad_norm} "
                    "tok/s={tokens_per_second:.0f} "
                    "eta={eta}".format(
                        step=step,
                        training_steps=training_steps,
                        training=losses["training"],
                        validation=losses["validation"],
                        loss=total_loss,
                        learning_rate=learning_rate,
                        grad_norm=grad_norm_text,
                        tokens_per_second=tokens_per_second,
                        eta=format_duration(eta_seconds),
                    ),
                    flush=True,
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


def format_duration(seconds):
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"

    if minutes:
        return f"{minutes}m {seconds}s"

    return f"{seconds}s"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the final LearnGPT model on prepared token data.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/learngpt.pt"))
    parser.add_argument("--resume-checkpoint-path", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--encoding-name", default=DEFAULT_ENCODING_NAME)

    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-transformer-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-weight-tying", action="store_true")
    parser.add_argument("--use-scaled-dot-product-attention", action="store_true")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--base-learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--decay-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--no-gradient-clip", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)

    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument(
        "--precision-dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    training_data, validation_data = load_training_and_validation_data(args.data_dir)

    model_config = ModelConfig(
        vocabulary_size=get_vocabulary_size(args.encoding_name),
        context_size=args.context_size,
        embedding_size=args.embedding_size,
        num_heads=args.num_heads,
        num_transformer_blocks=args.num_transformer_blocks,
        dropout=args.dropout,
        tie_weights=not args.no_weight_tying,
        use_scaled_dot_product_attention=args.use_scaled_dot_product_attention,
    )
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        training_steps=args.training_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        base_learning_rate=args.base_learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        weight_decay=args.weight_decay,
        gradient_clip=None if args.no_gradient_clip else args.gradient_clip,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        resume_from_checkpoint=args.resume_checkpoint_path is not None,
        compile_model=args.compile_model,
        mixed_precision=args.mixed_precision,
        precision_dtype=args.precision_dtype,
    )

    model = LanguageModel(**model_config.to_model_kwargs())
    model = maybe_compile_model(model, compile_model=args.compile_model)
    optimizer = configure_optimizer(
        model=model,
        learning_rate=args.base_learning_rate,
        weight_decay=args.weight_decay,
        device=device,
    )

    tokenizer_config = {"encoding_name": args.encoding_name}

    print("LearnGPT training")
    print(json.dumps(
        {
            "device": str(device),
            "data_dir": str(args.data_dir),
            "checkpoint_path": str(args.checkpoint_path),
            "train_tokens": int(len(training_data)),
            "val_tokens": int(len(validation_data)),
            "model_config": model_config.to_checkpoint_dict(),
            "training_config": training_config.to_checkpoint_dict(),
        },
        indent=2,
    ))
    print()

    history, best_checkpoint_path = train_model(
        model=model,
        optimizer=optimizer,
        training_data=training_data,
        validation_data=validation_data,
        batch_size=args.batch_size,
        context_size=args.context_size,
        training_steps=args.training_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_path=args.checkpoint_path,
        model_config=model_config.to_checkpoint_dict(),
        tokenizer_config=tokenizer_config,
        base_learning_rate=args.base_learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        gradient_clip=None if args.no_gradient_clip else args.gradient_clip,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        resume_checkpoint_path=args.resume_checkpoint_path,
        training_config=training_config.to_checkpoint_dict(),
        mixed_precision=args.mixed_precision,
        precision_dtype=args.precision_dtype,
        device=device,
        print_progress=True,
    )

    print()
    print("Training finished.")
    print("Best checkpoint:", best_checkpoint_path)
    if history:
        print("Last metrics:")
        print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
