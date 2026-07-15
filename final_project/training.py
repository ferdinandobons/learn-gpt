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
from .checkpoint import load_checkpoint, load_checkpoint_payload, save_checkpoint
from .config import ModelConfig, TrainingConfig
from .device import (
    get_default_device,
    get_device_type,
    get_precision_dtype,
    resolve_device,
    supports_mixed_precision,
)
from .model import LanguageModel
from .quality import estimate_context_sensitivity
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


def get_latest_checkpoint_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}-latest{checkpoint_path.suffix}"
    )


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
    context_sensitivity_contexts=0,
    min_context_js_divergence=None,
    stop_on_low_context_sensitivity=False,
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

    if context_sensitivity_contexts < 0:
        raise ValueError("context_sensitivity_contexts cannot be negative.")

    if context_sensitivity_contexts == 1:
        raise ValueError("context_sensitivity_contexts must be 0 or at least 2.")

    if min_context_js_divergence is not None and min_context_js_divergence <= 0:
        raise ValueError(
            "min_context_js_divergence must be greater than 0 when set."
        )

    if (
        min_context_js_divergence is not None
        and context_sensitivity_contexts < 2
    ):
        raise ValueError(
            "min_context_js_divergence requires at least two context samples."
        )

    if stop_on_low_context_sensitivity and min_context_js_divergence is None:
        raise ValueError(
            "stop_on_low_context_sensitivity requires min_context_js_divergence."
        )

    device = device or get_default_device()
    model.to(device)
    model.train()

    history = []
    best_validation_loss = math.inf
    best_checkpoint_path = None
    latest_checkpoint_path = get_latest_checkpoint_path(checkpoint_path)
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
        saved_best_validation_loss = checkpoint.get("best_validation_loss")
        if saved_best_validation_loss is not None:
            best_validation_loss = float(saved_best_validation_loss)
        best_checkpoint_path = (
            Path(checkpoint_path)
            if Path(checkpoint_path).exists()
            else Path(resume_checkpoint_path)
        )

    if start_step > training_steps:
        raise ValueError(
            "The resume checkpoint is already at or beyond training_steps. "
            "Choose a larger total --training-steps value."
        )

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

        should_evaluate = (
            step == 1
            or step % eval_interval == 0
            or step == training_steps
        )

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
            metrics = {
                "step": step,
                "learning_rate": learning_rate,
                "loss": total_loss,
                "training": losses["training"],
                "validation": losses["validation"],
                "grad_norm": grad_norm,
                "tokens_per_second": tokens_per_second,
                "eta_seconds": eta_seconds,
            }
            context_gate_failed = False
            if context_sensitivity_contexts:
                context_metrics = estimate_context_sensitivity(
                    model=model,
                    validation_data=validation_data,
                    context_size=context_size,
                    num_contexts=context_sensitivity_contexts,
                    device=device,
                )
                metrics.update(context_metrics)
                if min_context_js_divergence is not None:
                    context_gate_failed = (
                        context_metrics["context_js_divergence"]
                        < min_context_js_divergence
                    )
                    metrics["context_gate_passed"] = not context_gate_failed
            history.append(metrics)

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
                if context_sensitivity_contexts:
                    gate_text = "not-set"
                    if min_context_js_divergence is not None:
                        gate_text = "failed" if context_gate_failed else "passed"
                    print(
                        "context_js={context_js:.2e} "
                        "context_logit_std={context_logit_std:.2e} "
                        "context_gate={gate}".format(
                            context_js=metrics["context_js_divergence"],
                            context_logit_std=metrics["context_logit_std"],
                            gate=gate_text,
                        ),
                        flush=True,
                    )

            is_best_checkpoint = losses["validation"] < best_validation_loss
            if is_best_checkpoint:
                best_validation_loss = losses["validation"]

            checkpoint_losses = dict(losses)
            for metric_name in (
                "context_js_divergence",
                "context_logit_std",
                "context_gate_passed",
            ):
                if metric_name in metrics:
                    checkpoint_losses[metric_name] = metrics[metric_name]

            save_checkpoint(
                checkpoint_path=latest_checkpoint_path,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                step=step,
                losses=checkpoint_losses,
                tokenizer_config=tokenizer_config,
                training_config=training_config,
                best_validation_loss=best_validation_loss,
            )

            if is_best_checkpoint:
                best_checkpoint_path = save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    model_config=model_config,
                    step=step,
                    losses=checkpoint_losses,
                    tokenizer_config=tokenizer_config,
                    training_config=training_config,
                    best_validation_loss=best_validation_loss,
                )

            if context_gate_failed and stop_on_low_context_sensitivity:
                raise RuntimeError(
                    "Context-sensitivity gate failed at step "
                    f"{step}: measured {metrics['context_js_divergence']:.2e}, "
                    "which is below --min-context-js-divergence "
                    f"{min_context_js_divergence:.2e}."
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
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-transformer-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-weight-tying", action="store_true")
    parser.add_argument("--use-scaled-dot-product-attention", action="store_true")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--training-steps", type=int, default=None)
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
    parser.add_argument("--context-sensitivity-contexts", type=int, default=0)
    parser.add_argument("--min-context-js-divergence", type=float, default=None)
    parser.add_argument("--stop-on-low-context-sensitivity", action="store_true")

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
    checkpoint_preview = None

    if args.resume_checkpoint_path is not None:
        checkpoint_preview = load_checkpoint_payload(
            args.resume_checkpoint_path,
            device="cpu",
        )
        model_config = ModelConfig.from_checkpoint_dict(
            checkpoint_preview["model_config"]
        )
        tokenizer_config = checkpoint_preview.get("tokenizer_config") or {
            "encoding_name": DEFAULT_ENCODING_NAME,
        }
        encoding_name = tokenizer_config.get(
            "encoding_name",
            DEFAULT_ENCODING_NAME,
        )
    else:
        encoding_name = args.encoding_name
        tokenizer_config = {"encoding_name": encoding_name}
        model_config = ModelConfig(
            vocabulary_size=get_vocabulary_size(encoding_name),
            context_size=args.context_size,
            embedding_size=args.embedding_size,
            num_heads=args.num_heads,
            num_transformer_blocks=args.num_transformer_blocks,
            dropout=args.dropout,
            tie_weights=not args.no_weight_tying,
            use_scaled_dot_product_attention=args.use_scaled_dot_product_attention,
        )

    if checkpoint_preview is not None and checkpoint_preview.get("training_config"):
        training_config = TrainingConfig.from_checkpoint_dict(
            checkpoint_preview["training_config"]
        )
        if args.training_steps is not None:
            training_config.training_steps = args.training_steps
        training_config.resume_from_checkpoint = True
        training_config.compile_model = args.compile_model
        if args.mixed_precision:
            training_config.mixed_precision = True
            training_config.precision_dtype = args.precision_dtype
        training_config.__post_init__()
    else:
        training_config = TrainingConfig(
            seed=args.seed,
            batch_size=args.batch_size,
            training_steps=(
                1000 if args.training_steps is None else args.training_steps
            ),
            eval_interval=args.eval_interval,
            eval_batches=args.eval_batches,
            base_learning_rate=args.base_learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_steps=args.warmup_steps,
            decay_steps=args.decay_steps,
            weight_decay=args.weight_decay,
            gradient_clip=None if args.no_gradient_clip else args.gradient_clip,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            context_sensitivity_contexts=args.context_sensitivity_contexts,
            min_context_js_divergence=args.min_context_js_divergence,
            stop_on_low_context_sensitivity=args.stop_on_low_context_sensitivity,
            resume_from_checkpoint=args.resume_checkpoint_path is not None,
            compile_model=args.compile_model,
            mixed_precision=args.mixed_precision,
            precision_dtype=args.precision_dtype,
        )

    checkpoint_preview = None
    torch.manual_seed(training_config.seed)
    training_data, validation_data = load_training_and_validation_data(
        args.data_dir,
        encoding_name=encoding_name,
    )

    model = LanguageModel(**model_config.to_model_kwargs())
    model = maybe_compile_model(
        model,
        compile_model=training_config.compile_model,
    )
    optimizer = configure_optimizer(
        model=model,
        learning_rate=training_config.base_learning_rate,
        weight_decay=training_config.weight_decay,
        device=device,
    )

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
        batch_size=training_config.batch_size,
        context_size=model_config.context_size,
        training_steps=training_config.training_steps,
        eval_interval=training_config.eval_interval,
        eval_batches=training_config.eval_batches,
        checkpoint_path=args.checkpoint_path,
        model_config=model_config.to_checkpoint_dict(),
        tokenizer_config=tokenizer_config,
        base_learning_rate=training_config.base_learning_rate,
        min_learning_rate=training_config.min_learning_rate,
        warmup_steps=training_config.warmup_steps,
        decay_steps=training_config.decay_steps,
        gradient_clip=training_config.gradient_clip,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        context_sensitivity_contexts=training_config.context_sensitivity_contexts,
        min_context_js_divergence=training_config.min_context_js_divergence,
        stop_on_low_context_sensitivity=(
            training_config.stop_on_low_context_sensitivity
        ),
        resume_checkpoint_path=args.resume_checkpoint_path,
        training_config=training_config.to_checkpoint_dict(),
        mixed_precision=training_config.mixed_precision,
        precision_dtype=training_config.precision_dtype,
        device=device,
        print_progress=True,
    )

    print()
    print("Training finished.")
    print("Best checkpoint:", best_checkpoint_path)
    print("Latest checkpoint:", get_latest_checkpoint_path(args.checkpoint_path))
    if history:
        print("Last metrics:")
        print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
