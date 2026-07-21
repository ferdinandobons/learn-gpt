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
import platform
from pathlib import Path
import time

import torch

from .batching import (
    DEFAULT_DATA_DIR,
    create_batch,
    create_dataset_fingerprint,
    load_training_and_validation_data,
)
from .checkpoint import (
    capture_rng_state,
    load_checkpoint,
    load_checkpoint_payload,
    restore_checkpoint_rng_state,
    save_checkpoint,
    unwrap_model,
)
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


CUDA_AMP_OVERFLOW_RETRY_ATTEMPTS = 8


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


def validate_checkpoint_start(
    checkpoint_path,
    resume_checkpoint_path=None,
    overwrite_checkpoints=False,
):
    """Protect existing runs from accidental fresh-training overwrites."""
    checkpoint_path = Path(checkpoint_path)

    if resume_checkpoint_path is not None:
        if overwrite_checkpoints:
            raise ValueError(
                "--overwrite-checkpoints cannot be combined with "
                "--resume-checkpoint-path."
            )
        resume_checkpoint_path = Path(resume_checkpoint_path)
        latest_checkpoint_path = get_latest_checkpoint_path(checkpoint_path)
        if resume_checkpoint_path.resolve() != latest_checkpoint_path.resolve():
            raise ValueError(
                "--resume-checkpoint-path must be the -latest checkpoint from "
                "the same family as --checkpoint-path. Branching into a new "
                "checkpoint family is intentionally not implicit."
            )
        missing_paths = [
            path
            for path in (checkpoint_path, latest_checkpoint_path)
            if not path.exists()
        ]
        if missing_paths:
            raise FileNotFoundError(
                "A safe resume requires both the existing best and latest "
                "checkpoints from one family. Missing: "
                + ", ".join(str(path) for path in missing_paths)
            )
        return

    existing_paths = [
        path
        for path in (checkpoint_path, get_latest_checkpoint_path(checkpoint_path))
        if path.exists()
    ]
    if existing_paths and not overwrite_checkpoints:
        names = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "Fresh training would overwrite an existing checkpoint: "
            f"{names}. Choose a new --checkpoint-path or explicitly pass "
            "--overwrite-checkpoints."
        )


def validate_resume_dataset(
    checkpoint,
    dataset_fingerprint,
    allow_data_change=False,
):
    """Reject a resume against different token files unless explicitly allowed."""
    saved_fingerprint = checkpoint.get("dataset_fingerprint")
    if saved_fingerprint is None or dataset_fingerprint is None:
        return "unverified"

    if saved_fingerprint.get("value") == dataset_fingerprint.get("value"):
        return "matched"

    if not allow_data_change:
        raise RuntimeError(
            "The resume checkpoint was created from different token files. "
            "Use the original --data-dir, or pass --allow-data-change only for "
            "an intentional non-reproducible continuation."
        )

    return "changed-by-user"


def get_runtime_metadata(device):
    metadata = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "device": str(device),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    if get_device_type(device) == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata.update(
            {
                "cuda_version": str(torch.version.cuda),
                "cuda_device_name": properties.name,
                "cuda_device_memory_bytes": int(properties.total_memory),
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )

    return metadata


def restore_rng_state(rng_state):
    restore_checkpoint_rng_state({"rng_state": rng_state})


def preallocate_gradient_buffers(model):
    """Allocate persistent gradients before MPS enters AccumulateGrad."""
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.grad is None:
                parameter.grad = torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                )


def clear_gradient_buffers(model):
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.zero_()


def get_gradient_norm(model):
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=float("inf"),
        error_if_nonfinite=True,
    )
    return float(gradient_norm.detach().cpu().item())


def is_nonfinite_gradient_error(error):
    """Identify the non-finite norm error produced after AMP unscaling."""
    message = str(error).lower()
    return "non-finite" in message or "nonfinite" in message


def get_gradient_signature(model):
    """Sample every gradient so the MPS startup check compares direction too."""
    signature_parts = []
    for parameter in model.parameters():
        if parameter.grad is None or parameter.grad.numel() == 0:
            continue
        flattened = parameter.grad.detach().reshape(-1)
        indices = sorted({0, flattened.numel() // 2, flattened.numel() - 1})
        signature_parts.append(flattened[indices].float().cpu())

    if not signature_parts:
        raise RuntimeError("The gradient self-check produced no gradients.")

    return torch.cat(signature_parts)


def get_gradient_cosine_similarity(first_model, second_model):
    dot_product = 0.0
    first_squared_norm = 0.0
    second_squared_norm = 0.0

    first_parameters = dict(first_model.named_parameters())
    second_parameters = dict(second_model.named_parameters())
    if first_parameters.keys() != second_parameters.keys():
        raise RuntimeError("Gradient comparison models have different parameters.")

    for name, first_parameter in first_parameters.items():
        second_parameter = second_parameters[name]
        if first_parameter.grad is None or second_parameter.grad is None:
            raise RuntimeError(f"Missing gradient for parameter {name!r}.")
        first_gradient = first_parameter.grad.detach().float().cpu().reshape(-1)
        second_gradient = second_parameter.grad.detach().float().cpu().reshape(-1)
        dot_product += torch.dot(first_gradient, second_gradient).item()
        first_squared_norm += torch.dot(first_gradient, first_gradient).item()
        second_squared_norm += torch.dot(second_gradient, second_gradient).item()

    denominator = math.sqrt(first_squared_norm * second_squared_norm)
    if denominator == 0.0:
        raise RuntimeError("Gradient comparison produced a zero norm.")

    return dot_product / denominator


def run_backward_batches(
    model,
    batches,
    device,
    mixed_precision,
    precision_dtype,
):
    total_loss = 0.0
    for input_tensor, target_tensor in batches:
        with get_autocast_context(
            device=device,
            mixed_precision=mixed_precision,
            precision_dtype=precision_dtype,
        ):
            _, loss = model(input_tensor, target_tensor)
            loss = loss / len(batches)
        total_loss += loss.item()
        loss.backward()

    return total_loss


def run_mps_gradient_self_check(
    model,
    model_config,
    training_data,
    batch_size,
    context_size,
    gradient_accumulation_steps,
    device,
    mixed_precision=False,
    precision_dtype="float16",
):
    """Warm MPS autograd, then require two identical backward passes to agree."""
    if get_device_type(device) != "mps":
        return None

    original_rng_state = capture_rng_state()
    was_training = model.training
    try:
        model.eval()
        batches = [
            create_batch(
                data=training_data,
                batch_size=batch_size,
                context_size=context_size,
                device=device,
            )
            for _ in range(gradient_accumulation_steps)
        ]
        backward_rng_state = capture_rng_state()
        preallocate_gradient_buffers(model)

        # A first real-shape backward initializes MPS kernels and gradient paths.
        restore_rng_state(backward_rng_state)
        run_backward_batches(
            model=model,
            batches=batches,
            device=device,
            mixed_precision=mixed_precision,
            precision_dtype=precision_dtype,
        )
        clear_gradient_buffers(model)

        probe_results = []
        for probe_index in range(2):
            restore_rng_state(backward_rng_state)
            probe_loss = run_backward_batches(
                model=model,
                batches=batches,
                device=device,
                mixed_precision=mixed_precision,
                precision_dtype=precision_dtype,
            )
            probe_results.append(
                (probe_loss, get_gradient_norm(model), get_gradient_signature(model))
            )
            if probe_index == 0:
                clear_gradient_buffers(model)

        first_loss, first_norm, first_signature = probe_results[0]
        second_loss, second_norm, second_signature = probe_results[1]
        probe_losses_match = math.isclose(
            first_loss,
            second_loss,
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
        norms_match = math.isclose(first_norm, second_norm, rel_tol=0.01, abs_tol=1e-5)
        signatures_match = torch.allclose(
            first_signature,
            second_signature,
            rtol=0.01,
            atol=1e-6,
        )
        if not probe_losses_match or not norms_match or not signatures_match:
            raise RuntimeError(
                "MPS gradient self-check failed before training: identical "
                f"backward passes produced norms {first_norm:.6g} and "
                f"{second_norm:.6g}. No model update was applied."
            )

        cpu_model_config = ModelConfig.from_checkpoint_dict(model_config)
        cpu_model = LanguageModel(**cpu_model_config.to_model_kwargs())
        cpu_model.load_state_dict(unwrap_model(model).state_dict())
        # CPU's monolithic projection is the independent reference for the
        # chunked MPS workaround.
        cpu_model.output_chunk_size = 0
        cpu_model.eval()
        cpu_batches = [
            (input_tensor.cpu(), target_tensor.cpu())
            for input_tensor, target_tensor in batches
        ]
        cpu_loss = run_backward_batches(
            model=cpu_model,
            batches=cpu_batches,
            device="cpu",
            mixed_precision=False,
            precision_dtype="float32",
        )
        cpu_norm = get_gradient_norm(cpu_model)
        gradient_cosine = get_gradient_cosine_similarity(
            unwrap_model(model),
            cpu_model,
        )
        norm_relative_error = abs(second_norm - cpu_norm) / max(cpu_norm, 1e-12)
        losses_match = math.isclose(
            second_loss,
            cpu_loss,
            rel_tol=1e-4,
            abs_tol=1e-5,
        )
        if (
            not losses_match
            or norm_relative_error > 0.01
            or gradient_cosine < 0.999
        ):
            raise RuntimeError(
                "MPS gradient CPU-parity check failed before training: "
                f"MPS norm={second_norm:.6g}, CPU norm={cpu_norm:.6g}, "
                f"relative error={norm_relative_error:.2%}, "
                f"cosine={gradient_cosine:.6f}. No model update was applied."
            )

        return {
            "first_grad_norm": first_norm,
            "second_grad_norm": second_norm,
            "cpu_grad_norm": cpu_norm,
            "gradient_cosine": gradient_cosine,
        }
    finally:
        restore_rng_state(original_rng_state)
        clear_gradient_buffers(model)
        if was_training:
            model.train()


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
    mixed_precision=False,
    precision_dtype="float16",
):
    if eval_batches < 1:
        raise ValueError("eval_batches must be at least 1.")

    device = device or get_default_device()
    was_training = model.training
    model.eval()
    rng_state = capture_rng_state()

    losses_by_split = {}
    data_by_split = {
        "training": training_data,
        "validation": validation_data,
    }

    try:
        for split_name, split_data in data_by_split.items():
            split_losses = []

            for _ in range(eval_batches):
                input_tensor, target_tensor = create_batch(
                    data=split_data,
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
                loss_value = float(loss.detach().item())
                if not math.isfinite(loss_value):
                    raise RuntimeError(
                        f"Non-finite {split_name} loss during evaluation."
                    )
                split_losses.append(loss_value)

            losses_by_split[split_name] = sum(split_losses) / len(split_losses)
    finally:
        restore_rng_state(rng_state)
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
    log_interval=None,
    max_grad_norm_before_clip=None,
    gradient_retry_attempts=0,
    gradient_accumulation_steps=1,
    context_sensitivity_contexts=0,
    resume_checkpoint_path=None,
    training_config=None,
    runtime_metadata=None,
    dataset_fingerprint=None,
    allow_data_change=False,
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

    if log_interval is not None and log_interval < 1:
        raise ValueError("log_interval must be at least 1 when set.")

    if eval_batches < 1:
        raise ValueError("eval_batches must be at least 1.")

    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative.")

    if decay_steps <= warmup_steps:
        raise ValueError("decay_steps must be greater than warmup_steps.")

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1.")

    if (
        max_grad_norm_before_clip is not None
        and max_grad_norm_before_clip <= 0
    ):
        raise ValueError(
            "max_grad_norm_before_clip must be greater than 0 when set."
        )

    if gradient_retry_attempts < 0:
        raise ValueError("gradient_retry_attempts cannot be negative.")

    if gradient_retry_attempts and max_grad_norm_before_clip is None:
        raise ValueError(
            "gradient_retry_attempts requires max_grad_norm_before_clip."
        )

    if context_sensitivity_contexts < 0:
        raise ValueError("context_sensitivity_contexts cannot be negative.")

    if context_sensitivity_contexts == 1:
        raise ValueError("context_sensitivity_contexts must be 0 or at least 2.")

    device = device or get_default_device()
    device_type = get_device_type(device)
    model.to(device)
    model.train()

    history = []
    best_validation_loss = math.inf
    best_checkpoint_path = None
    latest_checkpoint_path = get_latest_checkpoint_path(checkpoint_path)
    start_step = 1
    amp_overflow_count = 0

    scaler = create_gradient_scaler(
        device=device,
        mixed_precision=mixed_precision,
        precision_dtype=precision_dtype,
    )

    if scaler.is_enabled() and gradient_retry_attempts:
        raise ValueError(
            "gradient retries are not supported with a CUDA gradient scaler."
        )

    if resume_checkpoint_path is not None:
        checkpoint = load_checkpoint(
            checkpoint_path=resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
            gradient_scaler=scaler,
        )
        if (
            print_progress
            and scaler.is_enabled()
            and not checkpoint.get("gradient_scaler_state_dict")
        ):
            print(
                "Warning: this legacy CUDA checkpoint has no GradScaler state; "
                "mixed-precision scaling starts from the current default.",
                flush=True,
            )
        dataset_status = validate_resume_dataset(
            checkpoint=checkpoint,
            dataset_fingerprint=dataset_fingerprint,
            allow_data_change=allow_data_change,
        )
        if print_progress and dataset_status == "unverified":
            print(
                "Warning: this legacy checkpoint has no dataset fingerprint; "
                "the resume dataset identity cannot be verified.",
                flush=True,
            )
        elif print_progress and dataset_status == "changed-by-user":
            print(
                "Warning: resuming with different token files because "
                "--allow-data-change was explicitly set.",
                flush=True,
            )
        start_step = int(checkpoint.get("step", 0)) + 1
        saved_best_validation_loss = checkpoint.get("best_validation_loss")
        if saved_best_validation_loss is not None:
            best_validation_loss = float(saved_best_validation_loss)
        amp_overflow_count = int(
            (checkpoint.get("losses") or {}).get("amp_overflow_count", 0)
        )
        best_checkpoint_path = (
            Path(checkpoint_path) if Path(checkpoint_path).exists() else None
        )

    if start_step > training_steps:
        raise ValueError(
            "The resume checkpoint is already at or beyond training_steps. "
            "Choose a larger total --training-steps value."
        )

    mps_self_check = run_mps_gradient_self_check(
        model=model,
        model_config=model_config,
        training_data=training_data,
        batch_size=batch_size,
        context_size=context_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        device=device,
        mixed_precision=mixed_precision,
        precision_dtype=precision_dtype,
    )
    use_persistent_gradients = device_type == "mps"
    if use_persistent_gradients:
        preallocate_gradient_buffers(model)
    if print_progress and mps_self_check is not None:
        print(
            "MPS gradient self-check passed: "
            f"MPS norms={mps_self_check['first_grad_norm']:.6f},"
            f"{mps_self_check['second_grad_norm']:.6f}; "
            f"CPU norm={mps_self_check['cpu_grad_norm']:.6f}; "
            f"cosine={mps_self_check['gradient_cosine']:.6f}",
            flush=True,
        )

    training_start_time = time.time()
    tokens_per_step = batch_size * context_size * gradient_accumulation_steps

    for step in range(start_step, training_steps + 1):
        learning_rate = get_learning_rate(
            step=step,
            base_learning_rate=base_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
        )
        apply_learning_rate(optimizer=optimizer, learning_rate=learning_rate)

        batches = [
            create_batch(
                data=training_data,
                batch_size=batch_size,
                context_size=context_size,
                device=device,
            )
            for _ in range(gradient_accumulation_steps)
        ]
        backward_rng_state = capture_rng_state()
        total_loss = 0.0
        grad_norm = None
        gradient_retries = 0
        amp_overflow_retries = 0
        attempt_limit = (
            CUDA_AMP_OVERFLOW_RETRY_ATTEMPTS
            if scaler.is_enabled()
            else gradient_retry_attempts
        )

        for attempt in range(attempt_limit + 1):
            restore_rng_state(backward_rng_state)
            optimizer.zero_grad(set_to_none=not use_persistent_gradients)
            total_loss = 0.0
            amp_overflow_detected = False

            for input_tensor, target_tensor in batches:
                with get_autocast_context(
                    device=device,
                    mixed_precision=mixed_precision,
                    precision_dtype=precision_dtype,
                ):
                    _, loss = model(input_tensor, target_tensor)
                    loss = loss / gradient_accumulation_steps

                loss_value = float(loss.detach().item())
                if not math.isfinite(loss_value):
                    optimizer.zero_grad(set_to_none=not use_persistent_gradients)
                    raise RuntimeError(
                        f"Non-finite training loss at step {step}. "
                        "No optimizer update was applied."
                    )
                total_loss += loss_value
                scaler.scale(loss).backward()

            gradient_integrity_error = None
            if (
                scaler.is_enabled()
                or gradient_clip is not None
                or max_grad_norm_before_clip is not None
            ):
                scaler.unscale_(optimizer)
                try:
                    grad_norm = get_gradient_norm(model)
                except RuntimeError as error:
                    if scaler.is_enabled() and is_nonfinite_gradient_error(error):
                        # GradScaler already recorded the overflow in
                        # unscale_(). Back off the scale, then repeat this same
                        # batch and training step so 45,000 steps still means
                        # 45,000 successful optimizer updates.
                        amp_overflow_detected = True
                    else:
                        gradient_integrity_error = str(error)

            if (
                gradient_integrity_error is None
                and not amp_overflow_detected
                and max_grad_norm_before_clip is not None
                and grad_norm is not None
                and grad_norm > max_grad_norm_before_clip
            ):
                gradient_integrity_error = (
                    f"raw gradient norm {grad_norm:.6g} exceeded the configured "
                    f"limit {max_grad_norm_before_clip:.6g}"
                )

            if (
                gradient_integrity_error is None
                and not amp_overflow_detected
                and gradient_clip is not None
            ):
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as error:
                    gradient_integrity_error = str(error)

            if amp_overflow_detected:
                if attempt == attempt_limit:
                    raise RuntimeError(
                        f"CUDA AMP overflow persisted at step {step} after "
                        f"{attempt + 1} attempt(s). No optimizer update was applied."
                    )

                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                amp_overflow_retries += 1
                amp_overflow_count += 1
                if print_progress:
                    print(
                        f"step={step} CUDA AMP overflow retry "
                        f"{amp_overflow_retries}/{attempt_limit}: "
                        f"scale {previous_scale:.0f} -> {scaler.get_scale():.0f}",
                        flush=True,
                    )
                continue

            if gradient_integrity_error is None:
                gradient_retries = attempt if not scaler.is_enabled() else 0
                break

            if scaler.is_enabled() or attempt == gradient_retry_attempts:
                raise RuntimeError(
                    f"Gradient integrity check failed at step {step} after "
                    f"{attempt + 1} attempt(s): {gradient_integrity_error}. "
                    "No optimizer update was applied."
                )

            if print_progress:
                print(
                    f"step={step} gradient retry {attempt + 1}/"
                    f"{gradient_retry_attempts}: {gradient_integrity_error}",
                    flush=True,
                )

        scaler.step(optimizer)
        scaler.update()

        should_evaluate = (
            step == 1
            or step % eval_interval == 0
            or step == training_steps
        )

        if (
            print_progress
            and log_interval is not None
            and step % log_interval == 0
            and not should_evaluate
        ):
            elapsed_seconds = max(time.time() - training_start_time, 0.001)
            completed_steps = max(step - start_step + 1, 1)
            tokens_seen = completed_steps * tokens_per_step
            tokens_per_second = tokens_seen / elapsed_seconds
            remaining_steps = max(training_steps - step, 0)
            eta_seconds = remaining_steps * elapsed_seconds / completed_steps
            grad_norm_text = "n/a" if grad_norm is None else f"{grad_norm:.4f}"
            print(
                "step={step}/{training_steps} "
                "loss={loss:.4f} "
                "lr={learning_rate:.2e} "
                "grad_norm={grad_norm} "
                "amp_retries={amp_overflow_retries} "
                "amp_overflows={amp_overflow_count} "
                "tok/s={tokens_per_second:.0f} "
                "eta={eta} (log only)".format(
                    step=step,
                    training_steps=training_steps,
                    loss=total_loss,
                    learning_rate=learning_rate,
                    grad_norm=grad_norm_text,
                    amp_overflow_retries=amp_overflow_retries,
                    amp_overflow_count=amp_overflow_count,
                    tokens_per_second=tokens_per_second,
                    eta=format_duration(eta_seconds),
                ),
                flush=True,
            )

        if should_evaluate:
            elapsed_seconds = max(time.time() - training_start_time, 0.001)
            completed_steps = max(step - start_step + 1, 1)
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
                mixed_precision=mixed_precision,
                precision_dtype=precision_dtype,
            )
            metrics = {
                "step": step,
                "learning_rate": learning_rate,
                "loss": total_loss,
                "training": losses["training"],
                "validation": losses["validation"],
                "grad_norm": grad_norm,
                "gradient_retries": gradient_retries,
                "amp_overflow_retries": amp_overflow_retries,
                "amp_overflow_count": amp_overflow_count,
                "tokens_per_second": tokens_per_second,
                "eta_seconds": eta_seconds,
            }
            if context_sensitivity_contexts:
                context_metrics = estimate_context_sensitivity(
                    model=model,
                    validation_data=validation_data,
                    context_size=context_size,
                    num_contexts=context_sensitivity_contexts,
                    device=device,
                )
                metrics.update(context_metrics)
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
                    "grad_retries={gradient_retries} "
                    "amp_retries={amp_overflow_retries} "
                    "amp_overflows={amp_overflow_count} "
                    "tok/s={tokens_per_second:.0f} "
                    "eta={eta}".format(
                        step=step,
                        training_steps=training_steps,
                        training=losses["training"],
                        validation=losses["validation"],
                        loss=total_loss,
                        learning_rate=learning_rate,
                        grad_norm=grad_norm_text,
                        gradient_retries=gradient_retries,
                        amp_overflow_retries=amp_overflow_retries,
                        amp_overflow_count=amp_overflow_count,
                        tokens_per_second=tokens_per_second,
                        eta=format_duration(eta_seconds),
                    ),
                    flush=True,
                )
                if context_sensitivity_contexts:
                    print(
                        "context_js={context_js:.2e} "
                        "context_logit_std={context_logit_std:.2e} "
                        "context_true_loss={context_true_loss:.4f} "
                        "context_shuffled_loss={context_shuffled_loss:.4f} "
                        "context_loss_gain={context_loss_gain:+.4f}".format(
                            context_js=metrics["context_js_divergence"],
                            context_logit_std=metrics["context_logit_std"],
                            context_true_loss=metrics["context_true_loss"],
                            context_shuffled_loss=metrics[
                                "context_shuffled_loss"
                            ],
                            context_loss_gain=metrics["context_loss_gain"],
                        ),
                        flush=True,
                    )

            is_best_checkpoint = losses["validation"] < best_validation_loss
            if is_best_checkpoint:
                best_validation_loss = losses["validation"]

            checkpoint_losses = dict(losses)
            checkpoint_losses["grad_norm"] = grad_norm
            checkpoint_losses["gradient_retries"] = gradient_retries
            checkpoint_losses["amp_overflow_retries"] = amp_overflow_retries
            checkpoint_losses["amp_overflow_count"] = amp_overflow_count
            for metric_name in (
                "context_js_divergence",
                "context_logit_std",
                "context_true_loss",
                "context_shuffled_loss",
                "context_loss_gain",
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
                runtime_metadata=runtime_metadata,
                gradient_scaler=scaler,
                dataset_fingerprint=dataset_fingerprint,
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
                    runtime_metadata=runtime_metadata,
                    gradient_scaler=scaler,
                    dataset_fingerprint=dataset_fingerprint,
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
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--no-weight-tying", action="store_true")
    parser.add_argument("--use-scaled-dot-product-attention", action="store_true")
    parser.add_argument("--fused-attention", action="store_true")
    parser.add_argument("--output-chunk-size", type=int, default=32768)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--training-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--base-learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--decay-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--no-gradient-clip", action="store_true")
    parser.add_argument("--max-grad-norm-before-clip", type=float, default=None)
    parser.add_argument("--gradient-retry-attempts", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--context-sensitivity-contexts", type=int, default=0)

    parser.add_argument(
        "--overwrite-checkpoints",
        action="store_true",
        help="Allow a fresh run to replace existing best/latest checkpoint files.",
    )
    parser.add_argument(
        "--allow-data-change",
        action="store_true",
        help="Allow resume with token files that differ from the checkpoint fingerprint.",
    )

    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--compile-model",
        dest="compile_model",
        action="store_true",
    )
    compile_group.add_argument(
        "--no-compile-model",
        dest="compile_model",
        action="store_false",
    )
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument(
        "--mixed-precision",
        dest="mixed_precision",
        action="store_true",
    )
    precision_group.add_argument(
        "--no-mixed-precision",
        dest="mixed_precision",
        action="store_false",
    )
    parser.set_defaults(compile_model=None, mixed_precision=None)
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
    validate_checkpoint_start(
        checkpoint_path=args.checkpoint_path,
        resume_checkpoint_path=args.resume_checkpoint_path,
        overwrite_checkpoints=args.overwrite_checkpoints,
    )
    if args.allow_data_change and args.resume_checkpoint_path is None:
        raise ValueError("--allow-data-change requires --resume-checkpoint-path.")

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
            bias=not args.no_bias,
            tie_weights=not args.no_weight_tying,
            use_scaled_dot_product_attention=args.use_scaled_dot_product_attention,
            fused_attention=args.fused_attention,
            output_chunk_size=args.output_chunk_size,
        )

    if checkpoint_preview is not None and checkpoint_preview.get("training_config"):
        training_config = TrainingConfig.from_checkpoint_dict(
            checkpoint_preview["training_config"]
        )
        if args.training_steps is not None:
            training_config.training_steps = args.training_steps
        if args.log_interval is not None:
            training_config.log_interval = args.log_interval
        if args.max_grad_norm_before_clip is not None:
            training_config.max_grad_norm_before_clip = (
                args.max_grad_norm_before_clip
            )
        if args.gradient_retry_attempts is not None:
            training_config.gradient_retry_attempts = args.gradient_retry_attempts
        training_config.resume_from_checkpoint = True
        if args.compile_model is not None:
            training_config.compile_model = args.compile_model
        if args.mixed_precision is not None:
            training_config.mixed_precision = args.mixed_precision
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
            log_interval=args.log_interval,
            eval_batches=args.eval_batches,
            base_learning_rate=args.base_learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_steps=args.warmup_steps,
            decay_steps=args.decay_steps,
            weight_decay=args.weight_decay,
            gradient_clip=None if args.no_gradient_clip else args.gradient_clip,
            max_grad_norm_before_clip=args.max_grad_norm_before_clip,
            gradient_retry_attempts=(
                0
                if args.gradient_retry_attempts is None
                else args.gradient_retry_attempts
            ),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            context_sensitivity_contexts=args.context_sensitivity_contexts,
            resume_from_checkpoint=args.resume_checkpoint_path is not None,
            compile_model=bool(args.compile_model),
            mixed_precision=bool(args.mixed_precision),
            precision_dtype=args.precision_dtype,
        )

    checkpoint_preview = None
    torch.manual_seed(training_config.seed)
    print(f"Fingerprinting prepared token files in {args.data_dir}...", flush=True)
    dataset_fingerprint = create_dataset_fingerprint(args.data_dir)
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
    runtime_metadata = get_runtime_metadata(device)

    print("LearnGPT training")
    print(json.dumps(
        {
            "device": str(device),
            "data_dir": str(args.data_dir),
            "checkpoint_path": str(args.checkpoint_path),
            "train_tokens": int(len(training_data)),
            "val_tokens": int(len(validation_data)),
            "dataset_fingerprint": dataset_fingerprint["value"],
            "runtime": runtime_metadata,
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
        log_interval=training_config.log_interval,
        eval_batches=training_config.eval_batches,
        checkpoint_path=args.checkpoint_path,
        model_config=model_config.to_checkpoint_dict(),
        tokenizer_config=tokenizer_config,
        base_learning_rate=training_config.base_learning_rate,
        min_learning_rate=training_config.min_learning_rate,
        warmup_steps=training_config.warmup_steps,
        decay_steps=training_config.decay_steps,
        gradient_clip=training_config.gradient_clip,
        max_grad_norm_before_clip=training_config.max_grad_norm_before_clip,
        gradient_retry_attempts=training_config.gradient_retry_attempts,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        context_sensitivity_contexts=training_config.context_sensitivity_contexts,
        resume_checkpoint_path=args.resume_checkpoint_path,
        training_config=training_config.to_checkpoint_dict(),
        runtime_metadata=runtime_metadata,
        dataset_fingerprint=dataset_fingerprint,
        allow_data_change=args.allow_data_change,
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
