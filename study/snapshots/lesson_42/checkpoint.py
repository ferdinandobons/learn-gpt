"""
Changes compared with the previous files:
- This module belongs to the cleaned final LearnGPT project.
- It uses the English public project layout and the FineWeb-Edu/BPE pipeline.

File purpose:
- Save and load complete training checkpoints.
"""

from pathlib import Path

import torch


def unwrap_model(model):
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod

    return model


def canonicalize_model_state_dict(state_dict):
    canonical_state_dict = state_dict

    while canonical_state_dict and all(
        key.startswith("_orig_mod.") for key in canonical_state_dict
    ):
        canonical_state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in canonical_state_dict.items()
        }

    return canonical_state_dict


def capture_rng_state():
    rng_state = {"cpu": torch.get_rng_state()}

    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()

    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        rng_state["mps"] = torch.mps.get_rng_state()

    return rng_state


def restore_checkpoint_rng_state(checkpoint):
    rng_state = checkpoint.get("rng_state") or {}

    if "cpu" in rng_state:
        torch.set_rng_state(rng_state["cpu"].cpu())

    if "cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])

    if (
        "mps" in rng_state
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "set_rng_state")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.set_rng_state(rng_state["mps"].cpu())


def load_checkpoint_payload(checkpoint_path, device=None):
    return torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )


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
    runtime_metadata=None,
    gradient_scaler=None,
    dataset_fingerprint=None,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    gradient_scaler_state = None
    if gradient_scaler is not None:
        # A disabled GradScaler returns an empty dictionary. Store that as
        # ``None`` so a later, intentional FP16 resume does not try to load an
        # invalid empty state into an enabled scaler.
        gradient_scaler_state = gradient_scaler.state_dict() or None

    checkpoint = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "step": step,
        "losses": losses,
        "tokenizer_config": tokenizer_config,
        "training_config": training_config,
        "best_validation_loss": best_validation_loss,
        "gradient_scaler_state_dict": gradient_scaler_state,
        "dataset_fingerprint": dataset_fingerprint,
        "rng_state": capture_rng_state(),
        "runtime": runtime_metadata or {"torch_version": str(torch.__version__)},
    }

    temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return checkpoint_path


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    device=None,
    restore_rng_state=True,
    gradient_scaler=None,
):
    checkpoint = load_checkpoint_payload(checkpoint_path, device=device)
    model_state_dict = canonicalize_model_state_dict(
        checkpoint["model_state_dict"]
    )
    unwrap_model(model).load_state_dict(model_state_dict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scaler_state = checkpoint.get("gradient_scaler_state_dict")
    if gradient_scaler is not None and scaler_state:
        gradient_scaler.load_state_dict(scaler_state)

    if restore_rng_state:
        restore_checkpoint_rng_state(checkpoint)

    return checkpoint
