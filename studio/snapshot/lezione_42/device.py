"""
Differenza rispetto ai file precedenti:
- Prima il codice finale non sceglieva esplicitamente dove eseguire il modello.
- Qui aggiungiamo funzioni condivise per scegliere CPU, CUDA o MPS e per
  descrivere perché un acceleratore è disponibile oppure no.

Scopo del file:
- Usare CUDA quando disponibile.
- Usare Metal/MPS sui Mac compatibili quando CUDA non è disponibile.
- Tornare alla CPU quando non ci sono acceleratori.
- Tenere mixed precision e ottimizzazioni avanzate dietro flag espliciti.
"""

import torch


def is_mps_available():
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def is_mps_built():
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
    )


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if is_mps_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_device_type(device):
    return torch.device(device).type


def get_precision_dtype(precision_dtype):
    if precision_dtype == "float16":
        return torch.float16

    if precision_dtype == "bfloat16":
        return torch.bfloat16

    if precision_dtype == "float32":
        return torch.float32

    raise ValueError("precision_dtype deve essere 'float16', 'bfloat16' o 'float32'.")


def supports_mixed_precision(device):
    device_type = get_device_type(device)

    return device_type in {"cuda", "mps"}


def get_device_report():
    mps_built = is_mps_built()
    mps_available = is_mps_available()

    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_built": mps_built,
        "mps_available": mps_available,
        "selected_device": str(get_default_device()),
    }
