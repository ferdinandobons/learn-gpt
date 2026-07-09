"""
Differenza rispetto ai file precedenti:
- Prima il codice finale non sceglieva esplicitamente dove eseguire il modello.
- Qui aggiungiamo una piccola funzione condivisa per scegliere CPU, CUDA o MPS.

Scopo del file:
- Usare CUDA quando disponibile.
- Usare Metal/MPS sui Mac compatibili quando CUDA non è disponibile.
- Tornare alla CPU quando non ci sono acceleratori.
"""

import torch


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
