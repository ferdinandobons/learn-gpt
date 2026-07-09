"""
Differenza rispetto ai file precedenti:
- Prima la creazione del batch era dentro `08_batch_python.py` e
  `09_batch_torch.py`.
- Qui spostiamo quella logica in un modulo riutilizzabile.

Scopo del file:
- Creare batch di input e target in formato tensore PyTorch.
- Preparare una funzione comune che potremo usare durante il training del
  modello.
"""

import random

import torch


def create_example(data, context_size):
    start_position = random.randint(0, len(data) - context_size - 1)

    input_tokens = data[start_position:start_position + context_size]
    target_tokens = data[start_position + 1:start_position + context_size + 1]

    return input_tokens, target_tokens


def create_batch(data, batch_size, context_size):
    batch_inputs = []
    batch_targets = []

    for _ in range(batch_size):
        input_tokens, target_tokens = create_example(data, context_size)

        batch_inputs.append(input_tokens)
        batch_targets.append(target_tokens)

    input_tensor = torch.tensor(batch_inputs)
    target_tensor = torch.tensor(batch_targets)

    return input_tensor, target_tensor