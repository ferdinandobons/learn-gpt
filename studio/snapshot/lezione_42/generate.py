"""
Differenza rispetto ai file precedenti:
- Prima la generazione ricaricava un tokenizer a caratteri salvato nel
  checkpoint.
- Qui ricarica la configurazione BPE e genera testo con lo stesso tokenizer
  usato per preparare FineWeb-Edu.
- La generazione può compilare il modello solo quando richiesto esplicitamente.

Scopo del file:
- Ricostruire un `LanguageModel` partendo dal checkpoint.
- Generare testo su CPU, CUDA o MPS.
- Decodificare gli ID generati con GPT-2 BPE.
- Tenere `torch.compile` come opzione avanzata, non come default.
"""

import torch

from .device import get_default_device
from .model import LanguageModel
from .tokenizer import DEFAULT_ENCODING_NAME, decode, encode


def load_model_from_checkpoint(checkpoint_path, device=None, compile_model=False):
    device = device or get_default_device()
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = LanguageModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                "torch.compile non è disponibile in questa versione di PyTorch."
            )
        model = torch.compile(model)

    return model, checkpoint


def generate_text_from_checkpoint(
    checkpoint_path,
    prompt_text,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    device=None,
    compile_model=False,
):
    device = device or get_default_device()

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens non può essere negativo.")

    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        compile_model=compile_model,
    )
    tokenizer_config = checkpoint.get(
        "tokenizer_config",
        {"encoding_name": DEFAULT_ENCODING_NAME},
    )
    encoding_name = tokenizer_config.get("encoding_name", DEFAULT_ENCODING_NAME)

    prompt_ids = encode(prompt_text, encoding_name=encoding_name)
    if len(prompt_ids) == 0:
        raise ValueError("prompt_text deve produrre almeno un token.")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    generated_text = decode(
        generated_ids[0].detach().cpu().tolist(),
        encoding_name=encoding_name,
    )

    return generated_text, checkpoint


def generate_samples_from_checkpoint(
    checkpoint_path,
    prompt_text,
    max_new_tokens,
    num_samples,
    temperature=1.0,
    top_k=None,
    device=None,
    compile_model=False,
):
    if num_samples < 1:
        raise ValueError("num_samples deve essere almeno 1.")

    samples = []
    checkpoint = None

    for _ in range(num_samples):
        generated_text, checkpoint = generate_text_from_checkpoint(
            checkpoint_path=checkpoint_path,
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
            compile_model=compile_model,
        )
        samples.append(generated_text)

    return samples, checkpoint
