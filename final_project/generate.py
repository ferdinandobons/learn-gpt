"""
Changes compared with the previous files:
- This module belongs to the cleaned final LearnGPT project.
- It uses the English public project layout and the FineWeb-Edu/BPE pipeline.

File purpose:
- Generate text from a saved checkpoint.
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
                "torch.compile is not available in this version of PyTorch."
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
        raise ValueError("max_new_tokens cannot be negative.")

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
        raise ValueError("prompt_text must produce at least one token.")

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
        raise ValueError("num_samples must be at least 1.")

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
