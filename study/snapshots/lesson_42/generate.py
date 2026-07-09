"""
Changes compared with the previous files:
- This module belongs to the cleaned final LearnGPT project.
- It uses the English public project layout and the FineWeb-Edu/BPE pipeline.

File purpose:
- Generate text from a saved checkpoint.
"""

import argparse
from pathlib import Path

import torch

from .device import get_default_device, resolve_device
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text with a trained LearnGPT checkpoint.",
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--prompt", default="The")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--no-top-k", action="store_true")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--compile-model", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    top_k = None if args.no_top_k else args.top_k

    samples, _ = generate_samples_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        prompt_text=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_k=top_k,
        device=device,
        compile_model=args.compile_model,
    )

    for index, sample in enumerate(samples, start=1):
        if args.num_samples > 1:
            print(f"--- sample {index} ---")
        print(sample)


if __name__ == "__main__":
    main()
