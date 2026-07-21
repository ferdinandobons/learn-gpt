"""
Changes compared with the previous files:
- This module belongs to the cleaned final LearnGPT project.
- It uses the English public project layout and the FineWeb-Edu/BPE pipeline.

File purpose:
- Generate text from a saved checkpoint.
"""

import argparse
from pathlib import Path
import sys

import torch

from .checkpoint import canonicalize_model_state_dict, load_checkpoint_payload
from .config import ModelConfig
from .device import get_default_device, resolve_device
from .model import LanguageModel
from .tokenizer import DEFAULT_ENCODING_NAME, decode, encode


def configure_utf8_stdout():
    """Make arbitrary BPE output printable on Windows and redirected consoles."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def load_model_from_checkpoint(checkpoint_path, device=None, compile_model=False):
    device = device or get_default_device()
    checkpoint = load_checkpoint_payload(checkpoint_path, device="cpu")

    model_config = ModelConfig.from_checkpoint_dict(checkpoint["model_config"])
    model = LanguageModel(**model_config.to_model_kwargs())
    model.load_state_dict(
        canonicalize_model_state_dict(checkpoint["model_state_dict"])
    )
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
    seed=None,
):
    device = device or get_default_device()

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative.")
    if seed is not None and seed < 0:
        raise ValueError("seed cannot be negative.")

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
    if seed is not None:
        # Seed sampling after model construction/loading so implementation
        # details in the loader cannot consume part of the sampling stream.
        torch.manual_seed(seed)

    with torch.inference_mode():
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
    seed=None,
):
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    if seed is not None and seed < 0:
        raise ValueError("seed cannot be negative.")

    device = device or get_default_device()
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
    if seed is not None:
        # One seed controls the complete ordered sample set and is independent
        # of random numbers consumed while reconstructing the model.
        torch.manual_seed(seed)

    samples = []
    for _ in range(num_samples):
        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional sampling seed for reproducible generated text.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--compile-model", action="store_true")

    return parser.parse_args()


def main():
    configure_utf8_stdout()
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
        seed=args.seed,
    )

    for index, sample in enumerate(samples, start=1):
        if args.num_samples > 1:
            print(f"--- sample {index} ---")
        print(sample)


if __name__ == "__main__":
    main()
