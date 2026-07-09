"""
Differenza rispetto ai file precedenti:
- Prima la generazione da checkpoint usava sempre il campionamento standard.
- Qui esponiamo `temperature`, `top_k` e `num_samples`.

Scopo del file:
- Ricostruire un `LanguageModel` partendo dal checkpoint.
- Generare testo usando il modello ricaricato, senza eseguire training.
- Produrre uno o più testi con parametri di campionamento espliciti.
"""

import torch

from .model import LanguageModel
from .tokenizer import decode, encode


def load_model_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    model = LanguageModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def generate_text_from_checkpoint(
    checkpoint_path,
    prompt_text,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
):
    model, checkpoint = load_model_from_checkpoint(checkpoint_path)
    char_to_id = checkpoint["char_to_id"]
    id_to_char = checkpoint["id_to_char"]

    unknown_chars = sorted(set(prompt_text) - set(char_to_id))
    if unknown_chars:
        raise ValueError(f"Il prompt contiene caratteri non nel vocabolario: {unknown_chars}")

    prompt_ids = encode(prompt_text, char_to_id)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    generated_text = decode(generated_ids[0].tolist(), id_to_char)

    return generated_text, checkpoint


def generate_samples_from_checkpoint(
    checkpoint_path,
    prompt_text,
    max_new_tokens,
    num_samples,
    temperature=1.0,
    top_k=None,
):
    samples = []
    checkpoint = None

    for _ in range(num_samples):
        generated_text, checkpoint = generate_text_from_checkpoint(
            checkpoint_path=checkpoint_path,
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        samples.append(generated_text)

    return samples, checkpoint
