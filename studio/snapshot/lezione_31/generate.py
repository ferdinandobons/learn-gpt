"""
Differenza rispetto ai file precedenti:
- Prima ricaricavamo un checkpoint solo per verificare che i logits fossero
  identici.
- Qui aggiungiamo funzioni dedicate alla generazione da checkpoint.

Scopo del file:
- Ricostruire un `LanguageModel` partendo dal checkpoint.
- Generare testo usando il modello ricaricato, senza eseguire training.
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


def generate_text_from_checkpoint(checkpoint_path, prompt_text, max_new_tokens):
    model, checkpoint = load_model_from_checkpoint(checkpoint_path)
    char_to_id = checkpoint["char_to_id"]
    id_to_char = checkpoint["id_to_char"]

    unknown_chars = sorted(set(prompt_text) - set(char_to_id))
    if unknown_chars:
        raise ValueError(f"Il prompt contiene caratteri non nel vocabolario: {unknown_chars}")

    prompt_ids = encode(prompt_text, char_to_id)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)

    with torch.no_grad():
        generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)

    generated_text = decode(generated_ids[0].tolist(), id_to_char)

    return generated_text, checkpoint
