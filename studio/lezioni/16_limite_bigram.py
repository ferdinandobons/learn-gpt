"""
Differenza rispetto al file precedente:
- Prima generavamo testo con il modello bigram.
- Qui mostriamo un limite importante: il bigram guarda solo l'ultimo token.

Scopo del file:
- Dimostrare che due prompt diversi ma con lo stesso ultimo carattere producono
  gli stessi punteggi per il prossimo carattere.
- Capire perché il bigram non basta per costruire un GPT vero.
- Preparare il passaggio a un modello che usa davvero il contesto.
"""

from pathlib import Path
import random
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"

sys.path.append(str(PROJECT_DIR))

from studio.snapshot.lezione_16.batching import create_batch
from studio.snapshot.lezione_16.model import LanguageModel
from studio.snapshot.lezione_16.tokenizer import create_vocabulary, decode, encode

CONTEXT_SIZE = 8
BATCH_SIZE = 32
TRAINING_STEPS = 300
LEARNING_RATE = 0.01


def train_model(training_data, vocabulary_size):
    model = LanguageModel(vocabulary_size=vocabulary_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(TRAINING_STEPS):
        input_tensor, target_tensor = create_batch(
            data=training_data,
            batch_size=BATCH_SIZE,
            context_size=CONTEXT_SIZE,
        )

        _, loss = model(input_tensor, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, loss


def show_prediction(model, prompt, char_to_id, id_to_char):
    prompt_ids = encode(prompt, char_to_id)
    input_ids = torch.tensor([prompt_ids])

    logits = model(input_ids)
    last_token_logits = logits[:, -1, :]
    predicted_token_id = torch.argmax(last_token_logits, dim=-1).item()

    print("Prompt:", repr(prompt))
    print("Ultimo carattere:", repr(prompt[-1]))
    print("Token previsto con punteggio massimo:", repr(decode([predicted_token_id], id_to_char)))
    print()

    return last_token_logits


def main():
    random.seed(42)
    torch.manual_seed(42)

    text = DATASET_PATH.read_text(encoding="utf-8")

    char_to_id, id_to_char = create_vocabulary(text)
    vocabulary_size = len(char_to_id)

    token_ids = encode(text, char_to_id)

    split_index = int(len(token_ids) * 0.9)
    training_data = token_ids[:split_index]

    model, loss = train_model(
        training_data=training_data,
        vocabulary_size=vocabulary_size,
    )

    print("Loss finale dopo breve training:", loss.item())
    print()

    logits_nel = show_prediction(
        model=model,
        prompt="Nel",
        char_to_id=char_to_id,
        id_to_char=id_to_char,
    )

    logits_sol = show_prediction(
        model=model,
        prompt="sol",
        char_to_id=char_to_id,
        id_to_char=id_to_char,
    )

    logits_nea = show_prediction(
        model=model,
        prompt="Nea",
        char_to_id=char_to_id,
        id_to_char=id_to_char,
    )

    print("`Nel` e `sol` finiscono entrambi con `l`.")
    print("I loro punteggi finali sono uguali?")
    print(torch.allclose(logits_nel, logits_sol))
    print()

    print("`Nel` e `Nea` finiscono con caratteri diversi.")
    print("I loro punteggi finali sono uguali?")
    print(torch.allclose(logits_nel, logits_nea))


if __name__ == "__main__":
    main()
