"""
Differenza rispetto al file precedente:
- Prima usavamo PyTorch dentro il batch, ma senza fermarci a guardare cosa sia
  un tensore.
- Qui facciamo un controllo esplicito di PyTorch e osserviamo alcune operazioni
  minime sui tensori.

Scopo del file:
- Verificare che PyTorch sia installato.
- Capire che un tensore è una tabella di numeri con una forma precisa.
- Preparare il terreno per il primo modello neurale.
"""

import torch


def main():
    print("Versione PyTorch:")
    print(torch.__version__)
    print()

    token_ids = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]

    tensor = torch.tensor(token_ids)

    print("Tensore:")
    print(tensor)
    print()

    print("Forma del tensore:")
    print(tensor.shape)
    print()

    print("Prima riga:")
    print(tensor[0])
    print()

    print("Seconda colonna:")
    print(tensor[:, 1])
    print()

    print("Tipo dei dati contenuti:")
    print(tensor.dtype)


if __name__ == "__main__":
    main()
