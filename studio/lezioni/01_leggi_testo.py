"""
Differenza rispetto ai file precedenti:
- Questo è il primo file del percorso, quindi non confronta ancora codice
  precedente.

Scopo del file:
- Leggere `data/raw/fineweb_edu_sample.txt`.
- Stampare quanti caratteri contiene.
- Mostrare l'inizio del testo per verificare che il file venga letto bene.
"""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "fineweb_edu_sample.txt"


def main():
    text = DATASET_PATH.read_text(encoding="utf-8")

    print("File letto:", DATASET_PATH)
    print("Numero caratteri:", len(text))
    print()
    print("Primi 500 caratteri:")
    print(text[:500])


if __name__ == "__main__":
    main()
