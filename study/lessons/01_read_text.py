"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 01 of the guided LearnGPT path.

File purpose:
- Run the lesson example in a reproducible way.
- Print the relevant intermediate values, tensor shapes, losses, or generated
  text for inspection.
"""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "study_sample.txt"


def main():
    text = DATASET_PATH.read_text(encoding="utf-8")

    print("File read:", DATASET_PATH)
    print("Number of characters:", len(text))
    print()
    print("First 500 characters:")
    print(text[:500])


if __name__ == "__main__":
    main()
