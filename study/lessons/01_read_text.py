"""
Changes compared with the previous file:
- Lessons 01-03 are intentionally standalone: there is no project module to
  import yet.
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
