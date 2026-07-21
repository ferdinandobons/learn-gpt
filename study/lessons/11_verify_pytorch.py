"""
Changes compared with the previous file:
- This lesson script uses the English project layout and imports lesson-specific
  snapshot code.
- It belongs to lesson 11 of the guided LearnGPT path.

File purpose:
- Run the lesson example in a reproducible way.
- Print the relevant intermediate values, tensor shapes, losses, or generated
  text for inspection.
"""

import torch


def main():
    print("PyTorch version:")
    print(torch.__version__)
    print()

    token_ids = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]

    tensor = torch.tensor(token_ids)

    print("Tensor:")
    print(tensor)
    print()

    print("Tensor shape:")
    print(tensor.shape)
    print()

    print("First line:")
    print(tensor[0])
    print()

    print("Second column:")
    print(tensor[:, 1])
    print()

    print("Contained data type:")
    print(tensor.dtype)


if __name__ == "__main__":
    main()
