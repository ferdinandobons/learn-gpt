# LearnGPT

LearnGPT is a study-first GPT project inspired by Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT). Its goal is to decompose the
main ideas behind GPT-style decoder-only Transformers into small, precise
lessons before bringing them back together in a clean final PyTorch project.

The repository is intentionally more explicit than nanoGPT. Instead of hiding
the model behind compact production code, each concept is introduced step by
step: tokenization, batches, embeddings, causal self-attention, multi-head
attention, Transformer blocks, optimization, checkpointing, and generation.

The course is available in two Markdown versions:

- [course_en.md](course_en.md)
- [course_it.md](course_it.md)

## What This Project Contains

- A lesson-by-lesson study path for building a GPT-like language model.
- Reproducible lesson snapshots under `study/snapshots/`.
- A clean final project under `final_project/`.
- GPT-2 BPE tokenization with `tiktoken`.
- FineWeb-Edu data preparation for local training.
- Memmapped `train.bin` / `val.bin` loading for large local datasets.
- CPU, CUDA, and Apple Silicon MPS device selection.
- AdamW optimizer groups, gradient accumulation, learning-rate scheduling,
  gradient clipping, checkpoints, resume support, optional mixed precision, and
  optional `torch.compile`.

## Project Layout

```text
LearnGPT/
  README.md
  course_en.md
  course_it.md

  data/
    README.md
    raw/                 # ignored by Git
    processed/           # ignored by Git

  study/
    lessons/             # numbered lesson scripts
    snapshots/           # lesson-specific project snapshots

  final_project/
    config.py
    tokenizer.py
    prepare_data.py
    batching.py
    device.py
    model.py
    training.py
    checkpoint.py
    generate.py
    requirements.txt

  tools/
    validate_learngpt.py
```

`study/` is for learning. `final_project/` is the clean current version of the
project. Datasets, checkpoints, and generated model files are intentionally not
tracked by Git.

## Quick Start: Study The Course

Install dependencies:

```bash
python -m pip install -r final_project/requirements.txt
```

Validate the repository structure:

```bash
python -B tools/validate_learngpt.py
```

Run a specific lesson:

```bash
python -B study/lessons/01_read_text.py
```

Run the final lesson smoke test:

```bash
python -B study/lessons/42_final_project.py
```

Read the course while running the numbered lesson scripts. The Markdown files
explain the new code introduced by each lesson, while `study/snapshots/` keeps
the complete code state for that lesson.

## Quick Start: Local Training

The final project trains on
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) using
the GPT-2 BPE tokenizer. The dataset is not committed to the repository.

Prepare about 5 GB of tokenized data:

```bash
python -B final_project/prepare_data.py \
  --target-gb 5 \
  --output-dir data/processed/fineweb_edu
```

This creates:

```text
data/processed/fineweb_edu/
  train.bin
  val.bin
  meta.json
```

Validate that the local data exists:

```bash
python -B tools/validate_learngpt.py --require-data
```

Choose one training backend.

Apple Silicon MPS check:

```bash
python -c "import torch; print(torch.backends.mps.is_built(), torch.backends.mps.is_available())"
```

NVIDIA CUDA check:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda device')"
```

CPU always works when PyTorch is installed, but it is much slower for real
training.

Train on Apple Silicon MPS:

```bash
python -m final_project.training \
  --device mps \
  --data-dir data/processed/fineweb_edu \
  --checkpoint-path checkpoints/learngpt-mps.pt \
  --context-size 128 \
  --embedding-size 256 \
  --num-heads 4 \
  --num-transformer-blocks 4 \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --training-steps 1000 \
  --eval-interval 100 \
  --eval-batches 10
```

Train on NVIDIA CUDA:

```bash
python -m final_project.training \
  --device cuda \
  --data-dir data/processed/fineweb_edu \
  --checkpoint-path checkpoints/learngpt-cuda.pt \
  --context-size 128 \
  --embedding-size 256 \
  --num-heads 4 \
  --num-transformer-blocks 4 \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --training-steps 1000 \
  --eval-interval 100 \
  --eval-batches 10 \
  --mixed-precision \
  --precision-dtype float16
```

Train on CPU as a slow fallback:

```bash
python -m final_project.training \
  --device cpu \
  --data-dir data/processed/fineweb_edu \
  --checkpoint-path checkpoints/learngpt-cpu.pt \
  --context-size 128 \
  --embedding-size 256 \
  --num-heads 4 \
  --num-transformer-blocks 4 \
  --batch-size 2 \
  --gradient-accumulation-steps 1 \
  --training-steps 100 \
  --eval-interval 20 \
  --eval-batches 5
```

The training CLI prints the selected device, dataset size, model config,
training config, validation loss, learning rate, gradient norm, tokens per
second, and estimated remaining time.

Resume from a checkpoint by matching the device and checkpoint path you used:

```bash
python -m final_project.training \
  --device mps \
  --data-dir data/processed/fineweb_edu \
  --checkpoint-path checkpoints/learngpt-mps.pt \
  --resume-checkpoint-path checkpoints/learngpt-mps.pt
```

Generate text from a checkpoint:

```bash
python -m final_project.generate \
  --device mps \
  --checkpoint-path checkpoints/learngpt-mps.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 120 \
  --temperature 0.9 \
  --top-k 50
```

Use `--device cuda` with a CUDA checkpoint or `--device cpu` with a CPU
checkpoint. If MPS is not available in the current PyTorch runtime, fix the
PyTorch/macOS environment before training with `--device mps`.

## Publishing Checkpoints

Checkpoints can become large, so they are ignored by Git. If you want to share a
trained model publicly, prefer GitHub Release assets or an external model host
instead of committing `.pt`, `.pth`, or `.ckpt` files to the repository.

## Relationship To nanoGPT

LearnGPT follows the broad nanoGPT direction:

- decoder-only Transformer architecture
- learned token and position embeddings
- causal self-attention
- multi-head attention
- pre-LayerNorm Transformer blocks
- residual connections
- MLP/feed-forward blocks with GELU
- AdamW training
- checkpointing and autoregressive generation

The main difference is educational structure. nanoGPT is compact and optimized
for people who already know the moving parts. LearnGPT keeps names and steps
more verbose so the reader can inspect how each tensor shape and training step
fits into the complete model.
