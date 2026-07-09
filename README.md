# LearnGPT

`LearnGPT` is a didactic GPT project built step by step. The goal is to make
the main pieces of a small decoder-only language model understandable before
using them together in a final runnable project.

The full course document is [corso.md](corso.md). It contains the lesson-by-
lesson explanations, code snippets, diagrams, and the current learning path.

## Project Direction

The project broadly follows the architecture direction of nanoGPT:

- token and position embeddings
- causal self-attention
- multi-head attention
- Transformer blocks
- LayerNorm
- feed-forward / MLP layers
- residual connections
- AdamW training
- checkpointing
- text generation with sampling controls

The code intentionally stays more explicit than nanoGPT so each intermediate
step is easier to study.

## Data

The project uses FineWeb-Edu as the main dataset, but the repository does not
include datasets or processed binary files.

Local processed data should live in:

```text
data/processed/fineweb_edu/
  train.bin
  val.bin
  meta.json
```

The final project reads `train.bin` and `val.bin` with NumPy memmap so the
dataset does not need to be loaded fully into RAM. See [data/README.md](data/README.md)
for local dataset preparation.

## Directory Layout

```text
LearnGPT/
  corso.md
  README.md

  data/
    raw/
    processed/

  studio/
    lezioni/
    snapshot/

  progetto_finale/
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

  strumenti/
    validate_learngpt.py
```

`studio/lezioni/` contains numbered lesson scripts.

`studio/snapshot/` contains lesson-specific code snapshots, so old lessons can
remain reproducible while the final project evolves.

`progetto_finale/` contains the current final version of the project.

## Final Project Components

- `tokenizer.py`: GPT-2 BPE tokenizer wrapper using `tiktoken`
- `config.py`: model, training, and generation configuration dataclasses
- `prepare_data.py`: streams FineWeb-Edu and writes `train.bin` / `val.bin`
- `batching.py`: creates training batches from memmapped token files
- `device.py`: chooses CPU, CUDA, or Apple MPS
- `model.py`: decoder-only Transformer language model
- `training.py`: optimizer, scheduler, loss estimation, and training loop
- `checkpoint.py`: checkpoint save/load helpers
- `generate.py`: text generation from a saved checkpoint

## Common Commands

Install dependencies:

```bash
python -m pip install -r progetto_finale/requirements.txt
```

Validate the course structure:

```bash
python -B strumenti/validate_learngpt.py
```

Validate the local dataset too:

```bash
python -B strumenti/validate_learngpt.py --require-data
```

Run the final smoke test:

```bash
python -B studio/lezioni/42_final_project.py
```

Prepare FineWeb-Edu again:

```bash
python -B progetto_finale/prepare_data.py \
  --target-gb 5 \
  --output-dir data/processed/fineweb_edu \
  --overwrite
```

## Current Status

The final project uses:

- FineWeb-Edu processed data
- GPT-2 BPE tokenization
- memmapped binary token files
- CPU/CUDA/MPS device handling
- optimizer parameter groups
- gradient accumulation
- resume-capable checkpoints
- optional `torch.compile`
- optional mixed precision
- checkpoint-based generation

The early lessons still start with a deliberately rough character-level
tokenizer to make the idea of tokenization concrete. The course then moves
toward the BPE/FineWeb pipeline used by the final project.
