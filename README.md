# LearnGPT

LearnGPT is a study-first GPT project inspired by Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT). Its goal is to decompose the
main ideas behind GPT-style decoder-only Transformers into small, precise
lessons before bringing them back together in a clean final PyTorch project.

The repository is intentionally more explicit than nanoGPT. Instead of hiding
the model behind compact production code, each concept is introduced step by
step: tokenization, batches, embeddings, causal self-attention, multi-head
attention, Transformer blocks, optimization, checkpointing, and generation.

## About

Tags:
`gpt`, `transformer`, `nanogpt`, `pytorch`, `decoder-only-transformer`,
`language-model`, `llm`, `fineweb-edu`, `gpt-2-bpe`, `tokenization`,
`causal-self-attention`, `multi-head-attention`, `layernorm`, `gelu`,
`adamw`, `gradient-accumulation`, `checkpointing`, `text-generation`,
`apple-silicon`, `mps`, `cuda`, `cpu`, `mixed-precision`, `torch-compile`,
`machine-learning`, `deep-learning`, `education`, `study-project`.

The course is available in two Markdown versions:

- [course_en.md](course_en.md)
- [course_it.md](course_it.md)

## What This Project Contains

- A lesson-by-lesson study path for building a GPT-like language model.
- Reproducible lesson snapshots under `study/snapshots/`.
- A clean final project under `final_project/`.
- GPT-2 BPE tokenization with `tiktoken`.
- FineWeb-Edu data preparation for local training.
- Reproducible randomized experimental subsets derived from processed data.
- Memmapped `train.bin` / `val.bin` loading for large local datasets.
- CPU, CUDA, and Apple Silicon MPS device selection.
- AdamW optimizer groups, gradient accumulation, learning-rate scheduling,
  gradient clipping, GPT-style initialization, atomic best/latest checkpoints,
  resume support, context-sensitivity quality gates, optional mixed precision,
  and optional `torch.compile`.

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
    prepare_subset.py
    batching.py
    device.py
    model.py
    training.py
    checkpoint.py
    generate.py
    quality.py
    requirements.txt

  tools/
    validate_learngpt.py

  tests/
    test_final_project.py
```

`study/` is for learning. `final_project/` is the clean current version of the
project. Datasets, checkpoints, and generated model files are intentionally not
tracked by Git.

## Quick Start: Study The Course

Create and activate a local virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```bash
python -m pip install -r final_project/requirements.txt
```

Validate the repository structure:

```bash
python -B tools/validate_learngpt.py
```

Run the final-project regression tests:

```bash
python -B -m unittest discover -s tests -v
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

## PyTorch Installation By Backend

For most local study runs, `python -m pip install -r final_project/requirements.txt`
is enough. For real training, install the PyTorch build that matches your
hardware first, then install the remaining project dependencies.

Use the official [PyTorch install selector](https://pytorch.org/get-started/locally/)
when you need the latest backend-specific command.

<details>
<summary>Apple Silicon MPS PyTorch install</summary>

On macOS with Apple Silicon, install the standard macOS wheel:

```bash
python -m pip install torch
python -m pip install datasets numpy tiktoken
```

Verify MPS from a normal Terminal session:

```bash
python -c "import torch; print(torch.backends.mps.is_built(), torch.backends.mps.is_available()); print(torch.ones(1, device='mps'))"
```

Expected result:

```text
True True
tensor(..., device='mps:0')
```

If this prints `True False` inside a managed or sandboxed shell, rerun the same
check from a normal Terminal. Sandboxed processes can be blocked from creating a
Metal device even when the Mac supports MPS.

</details>

<details>
<summary>NVIDIA CUDA PyTorch install</summary>

Choose the CUDA wheel that matches your machine from the PyTorch install
selector. For example, with CUDA 12.8:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install datasets numpy tiktoken
```

Verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda device')"
```

</details>

<details>
<summary>CPU-only PyTorch install</summary>

Use this when you do not have a GPU backend available:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install datasets numpy tiktoken
```

Verify CPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.ones(1).device)"
```

</details>

## Quick Start: Local Training

The final project trains on
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) using
the GPT-2 BPE tokenizer. The dataset is not committed to the repository.

Prepare about 10 GB of tokenized data:

```bash
python -B final_project/prepare_data.py \
  --target-gb 10 \
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

Choose one training backend. Expand only the section that matches your machine.

<details>
<summary>Apple Silicon MPS training</summary>

Check that PyTorch can see MPS:

```bash
python -c "import torch; print(torch.backends.mps.is_built(), torch.backends.mps.is_available())"
```

Run this check from a normal Terminal session. Some managed or sandboxed shells
can be blocked from creating a Metal device and may print `True False` even
when MPS works normally outside the sandbox.

Smoke test MPS with one tiny training step:

```bash
python -m final_project.training \
  --device mps \
  --data-dir data/processed/fineweb_edu \
  --checkpoint-path /tmp/learngpt-mps-smoke.pt \
  --context-size 8 \
  --embedding-size 16 \
  --num-heads 4 \
  --num-transformer-blocks 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --training-steps 1 \
  --eval-interval 1 \
  --eval-batches 1 \
  --base-learning-rate 1e-4 \
  --min-learning-rate 1e-5 \
  --warmup-steps 0 \
  --decay-steps 1
```

Generate from the smoke-test checkpoint:

```bash
python -m final_project.generate \
  --device mps \
  --checkpoint-path /tmp/learngpt-mps-smoke.pt \
  --prompt "Once upon" \
  --max-new-tokens 8 \
  --temperature 1.0 \
  --top-k 20
```

### Controlled real training on an 8 GB Apple Silicon Mac

Do not train the 17.7M model for 45,000 steps directly on the complete 10 GiB
corpus. That budget sees only about 7% of its 5.3B training tokens and can
converge to an almost context-free frequency model. Keep the full corpus as the
canonical source and create a separate, reproducible 1 GiB experiment:

```bash
.venv/bin/python -B -m final_project.prepare_subset \
  --source-data-dir data/processed/fineweb_edu \
  --output-dir data/processed/fineweb_edu_experiment_1g \
  --target-gb 1 \
  --validation-ratio 0.01 \
  --seed 1337 \
  --chunk-tokens 65536
```

The command only reads `fineweb_edu` and writes a new 1 GiB local dataset. It
selects non-overlapping token chunks in a deterministic order, so the same
source dataset and seed reproduce the same experiment.

First run a 10,000-step quality probe. The architecture stays small enough for
the Mac, while the optimization is more conservative and the gate stops a
run that stops using its context:

```bash
caffeinate -i .venv/bin/python -B -m final_project.training \
  --device mps \
  --data-dir data/processed/fineweb_edu_experiment_1g \
  --checkpoint-path checkpoints/learngpt-mps-18m-experiment-1g.pt \
  --encoding-name gpt2 \
  --seed 1337 \
  --context-size 256 \
  --embedding-size 256 \
  --num-heads 4 \
  --num-transformer-blocks 6 \
  --dropout 0.0 \
  --use-scaled-dot-product-attention \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --training-steps 10000 \
  --eval-interval 250 \
  --eval-batches 20 \
  --base-learning-rate 1e-4 \
  --min-learning-rate 1e-5 \
  --warmup-steps 1000 \
  --decay-steps 80000 \
  --weight-decay 0.05 \
  --gradient-clip 1.0 \
  --context-sensitivity-contexts 8 \
  --min-context-js-divergence 1e-4 \
  --stop-on-low-context-sensitivity
```

`context_js` measures how differently the model distributes next-token
probability across eight fixed validation contexts. It is an anti-collapse
signal, not a text-quality score: a value below `1e-4` stops the run after
writing its latest checkpoint. The original collapsed checkpoint measured about
`2e-6`; a fresh model measured about `2e-2`.

With GPT-style initialization, the first loss should be close to `ln(50257)`,
approximately `10.82`, rather than tens or hundreds. After the probe, generate
from the best checkpoint and inspect several prompts before extending it.

Training writes two atomic checkpoints:

```text
checkpoints/learngpt-mps-18m-experiment-1g.pt         # best validation loss
checkpoints/learngpt-mps-18m-experiment-1g-latest.pt  # latest evaluated step
```

If the probe keeps passing the context gate and its samples improve, extend the
same run to the planned 80,000-step total target:

```bash
caffeinate -i .venv/bin/python -B -m final_project.training \
  --device mps \
  --data-dir data/processed/fineweb_edu_experiment_1g \
  --checkpoint-path checkpoints/learngpt-mps-18m-experiment-1g.pt \
  --resume-checkpoint-path checkpoints/learngpt-mps-18m-experiment-1g-latest.pt \
  --training-steps 80000
```

Resume restores the saved model, tokenizer, optimizer, random-number state,
architecture, and training configuration. Pass a larger `--training-steps`
only when intentionally extending the total target.

Generate:

```bash
.venv/bin/python -B -m final_project.generate \
  --device mps \
  --checkpoint-path checkpoints/learngpt-mps-18m-experiment-1g.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 120 \
  --temperature 0.9 \
  --top-k 50
```

If MPS is not available in the current PyTorch runtime, fix the PyTorch/macOS
environment before training with `--device mps`.

</details>

<details>
<summary>NVIDIA CUDA training</summary>

Check that PyTorch can see CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda device')"
```

Train:

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

Resume:

```bash
python -m final_project.training \
  --device cuda \
  --data-dir data/processed/fineweb_edu \
  --checkpoint-path checkpoints/learngpt-cuda.pt \
  --resume-checkpoint-path checkpoints/learngpt-cuda-latest.pt \
  --mixed-precision \
  --precision-dtype float16
```

Generate:

```bash
python -m final_project.generate \
  --device cuda \
  --checkpoint-path checkpoints/learngpt-cuda.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 120 \
  --temperature 0.9 \
  --top-k 50
```

</details>

<details>
<summary>CPU fallback training</summary>

CPU always works when PyTorch is installed, but it is much slower for real
training. Use this mainly for smoke tests or very small runs.

Train:

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

Resume:

```bash
python -m final_project.training \
  --device cpu \
  --data-dir data/processed/fineweb_edu \
  --checkpoint-path checkpoints/learngpt-cpu.pt \
  --resume-checkpoint-path checkpoints/learngpt-cpu-latest.pt
```

Generate:

```bash
python -m final_project.generate \
  --device cpu \
  --checkpoint-path checkpoints/learngpt-cpu.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 120 \
  --temperature 0.9 \
  --top-k 50
```

</details>

The training CLI prints the selected device, dataset size, model config,
training config, validation loss, learning rate, gradient norm, tokens per
second, and estimated remaining time.

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
