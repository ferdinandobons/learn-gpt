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
`fused-attention`,
`machine-learning`, `deep-learning`, `education`, `study-project`.

The course is maintained as one English guide:

- [course_en.md](course_en.md)
- [How to train runbook](docs/FINAL_TRAINING_RUNBOOK.md) for the canonical
  macOS/MPS and Windows/CUDA workflow.
- [CUDA training optimizations](CUDA_TRAINING_OPTIMIZATIONS.md) for measured
  fused-attention, batching, logging, and VRAM tradeoffs on NVIDIA hardware.
- [Video series guide](docs/VIDEO_SERIES_GUIDE.md) for teaching the 42
  checkpoints and the final experiment.
- [Model memory and training limits](docs/MODEL_MEMORY_AND_TRAINING_LIMITS.md)
  for understanding how parameters, activations, context, and optimizer state
  affect an 8 GB local training run.

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
  resume support, chunked vocabulary projection, MPS gradient-integrity checks,
  target-aware context diagnostics, optional mixed precision, fused multi-head
  QKV attention, lightweight progress logging, and optional `torch.compile`.

## Project Layout

```text
LearnGPT/
  README.md
  course_en.md
  CUDA_TRAINING_OPTIMIZATIONS.md

  docs/
    FINAL_TRAINING_RUNBOOK.md
    MODEL_MEMORY_AND_TRAINING_LIMITS.md
    VIDEO_SERIES_GUIDE.md
    training_workflow.json
    verified_runs/

  data/
    README.md
    study_sample.txt     # tracked, small, and used by lessons
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
    requirements-common.txt

  tools/
    validate_learngpt.py

  tests/
    test_final_project.py
```

`study/` is for learning. `final_project/` is the clean current version of the
project. Datasets, checkpoints, and generated model files are intentionally not
tracked by Git.

## Quick Start: Study The Course

Use Python 3.12 or newer; Python 3.13 is the recommended and CI-tested version.
Create a local virtual environment on macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```bash
python -m pip install -r final_project/requirements.txt
```

On Windows PowerShell, use the environment interpreter directly. For ordinary
CPU study runs, install the common dependencies and the CPU wheel explicitly:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r final_project\requirements-common.txt
.\.venv\Scripts\python.exe -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
```

If this machine will train with NVIDIA CUDA, install the CUDA wheel from the
backend section below instead of the CPU-wheel command.

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

Run the complete clean-clone teaching gate:

```bash
python -B tools/run_all_lessons.py
```

Read the course while running the numbered lesson scripts. The Markdown files
explain the new code introduced by each lesson, while `study/snapshots/` keeps
the complete code state for that lesson. Lessons use the tracked
`data/study_sample.txt`; the 10 GiB corpus is required only for real training.
GitHub Actions repeats the validator, regression suite, and all 42 lessons on
both Linux and Windows.

## PyTorch Installation By Backend

For most local study runs, `python -m pip install -r final_project/requirements.txt`
is enough. For real training, install the PyTorch build that matches your
hardware first, then install the remaining project dependencies. The
requirements file pins the exact versions used by the verified run.

Use the official [PyTorch install selector](https://pytorch.org/get-started/locally/)
when you need the latest backend-specific command.

<details>
<summary>Apple Silicon MPS PyTorch install</summary>

On macOS with Apple Silicon, install the tested standard macOS wheel and the
remaining pinned dependencies:

```bash
python -m pip install -r final_project/requirements.txt
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
selector. The controlled Windows profile pins PyTorch 2.12.1 and uses the
CUDA 12.6 wheel:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch
.\.venv\Scripts\python.exe -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r final_project\requirements-common.txt
```

The uninstall makes a CPU-to-CUDA transition unambiguous: pip otherwise treats
an already installed CPU build with the same version number as satisfied.

Verify CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda device')"
```

See the [how to train runbook](docs/FINAL_TRAINING_RUNBOOK.md) if the
official selector requires a different wheel index for the installed driver.

</details>

<details>
<summary>CPU-only PyTorch install</summary>

Use this when you do not have a GPU backend available:

```bash
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r final_project/requirements-common.txt
```

Verify CPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.ones(1).device)"
```

</details>

## How to Train: Copy-Ready Quick Start

The final project trains on
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) using
the GPT-2 BPE tokenizer. The dataset is not committed to the repository.

The canonical, fully explained workflow is the
[how to train runbook](docs/FINAL_TRAINING_RUNBOOK.md). The commands below
are a compact reference; keep the runbook open for setup, monitoring, resume,
Windows PowerShell, and troubleshooting.

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

Validate the canonical local data:

```bash
python -B tools/validate_learngpt.py \
  --training-data-dir data/processed/fineweb_edu
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
  --overwrite-checkpoints \
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

Keep the complete processed corpus as the canonical source and create a
separate, reproducible 1 GiB experiment for this compute-bounded run:

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

#### Why the previous MPS run failed

The failure was in the backward pass, not in FineWeb-Edu or tokenization. Two
MPS problems contributed. The monolithic `256 -> 50257` vocabulary projection
returned a wrong hidden-state gradient direction even when its forward loss was
correct. Separately, `optimizer.zero_grad(set_to_none=True)` made MPS allocate
new leaf-gradient buffers and intermittently produced enormous gradients.
Clipping those gradients to `1.0` limited their size but could not repair their
direction, so the model drifted toward nearly the same high-frequency
distribution for every prompt.

The corrected path now:

- projects the 50,257-token vocabulary in chunks of at most 32,768 entries;
- allocates persistent MPS gradient buffers and clears them in place;
- performs a discarded warm-up backward pass before training;
- requires two identical MPS backward passes to agree with each other and with
  a CPU reference before the first optimizer update;
- rejects a raw gradient norm above `100`, retries the exact same batches up to
  three times, and stops without applying an update if every attempt fails.

Do not resume one of the checkpoints produced by the affected training path.
Gradient clipping hid the corruption inside their learned weights, so there is
no reliable way to repair them. Start from random initialization with a new,
previously unused checkpoint path.

#### Complete 45,000-step run

This is the single controlled command for the 17.7M-parameter model. At batch
size 4, context 256, and eight accumulated micro-batches, it processes 8,192
tokens per optimizer step and about 368.6 million token positions in 45,000
steps:

```bash
caffeinate -i .venv/bin/python -B -m final_project.training \
  --device mps \
  --data-dir data/processed/fineweb_edu_experiment_1g \
  --checkpoint-path checkpoints/learngpt-mps-18m-stable-1g-v2.pt \
  --encoding-name gpt2 \
  --seed 1337 \
  --context-size 256 \
  --embedding-size 256 \
  --num-heads 4 \
  --num-transformer-blocks 6 \
  --dropout 0.0 \
  --use-scaled-dot-product-attention \
  --output-chunk-size 32768 \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --training-steps 45000 \
  --eval-interval 250 \
  --eval-batches 20 \
  --base-learning-rate 3e-4 \
  --min-learning-rate 3e-5 \
  --warmup-steps 1000 \
  --decay-steps 45000 \
  --weight-decay 0.05 \
  --gradient-clip 1.0 \
  --max-grad-norm-before-clip 100 \
  --gradient-retry-attempts 3 \
  --context-sensitivity-contexts 32
```

Do not add `--mixed-precision` or `--compile-model` to this MPS recipe. They are
optional features for other backends, not part of the verified path.

At startup, training runs the MPS repeatability and CPU-parity self-check. It
will stop before step 1 if the gradients do not agree. During training,
`grad_norm` is the raw norm measured before clipping; `grad_retries=0` is the
normal result. A persistent integrity failure aborts the run before the
optimizer can consume the bad gradient.

A complete 45,000-step MPS run passed the startup parity check and used zero
retries at its saved evaluations. The best checkpoint occurred at step 42,750
with validation loss `4.2894`; the latest checkpoint reached step 45,000 with
validation loss `4.4524`, raw gradient norm `2.3872`, and
`context_loss_gain=+6.1914`. The machine-readable result and a seeded sample
are recorded in `docs/verified_runs/mps-18m-1g-45000.json`.

`context_js` remains an observational measure of how much the output
distribution varies across contexts. A value near zero early in training is
not, by itself, a failure: a new model normally learns broad token frequencies
before it learns useful context. `context_loss_gain` is target-aware and equals
the loss with shuffled contexts minus the loss with the correct contexts. A
positive trend means the real contexts help predict their next tokens; a value
near zero during the early phase is expected. Neither metric is used as a
premature hard gate.

With GPT-style initialization, the first loss should be close to `ln(50257)`,
approximately `10.82`, rather than tens or hundreds. Loss should then trend
down over many evaluations; individual validation points can still fluctuate.

No artificial delay is needed between steps. `caffeinate` prevents sleep and
macOS already manages thermal throttling. Keep the Mac on a hard surface with
clear airflow and stop only if macOS reports sustained thermal or memory
pressure.

Training writes two atomic checkpoints:

```text
checkpoints/learngpt-mps-18m-stable-1g-v2.pt         # best validation loss
checkpoints/learngpt-mps-18m-stable-1g-v2-latest.pt  # latest evaluated step
```

If the terminal or Mac is interrupted, resume only this new, verified run. The
step count remains the total target, not 45,000 additional steps:

```bash
caffeinate -i .venv/bin/python -B -m final_project.training \
  --device mps \
  --data-dir data/processed/fineweb_edu_experiment_1g \
  --checkpoint-path checkpoints/learngpt-mps-18m-stable-1g-v2.pt \
  --resume-checkpoint-path checkpoints/learngpt-mps-18m-stable-1g-v2-latest.pt \
  --training-steps 45000
```

Generate reproducibly:

```bash
.venv/bin/python -B -m final_project.generate \
  --device mps \
  --checkpoint-path checkpoints/learngpt-mps-18m-stable-1g-v2.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 120 \
  --temperature 0.9 \
  --top-k 50 \
  --seed 1337
```

This run trains a small base language model. Its job is to continue prompts in
plausible English; it is not yet an instruction-following assistant and will
not reliably answer questions. Assistant-style behavior requires a later
instruction-tuning stage on prompt-response examples.

If MPS is not available in the current PyTorch runtime, fix the PyTorch/macOS
environment before training with `--device mps`.

</details>

<details>
<summary>NVIDIA CUDA training</summary>

The Windows/CUDA path trains the same controlled model as MPS: context 256,
embedding width 256, 4 heads, 6 blocks, the seeded 1 GiB subset, 8,192
effective tokens per optimizer step, and a total target of 45,000 steps. CUDA
uses FP16 autocast with a checkpointed GradScaler and zero MPS-style gradient
retries. A transient FP16 overflow lowers the scale and repeats the exact same
batch and step, up to eight times, before failing closed.

The canonical CUDA command also enables `--fused-attention`, which projects Q,
K, and V for every head together and runs one batched SDPA call per block.
`--log-interval` prints inexpensive progress between the less frequent
validation and checkpoint events controlled by `--eval-interval`.

Follow the exact two-phase PowerShell procedure in
[How to Train Runbook — Windows NVIDIA CUDA](docs/FINAL_TRAINING_RUNBOOK.md#6-windows-nvidia-cuda-smoke-gate-and-complete-run).
It runs a 20-step gate with the real architecture, then resumes that same
checkpoint to step 45,000. The section also includes 4, 6, and 8 GiB VRAM
profiles and the matching generation command.

For the larger measured NVIDIA profile and its throughput/VRAM evidence, read
[CUDA training optimizations](CUDA_TRAINING_OPTIMIZATIONS.md).

This backend path is code-reviewed and covered by CPU-side checkpoint tests,
including GradScaler state and overflow backoff/retry. The final hardware gate
must still run on the target NVIDIA machine before a long job is considered
verified.

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
  --resume-checkpoint-path checkpoints/learngpt-cpu-latest.pt \
  --training-steps 200
```

Generate:

```bash
python -m final_project.generate \
  --device cpu \
  --checkpoint-path checkpoints/learngpt-cpu.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 120 \
  --temperature 0.9 \
  --top-k 50 \
  --seed 1337
```

</details>

The training CLI prints the Python and PyTorch runtime, selected device,
dataset size, model and training configuration, validation loss, learning
rate, raw pre-clipping gradient norm, retry count, target-aware context
diagnostics, CUDA AMP retry/overflow counts, tokens per second, and estimated
remaining time. `--log-interval N` prints lightweight step metrics every `N`
updates without triggering validation or a checkpoint write.

## Publishing Checkpoints

Checkpoints can become large, so they are ignored by Git. If you want to share a
trained model publicly, prefer GitHub Release assets or an external model host
instead of committing `.pt`, `.pth`, or `.ckpt` files to the repository.

## Relationship To nanoGPT

LearnGPT follows the local `../nanoGPT` implementation as its architectural
and training reference:

- decoder-only Transformer architecture
- learned token and position embeddings
- causal self-attention
- multi-head attention
- pre-LayerNorm Transformer blocks
- residual connections
- MLP/feed-forward blocks with GELU
- tied token-embedding and output weights with GPT-style initialization
- AdamW training
- gradient accumulation, clipping, warmup, and cosine learning-rate decay
- train/validation evaluation, checkpointing, and autoregressive generation

The main difference is educational structure. nanoGPT batches Q/K/V into one
projection and includes production-oriented features such as real DDP, MFU
reporting, and pretrained GPT-2 import. LearnGPT uses separate Q/K/V projections
and verbose names so every tensor shape is inspectable. It keeps mixed precision
and `torch.compile` optional, explains DDP without launching it, and adds
reproducible FineWeb-Edu subsets, MPS gradient-integrity checks, and target-aware
context diagnostics for compute-bounded local experiments.
