# How to Train LearnGPT

This is the canonical operational guide for turning the 42 LearnGPT lessons
into one complete, reproducible training experiment. Use `course_en.md` to
learn the implementation; use this runbook when you are ready to prepare real
data, train, resume, and evaluate a model.

## What the final experiment builds

| Property | Controlled value |
| --- | ---: |
| Parameters | 17,716,049 |
| Vocabulary | 50,257 GPT-2 BPE tokens |
| Context | 256 tokens |
| Embedding width | 256 |
| Attention heads | 4 |
| Transformer blocks | 6 |
| Effective tokens per optimizer step | 8,192 |
| Optimizer steps | 45,000 |
| Total token positions | 368,640,000 |
| Training data | seeded 1 GiB subset of a canonical 10 GiB FineWeb-Edu preparation |

This is a small base language model. A successful run produces plausible
English continuations that react to their prompt. It does not produce a
reliable chat assistant; instruction following requires a later fine-tuning
stage on prompt-response data.

## 1. Clone, install, and verify the environment

LearnGPT requires Python 3.12 or newer because the pinned NumPy build has the
same minimum. Python 3.13 is recommended; the verified MPS run used Python
3.13.12 and PyTorch 2.12.1. Run commands from the repository root.

Start from a clean clone:

```bash
git clone https://github.com/ferdinandobons/learn-gpt.git
cd learn-gpt
```

### macOS or Linux shell

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r final_project/requirements.txt
```

Verify Apple Silicon MPS:

```bash
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_built(), torch.backends.mps.is_available()); print(torch.ones(1, device='mps'))"
```

### Windows PowerShell with an NVIDIA GPU

Create the environment, install the pinned CUDA-enabled PyTorch wheel, and then
install the pinned common dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r final_project\requirements-common.txt
```

If the official PyTorch selector recommends a different CUDA wheel for your
driver, keep `torch==2.12.1` and replace only the wheel index. Do not install a
second CPU-only `torch` package afterward.

Verify CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

The last two lines must report `True` and the NVIDIA GPU name. If they do not,
fix the NVIDIA driver and PyTorch wheel before preparing a long run.

## 2. Prove the repository works before downloading data

The numbered lessons use the tracked `data/study_sample.txt`, and Lesson 42
creates its own small token arrays. These checks therefore work on a clean
clone:

```bash
.venv/bin/python -B tools/validate_learngpt.py
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B study/lessons/01_read_text.py
.venv/bin/python -B study/lessons/42_final_project.py
```

On Windows, replace `.venv/bin/python` with
`.\.venv\Scripts\python.exe`. The validator also checks that the final lesson
snapshot is byte-for-byte aligned with `final_project/`.

## 3. Prepare the canonical 10 GiB corpus

Do this once. It streams FineWeb-Edu, tokenizes it with GPT-2 BPE, and writes
local `uint16` token files:

```bash
.venv/bin/python -B final_project/prepare_data.py \
  --target-gb 10 \
  --output-dir data/processed/fineweb_edu
```

PowerShell equivalent:

```powershell
.\.venv\Scripts\python.exe -B final_project\prepare_data.py `
  --target-gb 10 `
  --output-dir data\processed\fineweb_edu
```

The result contains `train.bin`, `val.bin`, and `meta.json`. Keep this directory
unchanged as the canonical source. The Hugging Face cache uses the operating
system's temporary directory, so the command is portable across macOS, Linux,
and Windows.

## 4. Create the reproducible 1 GiB experiment

The controlled run samples non-overlapping chunks with seed 1337. It does not
modify the 10 GiB source:

```bash
.venv/bin/python -B -m final_project.prepare_subset \
  --source-data-dir data/processed/fineweb_edu \
  --output-dir data/processed/fineweb_edu_experiment_1g \
  --target-gb 1 \
  --validation-ratio 0.01 \
  --seed 1337 \
  --chunk-tokens 65536
```

PowerShell equivalent:

```powershell
.\.venv\Scripts\python.exe -B -m final_project.prepare_subset `
  --source-data-dir data\processed\fineweb_edu `
  --output-dir data\processed\fineweb_edu_experiment_1g `
  --target-gb 1 `
  --validation-ratio 0.01 `
  --seed 1337 `
  --chunk-tokens 65536
```

Validate exactly the dataset that the training command will use:

```bash
.venv/bin/python -B tools/validate_learngpt.py \
  --training-data-dir data/processed/fineweb_edu_experiment_1g
```

Training hashes the complete token files at startup and stores the resulting
dataset fingerprint in every new checkpoint. This takes a short extra pass over
the files and prevents an accidental resume on a different corpus.

## 5. Apple Silicon MPS: complete fresh run

Use a new checkpoint name. Fresh training now fails rather than overwriting an
existing best or latest checkpoint. `--overwrite-checkpoints` exists only for
an intentional discard.

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

Do not add mixed precision or `torch.compile` to this controlled MPS profile.
The MPS safety path uses four explicit protections:

1. vocabulary projection in chunks of at most 32,768 tokens;
2. persistent gradient buffers that are cleared in place;
3. a discarded warm-up backward followed by repeated-MPS and CPU-parity checks;
4. a raw gradient-norm limit with bounded retries before any optimizer update.

No artificial pause is required between steps. macOS manages thermal
throttling. Keep the computer on a hard surface with clear airflow and allow
`caffeinate` to prevent sleep.

## 6. Windows NVIDIA CUDA: smoke gate and complete run

This profile uses the same data, architecture, schedule, effective batch, and
45,000-step target as the MPS experiment. Only backend-specific execution
changes: CUDA uses FP16 autocast plus a checkpointed GradScaler, a monolithic
output projection, and no MPS-style gradient retries. A transient FP16
overflow lowers the scale and repeats the exact same batch and training step;
after eight unsuccessful backoffs the run stops before an optimizer update.

First run 20 steps into the final checkpoint family. This is not a different
model; it is the first short phase of the same run:

```powershell
.\.venv\Scripts\python.exe -B -m final_project.training `
  --device cuda `
  --data-dir data\processed\fineweb_edu_experiment_1g `
  --checkpoint-path checkpoints\learngpt-cuda-18m-stable-1g.pt `
  --encoding-name gpt2 `
  --seed 1337 `
  --context-size 256 `
  --embedding-size 256 `
  --num-heads 4 `
  --num-transformer-blocks 6 `
  --dropout 0.0 `
  --use-scaled-dot-product-attention `
  --output-chunk-size 0 `
  --batch-size 4 `
  --gradient-accumulation-steps 8 `
  --training-steps 20 `
  --eval-interval 250 `
  --eval-batches 20 `
  --base-learning-rate 3e-4 `
  --min-learning-rate 3e-5 `
  --warmup-steps 1000 `
  --decay-steps 45000 `
  --weight-decay 0.05 `
  --gradient-clip 1.0 `
  --max-grad-norm-before-clip 100 `
  --gradient-retry-attempts 0 `
  --context-sensitivity-contexts 32 `
  --mixed-precision `
  --precision-dtype float16
```

Confirm that loss is finite, the first loss is near 10.82, `amp_overflows` is
ideally 0 or stabilizes after a small startup backoff, and both checkpoint
files exist. Then continue the same run to the total target:

```powershell
.\.venv\Scripts\python.exe -B -m final_project.training `
  --device cuda `
  --data-dir data\processed\fineweb_edu_experiment_1g `
  --checkpoint-path checkpoints\learngpt-cuda-18m-stable-1g.pt `
  --resume-checkpoint-path checkpoints\learngpt-cuda-18m-stable-1g-latest.pt `
  --training-steps 45000
```

The smoke phase is still evaluated and checkpointed at its final step. Keeping
the production evaluation settings here means the resumed 45,000-step run does
not inherit a short-test cadence.

The resume command reads architecture, optimizer, schedule, mixed-precision
mode, GradScaler, RNG state, and dataset fingerprint from the checkpoint. Keep
Windows sleep disabled while plugged in; no delay between optimizer steps is
needed. `amp_retries` is the number of same-step retries in the current logged
iteration; `amp_overflows` is the cumulative count and survives resume.

If CUDA reports out of memory, preserve 8,192 effective tokens per step:

| Approximate VRAM | `--batch-size` | `--gradient-accumulation-steps` | Additional change |
| --- | ---: | ---: | --- |
| 8 GiB or more | 4 | 8 | controlled default |
| 6 GiB | 2 | 16 | none |
| 4 GiB | 1 | 32 | use `--output-chunk-size 32768` |

Start a new checkpoint family after changing these settings. Actual CUDA
compatibility and throughput must still be confirmed on the target NVIDIA
machine because no software-only test can certify an unavailable GPU.

## 7. How to read progress

| Signal | Healthy interpretation |
| --- | --- |
| `loss` | noisy loss from the current optimizer step |
| `train` | averaged estimate on training batches |
| `val` | averaged held-out estimate; use this for best checkpoint selection |
| `lr` | warm-up to `3e-4`, then cosine decay toward `3e-5` |
| `grad_norm` | raw norm before clipping; finite and normally well below 100 |
| `grad_retries` | normally 0; MPS retries only after an integrity failure |
| `amp_retries` | normally 0; CUDA repeats the same step after a transient FP16 overflow |
| `amp_overflows` | cumulative CUDA overflows; a small stable count can be healthy, persistent growth is not |
| `tok/s` | measured throughput, useful for comparing configurations |
| `context_loss_gain` | shuffled-context loss minus true-context loss; a positive trend means context helps |

`context_js` is observational and may be near zero early in a healthy run. Do
not stop on one noisy evaluation. Stop on a non-finite loss or non-AMP gradient,
a failed MPS parity check, exhausted gradient or AMP retries, persistently
growing `amp_overflows`, persistent validation deterioration, a data fingerprint
mismatch, or sustained operating-system thermal or memory warnings.

## 8. Best versus latest checkpoint

Training writes atomically:

```text
checkpoints/<run>.pt         # lowest observed validation loss
checkpoints/<run>-latest.pt  # newest evaluated step
```

Generate from the best checkpoint. Resume from the latest checkpoint. The
45,000-step verified MPS run ended at step 45,000, while its best validation
loss occurred at step 42,750; that difference is normal.

## 9. Resume MPS safely

```bash
caffeinate -i .venv/bin/python -B -m final_project.training \
  --device mps \
  --data-dir data/processed/fineweb_edu_experiment_1g \
  --checkpoint-path checkpoints/learngpt-mps-18m-stable-1g-v2.pt \
  --resume-checkpoint-path checkpoints/learngpt-mps-18m-stable-1g-v2-latest.pt \
  --training-steps 45000
```

`--training-steps` is the total target, not an additional number of steps. New
checkpoints reject a different dataset automatically. Old checkpoints created
before dataset fingerprints were introduced still load with an explicit
warning that identity cannot be verified. A safe resume also requires the
existing best and `-latest` files from the same checkpoint family; the CLI
rejects implicit branching or a mismatched destination path.

## 10. Generate a reproducible evaluation sample

```bash
.venv/bin/python -B -m final_project.generate \
  --device mps \
  --checkpoint-path checkpoints/learngpt-mps-18m-stable-1g-v2.pt \
  --prompt "The purpose of education is" \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-k 40 \
  --num-samples 3 \
  --seed 1337
```

Test several prompts and keep the seed and sampling settings in every recorded
demo. A good base-model check asks whether the text is plausible, whether it
responds to the prompt, and whether different prompts lead to different
continuations. It does not grade the model as a factual question-answering
assistant.

## Verified reference result

The completed MPS run produced a best validation loss of `4.2894` at step
42,750. Its latest step was 45,000 with validation loss `4.4524`, raw gradient
norm `2.3872`, zero retries, and context loss gain `+6.1914`. With seed 1337,
the best checkpoint continued the prompt with:

> The purpose of education is to assess and promote educational attainment to the students, to assess and develop the skills needed for all students or students of all ages.

The exact machine-readable record is in
`docs/verified_runs/mps-18m-1g-45000.json`. Local datasets and checkpoints
remain ignored by Git.

## Final checklist

- Repository validator and tests pass.
- The selected backend creates a device tensor.
- Canonical corpus and experimental subset metadata are complete.
- The training data directory passes `--training-data-dir` validation.
- The checkpoint name is new, unless overwrite is explicitly intended.
- The smoke gate produces finite loss, gradients, and loadable checkpoints.
- The full run preserves the controlled architecture and schedule.
- Resume uses latest; generation uses best.
- Evaluation records prompt, seed, temperature, top-k, and checkpoint.
- Results are described as a base language model, not an assistant.
