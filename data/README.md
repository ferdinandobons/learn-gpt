# Data

The repository includes `data/study_sample.txt`, a small original English text
written for LearnGPT. Lessons 01–35 use it so the complete teaching path runs
from a clean clone without downloading a training corpus. It is not the final
training corpus. The first GPT-2 BPE lesson may still download and cache the
small tokenizer vocabulary files through `tiktoken` if they are not already on
the machine.

Large datasets, processed binary files, checkpoints, and model outputs are not
committed.

To prepare FineWeb-Edu locally, install the final project dependencies and run:

```bash
python -B final_project/prepare_data.py \
  --target-gb 10 \
  --output-dir data/processed/fineweb_edu
```

The command creates:

```text
data/processed/fineweb_edu/
  train.bin
  val.bin
  meta.json
```

Those files are intentionally ignored by Git.

## Controlled local experiment

Keep `processed/fineweb_edu/` unchanged as the canonical 10 GiB source. To
create a separate, reproducible 1 GiB subset for a compute-bounded training
run, use:

```bash
python -B -m final_project.prepare_subset \
  --source-data-dir data/processed/fineweb_edu \
  --output-dir data/processed/fineweb_edu_experiment_1g \
  --target-gb 1 \
  --validation-ratio 0.01 \
  --seed 1337 \
  --chunk-tokens 65536
```

The generated `train.bin`, `val.bin`, and `meta.json` are also ignored by Git.
`meta.json` records the source dataset, seed, chunk size, and exact token
counts so that the experiment can be recreated.

Validate the exact directory before training:

```bash
python -B tools/validate_learngpt.py \
  --training-data-dir data/processed/fineweb_edu_experiment_1g
```

Every new training checkpoint stores a path-independent SHA-256 fingerprint of
`meta.json`, `train.bin`, and `val.bin`. Resume fails if the selected token
files differ. Full preparation, PowerShell equivalents, and backend commands
are in `docs/FINAL_TRAINING_RUNBOOK.md`.
