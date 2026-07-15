# Data

This repository does not include datasets, processed binary files, checkpoints,
or model outputs.

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
