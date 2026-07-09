# Data

This repository does not include datasets, processed binary files, checkpoints,
or model outputs.

To prepare FineWeb-Edu locally, install the final project dependencies and run:

```bash
python -B final_project/prepare_data.py \
  --target-gb 5 \
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
