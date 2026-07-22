# CUDA Training Optimizations

This document records the optimizations developed and measured for training
LearnGPT on a Windows PC with an NVIDIA GeForce RTX 4060. It explains what each
setting means, why it was changed, which changes preserve model quality, and
which configuration was selected for the long training run.

## Target machine and experiment

The measurements in this document were collected with:

| Property | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 |
| VRAM | 8,188 MiB |
| Driver | 591.86 |
| PyTorch | 2.12.1+cu130 |
| CUDA runtime | 13.0 |
| Operating system | Windows 11, WDDM |
| Dataset | 50 GiB FineWeb-Edu `sample-100BT` preparation |
| Dataset tokens | 26,843,545,600 |
| Model | approximately 124 million parameters |
| Context | 256 tokens |
| Embedding width | 768 |
| Attention heads | 12 |
| Transformer blocks | 12 |
| Training target | 300,000 optimizer steps |

The final model remains a decoder-only GPT-style base language model. None of
the optimizations reduce its width, depth, context, dataset, effective batch,
or training-step target.

## Effective batch and gradient accumulation

`batch-size` is the number of sequences processed by the GPU at the same time.
`context-size` is the number of tokens in each sequence.
`gradient-accumulation-steps` is the number of micro-batches whose gradients
are accumulated before AdamW updates the model.

The number of token positions contributing to one optimizer update is:

```text
tokens per optimizer step =
    batch size * context size * gradient accumulation steps
```

All benchmarked configurations kept this value constant:

| Micro-batch | Accumulation | Context | Effective tokens per step |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 256 | 8,192 |
| 2 | 16 | 256 | 8,192 |
| 4 | 8 | 256 | 8,192 |
| 8 | 4 | 256 | 8,192 |
| 16 | 2 | 256 | 8,192 |

This makes the performance comparison meaningful. Every configuration gives
the optimizer the equivalent of 32 sequences before an update and processes
the same 2,457,600,000 token positions in 300,000 steps. Only the division of
that work into GPU micro-batches changes.

A larger micro-batch exposes more parallel matrix work to the GPU and reduces
the number of Python and kernel-launch cycles. It also uses more VRAM. Gradient
accumulation allows the effective batch to stay large when the full batch does
not fit in memory.

## Baseline CUDA optimizations

### Mixed precision

Training uses:

```text
--mixed-precision --precision-dtype float16
```

The master model parameters and AdamW state remain FP32. CUDA performs eligible
forward and backward operations in FP16, while `GradScaler` protects gradients
from numerical underflow and overflow. Checkpoints therefore retain FP32 model
weights and are not 8-bit or 4-bit quantized models.

This reduces activation memory and enables Tensor Core execution without
changing the model architecture or intended training objective.

### Scaled dot-product attention

Training uses:

```text
--use-scaled-dot-product-attention
```

This delegates causal attention to PyTorch's optimized CUDA SDPA implementation
instead of explicitly materializing and masking the complete attention matrix
in Python-level model code.

### Fused AdamW

The optimizer setup automatically selects PyTorch's fused CUDA AdamW when the
runtime supports it. Parameter grouping, weight decay, beta values, and the
optimizer update remain the same; the CUDA implementation performs the update
with fewer separate kernel launches.

### Monolithic vocabulary projection

CUDA training uses:

```text
--output-chunk-size 0
```

The model projects hidden states to all 50,257 vocabulary logits in one matrix
operation. Chunking remains useful for constrained MPS or CUDA memory, but on
this machine the monolithic projection is faster and fits in VRAM.

## Micro-batch tuning

Before attention fusion, micro-batch size was increased while preserving 8,192
effective tokens per optimizer step:

| Batch | Accumulation | Observed VRAM | Measured throughput |
| ---: | ---: | ---: | ---: |
| 1 | 32 | about 4.4 GiB | 664 tokens/s |
| 2 | 16 | about 4.2 GiB | 1,352 tokens/s |
| 4 | 8 | about 4.9 GiB | 2,348 tokens/s |
| 8 | 4 | about 6.9 GiB | 3,740 tokens/s |

VRAM readings are instantaneous `nvidia-smi` observations rather than exact
allocator peaks. Throughput is the training runtime's measured token rate.

Batch 8 was the fastest stable pre-fusion configuration. It left enough WDDM
headroom for Windows while substantially reducing gradient-accumulation
overhead.

## Fused multi-head QKV attention

The original educational attention path represents every attention head as a
separate Python module. For a 12-block, 12-head model, every micro-batch runs:

```text
144 separate head executions
432 separate Q, K, and V projections
144 separate scaled dot-product attention calls
```

The optimized path is enabled with:

```text
--fused-attention
```

For each Transformer block it now performs:

1. one combined linear projection producing Q, K, and V for every head;
2. one reshape from embedding channels into the head dimension;
3. one batched CUDA SDPA call across all heads;
4. one reshape and output projection.

Across the 12 blocks, this reduces the work to 12 combined QKV projections and
12 batched attention calls per micro-batch.

The fused and separate implementations have the same number of trainable
parameters and implement the same attention computation, subject only to normal
floating-point ordering differences. A regression test copies the separate
head weights into the fused representation and verifies output parity. Legacy
checkpoints that do not contain the `fused_attention` configuration field keep
using the original separate-head implementation.

## Vectorized and pinned batch preparation

The original batch loader used a Python loop to extract every context window,
created individual tensors, stacked them, and then copied them to the GPU.

The optimized loader:

- constructs all token positions as one NumPy index matrix;
- reads the complete micro-batch in one vectorized operation;
- creates input and shifted-target tensors from the resulting array;
- pins CPU memory for CUDA transfers;
- uses non-blocking host-to-device copies.

The sampled token windows and training targets are unchanged. A regression test
verifies that every target is the input shifted by exactly one token.

## Lightweight logging versus evaluation

`eval-interval` previously controlled both progress visibility and expensive
work. Every evaluation performs multiple training and validation forwards,
context diagnostics, and writes checkpoint files of approximately 1.5 GB.
Running this every 100 steps would add substantial runtime and SSD writes.

The new `log-interval` setting reports inexpensive live metrics without running
validation or saving a checkpoint:

```text
--log-interval 100
--eval-interval 500
```

Every 100 steps the terminal reports current loss, learning rate, gradient
norm, AMP retry counts, throughput, and ETA. Every 500 steps the runtime also
computes train/validation estimates, context diagnostics, and atomically saves
best/latest checkpoints.

This improves visibility without changing gradient updates or training quality.

## Final fused-attention benchmark

The fused path was tested with fresh checkpoints and then resumed from step 20
to step 100 to verify checkpoint compatibility, RNG restoration, GradScaler
state, optimizer state, and stable throughput.

| Fused configuration | Throughput at step 100 | VRAM observation | Result |
| --- | ---: | ---: | --- |
| Batch 8, accumulation 4 | 20,312 tokens/s | about 5.9 GiB | Selected |
| Batch 16, accumulation 2 | 11,255 tokens/s | 7,773 MiB | Rejected |

Batch 16 reached 100% instantaneous GPU utilization, but it was slower and left
only about 415 MiB of VRAM headroom. The larger operation became less efficient
on this GPU and created an unacceptable stability margin for a multi-day WDDM
training run.

Batch 8 is therefore the measured optimum: it is faster, leaves safer memory
headroom, and preserves the same 8,192-token effective batch.

The selected step-100 smoke run reported:

```text
validation loss: 9.2871
gradient norm: 2.7334
AMP overflows: 0
context loss gain: +0.8963
throughput: 20,312 tokens/s
```

## Quality and functional equivalence

The optimized run preserves the original 300,000-step training specification:

- the same approximately 124-million-parameter decoder-only architecture;
- the same 12 blocks, 12 heads, width 768, and context length 256;
- the same FineWeb-Edu token files and GPT-2 tokenizer;
- the same 8,192 effective tokens per optimizer update;
- the same AdamW settings, learning-rate schedule, seed, and step count;
- FP32 master weights and optimizer state, with FP16 used only for eligible
  mixed-precision CUDA operations.

The separate-head and fused-attention step-100 smoke runs also produced closely
matching optimization signals:

| Attention path | Train loss | Validation loss | Reported loss | Gradient norm | Context loss gain | AMP overflows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original separate heads | 9.2490 | 9.2847 | 9.2692 | 2.5580 | +0.8375 | 0 |
| Fused QKV | 9.2479 | 9.2871 | 9.2717 | 2.7334 | +0.8963 | 0 |

The validation-loss difference was 0.0024, approximately 0.026%, which is
consistent with ordinary initialization, sampling, and floating-point ordering
variation. The two fresh runs are not expected to be bit-for-bit identical,
but they are architecturally, functionally, and statistically equivalent for
the intended training objective.

At that measured rate, 300,000 steps represent about 33.6 hours of pure
training computation. Full validation, context diagnostics, checkpoint writes,
Windows scheduling, and thermal behavior add overhead, so approximately 1.5 to
2 days is a more realistic operational estimate for a dedicated machine.

## Final clean training command

Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -B -m final_project.training `
  --device cuda `
  --data-dir data\processed\fineweb_edu_50g_100bt `
  --checkpoint-path checkpoints\learngpt-cuda-124m-fused-b8-clean-300k.pt `
  --encoding-name gpt2 `
  --seed 1337 `
  --context-size 256 `
  --embedding-size 768 `
  --num-heads 12 `
  --num-transformer-blocks 12 `
  --dropout 0.0 `
  --use-scaled-dot-product-attention `
  --fused-attention `
  --output-chunk-size 0 `
  --batch-size 8 `
  --gradient-accumulation-steps 4 `
  --training-steps 300000 `
  --log-interval 100 `
  --eval-interval 500 `
  --eval-batches 10 `
  --base-learning-rate 2e-4 `
  --min-learning-rate 2e-5 `
  --warmup-steps 3000 `
  --decay-steps 300000 `
  --weight-decay 0.1 `
  --gradient-clip 1.0 `
  --max-grad-norm-before-clip 100 `
  --gradient-retry-attempts 0 `
  --context-sensitivity-contexts 16 `
  --mixed-precision `
  --precision-dtype float16
```

This is a fresh run because it does not specify `--resume-checkpoint-path`.

For the run to start unambiguously from step zero, neither of these files should
exist before launching it:

```text
checkpoints\learngpt-cuda-124m-fused-b8-clean-300k.pt
checkpoints\learngpt-cuda-124m-fused-b8-clean-300k-latest.pt
```

Before the production launch, all 14 earlier experiment checkpoints were
removed from `checkpoints`, reclaiming 19.75 GiB. This included the original,
fast, batch-size comparison, fused smoke-test, and previous clean-test
checkpoints. The directory was verified empty, both production paths above were
verified absent, and no Python training process was active. Dataset files,
source code, tests, and documentation were not removed.

The cleanup is a one-time recorded state, not an instruction to delete future
production checkpoints. Once training starts, keep the best checkpoint for
generation and the automatically maintained `-latest` checkpoint for resume.

## Resume command

Resume from the latest checkpoint while keeping every saved model and training
setting:

```powershell
.\.venv\Scripts\python.exe -B -m final_project.training `
  --device cuda `
  --data-dir data\processed\fineweb_edu_50g_100bt `
  --checkpoint-path checkpoints\learngpt-cuda-124m-fused-b8-clean-300k.pt `
  --resume-checkpoint-path checkpoints\learngpt-cuda-124m-fused-b8-clean-300k-latest.pt `
  --training-steps 300000
```

The total target remains 300,000 steps; resume does not add another 300,000
steps. Generate from the best checkpoint and resume from the `-latest`
checkpoint.

## Validation performed

The optimized implementation passed:

- the repository structure and snapshot validator;
- 37 regression tests;
- fused-versus-separate attention output parity;
- equal parameter-count verification;
- causal-attention tests;
- deterministic vectorized batch-shift tests;
- legacy checkpoint configuration tests;
- CUDA fresh-training and resume smoke runs;
- FP16 gradient integrity checks with zero AMP overflows.

The optimizations intentionally leave `torch.compile` disabled. The current
Windows environment reports that Triton is unavailable, so compilation was not
introduced into a long production run without a separately verified backend.
