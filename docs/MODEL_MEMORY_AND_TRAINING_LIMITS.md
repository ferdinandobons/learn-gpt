# Understand Small-Model Memory and Training Limits

This note explains why the controlled LearnGPT experiment uses approximately
18 million parameters and what becomes expensive when the model grows.

> **18M is not the largest model whose weights fit in RAM.**
> It is a practical balance between memory, throughput, numerical safeguards,
> data volume, and training time on an 8 GB Apple Silicon Mac.

## 1. What is a parameter?

A language model receives token IDs and predicts the next token. Its parameters,
or weights, are the trainable numbers changed by the optimizer after each
backward pass.

LearnGPT's controlled model has 17,716,049 parameters. More parameters generally
increase representational capacity, but they do not guarantee a better model.
The run must also provide enough data, optimizer updates, and compute.

## 2. The controlled LearnGPT configuration

| Setting | Value | Meaning |
| --- | ---: | --- |
| Parameters | 17,716,049 | Trainable model values |
| Vocabulary | 50,257 tokens | Possible output tokens |
| Context | 256 tokens | Tokens visible in one sequence |
| Embedding width | 256 | Width of each internal token vector |
| Transformer blocks | 6 | Model depth |
| Physical batch | 4 sequences | Sequences resident in one micro-batch |
| Gradient accumulation | 8 | Micro-batches per optimizer update |
| Effective tokens/update | 8,192 | `4 × 256 × 8` |
| Optimizer updates | 45,000 | Complete controlled run |
| Token positions processed | 368,640,000 | Total training work |

The controlled subset occupies 1 GiB on disk. LearnGPT reads it with memory
mapping, so the operating system loads only the required regions instead of
copying the complete file into RAM.

## 3. Disk, RAM, and accelerator memory

| Resource | Purpose | Examples |
| --- | --- | --- |
| SSD | Persistent storage | Dataset, checkpoints, source code |
| RAM | Temporary working data | Python runtime, batches, caches |
| Accelerator memory | Tensors used for device computation | Weights, gradients, activations |

Apple Silicon uses unified memory: the CPU, GPU, macOS, applications, and the
training process share the same physical pool. An 8 GB Mac therefore cannot
dedicate all 8 GB to the model.

## 4. Why 18M parameters require more than 71 MB during training

An FP32 value occupies four bytes. Model weights alone therefore require about:

```text
17.7 million parameters × 4 bytes ≈ 71 MB
```

Training also keeps a gradient and two AdamW state values for most parameters:

```text
weight              current trainable value
gradient            direction computed by backward
AdamW first moment  moving average of gradients
AdamW second moment moving average of squared gradients
```

A minimal estimate is therefore:

```text
17.7M × 4 values × 4 bytes ≈ 283 MB
```

This excludes activations, temporary tensors, allocator overhead, framework
state, and possible backend copies. It is not a guaranteed upper bound.

Inference is cheaper because generation primarily needs weights and the current
forward-pass state. Training additionally needs gradients, optimizer state, and
intermediate activations for the backward pass.

## 5. Activations

Every layer produces intermediate tensors during the forward pass. Backward
needs many of them later, so they cannot all be discarded immediately. Their
memory cost grows mainly with:

```text
batch × context length × model width × number of blocks
```

One controlled micro-batch contains:

```text
4 sequences × 256 tokens = 1,024 token positions
```

Doubling the physical batch roughly doubles many activations. Increasing the
model width also enlarges attention and feed-forward projections.

## 6. Why a 512-token context is not merely twice as expensive

Classical self-attention compares token positions pairwise. Its compute and
some intermediate tensors scale approximately with the square of context
length:

```text
attention work ∝ context²
```

| Context | Relative pair comparisons |
| ---: | ---: |
| 256 | 1× |
| 512 | 4× |
| 1,024 | 16× |

Optimized kernels can reduce intermediate-memory pressure, but they do not make
the underlying pairwise work linear.

## 7. Vocabulary logits

The output head assigns a logit to every vocabulary token. A complete FP32
logit tensor for one controlled micro-batch would have shape:

```text
[batch, context, vocabulary] = [4, 256, 50,257]
```

It occupies approximately 196 MiB before loss and backward temporaries. The
controlled profile uses `output_chunk_size=32768` so the vocabulary projection
is computed in chunks, reducing peak memory and avoiding one very large MPS
backward operation.

## 8. Physical batch and gradient accumulation

Gradient accumulation approximates a larger effective batch without holding all
sequences in memory simultaneously:

1. Process four sequences.
2. Backpropagate their scaled loss without updating weights.
3. Repeat for eight micro-batches.
4. Apply one AdamW update using the accumulated gradient.

```text
physical batch:             4 sequences
tokens per micro-batch:     4 × 256 = 1,024
accumulated micro-batches:  8
tokens per update:          1,024 × 8 = 8,192
```

Accumulation trades time for memory. It still performs all eight forward and
backward passes.

## 9. Why increasing parameters is not enough

A larger model needs more compute and usually more training data. If model size
increases fivefold while the run remains fixed at 45,000 updates and 368.6
million token positions, the larger model may be undertrained.

Always separate these questions:

1. Does the configuration fit in memory?
2. Is there enough data and compute to train it usefully?

The first answer can be yes while the second is no.

## 10. Other practical limits on an 8 GB Mac

### Memory pressure

When free unified memory becomes scarce, macOS can compress memory or swap to
the SSD. The process may continue but become dramatically slower, or MPS may
terminate the operation.

### Throughput and heat

A larger model performs more matrix multiplication per update. A configuration
can fit in memory yet take several impractical days and experience thermal
throttling on a fanless laptop.

### Backend validation

The MPS runtime includes explicit gradient checks before optimizer updates. A
substantially different architecture or batch profile requires a new smoke test
and validation run; changing dimensions is not sufficient evidence of safety.

## 11. What each control changes

| Change | Memory | Time/update | Main effect |
| --- | --- | --- | --- |
| Double physical batch | Activations roughly double | Increases | More simultaneous examples |
| Double context | Attention cost rises strongly | Rises strongly | More visible history |
| Double width | Rises strongly | Rises strongly | Larger internal matrices |
| Add blocks | Rises | Approximately linear increase | Deeper model |
| Increase vocabulary | Output and embedding grow | Increases | More distinct token IDs |
| Increase accumulation | Nearly unchanged peak | Increases | Larger effective batch |

Exact scaling depends on the backend, precision, and selected kernels.

## 12. Reasonable experimental scales

| Model scale | 8 GB Mac assessment | Expected trade-off |
| ---: | --- | --- |
| 18M | Verified controlled profile | Current balance |
| 30–50M | Plausible with conservative batch/context | Slower; must be revalidated |
| 100–125M | Only with aggressive constraints | Much higher pressure and duration |
| Larger | Impractical for this local run | Needs more resources or another profile |

A 100M model is not only about six times the weight memory of an 18M model.
Optimizer state, activations, temporary tensors, and the required training
budget also grow.

## 13. Mental model

```text
parameters     capacity and weight memory
optimizer      extra state during training
physical batch simultaneous activation memory
context        attention memory and compute
vocabulary     embedding and output cost
training tokens how much evidence the model receives
accelerator    real throughput and supported kernels
MPS safeguards evidence that updates are numerically trustworthy
```

For LearnGPT, 17.7M parameters, context 256, physical batch 4, and accumulation
8 create a model large enough to demonstrate a real training pipeline while
remaining reproducible on an 8 GB Mac. The next sensible experiment is a short,
measured 30–50M run that records peak memory, tokens per second, loss, and
gradient integrity before committing to a complete training budget.
