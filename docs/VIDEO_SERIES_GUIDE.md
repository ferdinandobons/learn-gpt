# LearnGPT Video Series Guide

This guide turns the 42 executable checkpoints into a coherent YouTube series.
It is a teaching outline, not a second training manual. Keep the exact commands
in `docs/FINAL_TRAINING_RUNBOOK.md` so the video, repository, and website point
to one operational source.

## A repeatable episode format

Use the same six-part structure in every episode:

1. **Question** — state the problem that the current model cannot solve.
2. **Previous state** — run the last checkpoint and show its limitation.
3. **One visible change** — explain the new tensor operation and its shape.
4. **Live checkpoint** — run the numbered lesson and inspect its output.
5. **Success test** — show the assertion, loss, shape, or generation difference
   that proves the change works.
6. **Thirty-second recap** — connect the new component to the final training
   pipeline and preview the next problem.

Display the checkpoint number, source path, seed, and device whenever code is
run. Avoid promising that a short lesson training loop produces useful prose;
the lesson validates a mechanism, while the final experiment validates the
assembled system.

## Episode 1 — Text becomes data

**Checkpoints:** 01–05
**Question:** How can a neural network receive text?

- Show the tracked `data/study_sample.txt` so a clean clone is immediately
  runnable.
- Build a character vocabulary, encode text to IDs, decode it, and split train
  from validation.
- Keep the distinction clear: the character tokenizer is a teaching tool;
  GPT-2 BPE appears later for the real project.
- Live success check: decoded text exactly equals the original sample.
- Common misconception: a token ID is a magnitude. It is an index, not a
  numerical measurement of meaning.

## Episode 2 — Context windows and batches

**Checkpoints:** 06–11
**Question:** What is one next-token training example, and how do many examples
become a PyTorch batch?

- Slide the input and target windows by one token.
- Show random start positions, Python lists, tensors, batch size, and context
  size.
- Draw `[B, T]` and name both dimensions before displaying any model.
- Live success check: every target row is the corresponding input row shifted
  one position into the future.
- Common misconception: increasing batch size gives the model more context.
  Context length is `T`; batch size is the number of independent windows.

## Episode 3 — The smallest language model

**Checkpoints:** 12–16
**Question:** Can one lookup table learn next-token probabilities?

- Convert token IDs directly into vocabulary logits.
- Explain cross-entropy, backward, AdamW, and autoregressive sampling with the
  bigram model.
- Compare prompts that end in the same character to prove the limitation.
- Live success check: loss falls, parameters change, and identical final tokens
  yield identical next-token scores regardless of earlier context.
- Common misconception: generated text alone proves context use. The paired
  prompt comparison is the controlled test.

## Episode 4 — Embeddings and position

**Checkpoints:** 17–18
**Question:** How can the model represent token identity and order?

- Show token embeddings and position embeddings separately.
- Add the two `[B, T, C]` tensors and inspect one position by hand.
- Explain why the sum keeps the same interface for later blocks.
- Live success check: the manual sum equals the tensor passed to the model.
- Common misconception: an embedding is a probability distribution. It is a
  learned feature vector.

## Episode 5 — Causal multi-head attention

**Checkpoints:** 19–21
**Question:** How can each token read useful earlier tokens without seeing the
future?

- Build queries, keys, values, scaled scores, the causal mask, and softmax.
- Checkpoint 20 concatenates independent head outputs; checkpoint 21 adds the
  output projection that mixes those head features.
- Keep the shapes visible: each head emits `H`, concatenation restores `C`, and
  projection maps `C → C`.
- Live success check: masked future weights are zero and every allowed row sums
  to one.
- Common misconception: multiple heads extend the context window. They provide
  parallel relationship channels inside the same window.

## Episode 6 — Assemble a Transformer block

**Checkpoints:** 22–27
**Question:** How do attention and per-token computation become a stable,
repeatable block?

- Add the attention residual, pre-LayerNorm, 4C GELU feed-forward path, second
  residual, and the `TransformerBlock` wrapper.
- Repeat blocks without changing `[B, T, C]`, then add final LayerNorm.
- Checkpoint 24 exposes the feed-forward path and second residual; checkpoint
  25 packages the same operations into the reusable block abstraction.
- Live success check: the manual sequence matches `TransformerBlock.forward`.
- Common misconception: the feed-forward network mixes positions. Attention
  mixes positions; the MLP transforms features independently at each position.

## Episode 7 — Train, evaluate, and save

**Checkpoints:** 28–35
**Question:** How does a promising architecture become a controlled
experiment?

- Train the Transformer, estimate train and validation loss, save a checkpoint,
  reload it for generation, and add temperature/top-k controls.
- Explain best versus latest before the long run appears.
- Introduce warm-up, cosine decay, clipping, dropout, and weight tying.
- Be precise about historical checkpoints: early lessons save only the state
  introduced at that stage; the final project later adds atomic writes, RNG,
  data identity, and complete resume state.
- Live success check: a newly constructed model loads the checkpoint and
  reproduces generation from saved weights.
- Common misconception: the final step is always the best model. Validation
  loss selects the best checkpoint.

## Episode 8 — Move from teaching code to the real runtime

**Checkpoints:** 36–41
**Question:** What infrastructure is required when the model leaves tiny
in-memory examples?

- Checkpoint 36 is an explicit integration boundary: character IDs become
  GPT-2 BPE, Python lists become `uint16` memmaps, execution becomes
  device-aware, and AdamW parameters are split into decay/no-decay groups.
- Add gradient accumulation, configuration dataclasses, resume, last-token-only
  inference, optional scaled dot-product attention, autocast, and compile.
- Explain DDP conceptually without implying that LearnGPT launches it.
- Live success check: effective tokens per update equal
  `batch × context × accumulation`, and resume continues at `N + 1`.
- Common misconception: a performance flag changes the model architecture.
  These flags change execution while preserving the mathematical contract.

## Episode 9 — The final guarded project

**Checkpoint:** 42
**Question:** How do all components become one trustworthy local training
system?

- Run the self-contained Lesson 42 demo first. State explicitly that it proves
  integration with a compact configuration, not final language quality.
- Trace the final path: GPT-2 BPE, memmaps, model, loss, backward, optimizer,
  evaluation, best/latest checkpoint, seeded generation.
- Explain the MPS failure and fix as a cause-and-effect sequence: monolithic
  vocabulary backward and reallocating leaf gradients produced bad directions;
  chunked projection, persistent buffers, warm-up/parity checks, and raw-norm
  retry protection make the run fail closed.
- Show that CUDA FP16 stores the GradScaler and that every new checkpoint stores
  a dataset fingerprint.
- Trigger a synthetic CUDA-style overflow test: the scale backs off and the
  exact same batch/step is retried, while persistent overflow fails closed.
- Live success check: repository tests pass and the Lesson 42 sample can be
  generated from its saved checkpoint.
- Common misconception: gradient clipping repairs a corrupted direction. It
  limits magnitude only; integrity checks must happen first.

## Episode 10 — Prepare and launch the real experiment

**Source:** `docs/FINAL_TRAINING_RUNBOOK.md`
**Question:** What exact sequence turns a clean repository into the controlled
17.7M-parameter run?

- Verify the backend and repository.
- Prepare the canonical 10 GiB corpus and seeded 1 GiB subset.
- Validate the exact training directory.
- Run the backend smoke gate, then the 45,000-step profile.
- Read loss, raw gradients, MPS retries, CUDA AMP overflows, throughput, ETA,
  and context loss gain.
- Demonstrate interruption and resume from latest without changing the total
  target.
- Generate seeded samples from best and contrast them with the old collapsed
  output.
- End with the honest boundary: this is a small base model, not a chat model.

## Recording checklist

- Pull both repositories and show their commit IDs.
- Run the validator before recording a lesson.
- Use the checkpoint deep link on the website for the current lesson.
- Keep terminal commands in English and run them from the repository root.
- Show complete error messages when teaching a failure.
- Record device, PyTorch version, dataset profile, checkpoint name, and seed.
- Use a disposable checkpoint for demonstrations; never overwrite the verified
  run.
- Link `course_en.md`, the final runbook, and the verified run manifest in the
  video description.
- Re-run website generation after source changes so the displayed code revision
  matches the repository.

## Final narrative in one sentence

LearnGPT begins with a text file, turns it into next-token examples, builds the
decoder one mechanism at a time, protects the training state, and ends with a
reproducible experiment whose result can be explained rather than merely shown.
