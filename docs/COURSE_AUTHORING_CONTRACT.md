# LearnGPT course authoring contract

This document preserves the editorial and product choices behind LearnGPT. Use
it before editing `course_en_graphic.md`, `course_it_graphic.md`, lesson code,
snapshots, or public documentation.

## Product Boundaries

- LearnGPT teaches how to build a small GPT-style decoder-only language model
  step by step.
- The model is educational. It is not a frontier model, not an assistant, and
  not a ChatGPT replacement.
- The course covers the implemented path: tokenization, batches, embeddings,
  attention, Transformer blocks, optimization, checkpointing, evaluation, and
  generation.
- Classification fine-tuning, instruction fine-tuning, LoRA, GPT-2 weight
  loading, newer architecture experiments, and production-scale training can be
  mentioned as future paths only when the current lesson does not implement
  them.
- The public repository is the source for code, snapshots, tests, and final
  project. The web repository is private and should keep linking to this public
  repository.

## Lesson 00

Lesson 00 is the first lesson opened by the free-course CTA. It must orient the
student before technical content starts.

It should explain:

- what the course builds and what it does not build;
- which hardware, software, GitHub knowledge, Python knowledge, and PyTorch
  familiarity help the student;
- that the course can be read without installing anything;
- when the local setup is needed;
- how the public GitHub repository is used with the course;
- what `study/lessons/`, `study/snapshots/`, and `final_project/` mean;
- what the central lesson, Graph panel, Mathematics panel, and Programming
  panel are responsible for.

## Italian Writing

Italian is the default website experience. The Italian course must feel written
directly in Italian, not mechanically translated from English.

Rules:

- Use fluent, natural Italian word order.
- Use correct accents: `è`, `perché`, `può`, `più`, `così`, `già`, `cioè`,
  `qual è`.
- Keep standard technical terms in English when Italian would sound forced:
  `token`, `tensor`, `embedding`, `training`, `inference`, `fine-tuning`,
  `Transformer`, `attention`, `LayerNorm`, `dropout`, `optimizer`,
  `checkpoint`, `generation`.
- Avoid over-translating terms that students will find in code, papers, and
  PyTorch documentation.
- Use Italian syntax around English technical terms. Keeping `token`,
  `embedding`, or `checkpoint` does not justify copying English word order.
- Do not use metaphors or object analogies in explanations. Explain the actual
  concept, file, tensor, operation, or model behavior.
- Avoid personification. Data do not "teach", tensors do not "answer", and an
  operation does not "arrive". State that data are used for training, a tensor
  contains a value, or an operation is introduced in a specific lesson.
- Do not use child-oriented labels in course text.
- Prefer short direct paragraphs. A beginner-friendly explanation can still be
  precise.
- Prefer an explicit subject when pronouns such as `questo`, `quello`, or
  `esso` could refer to more than one tensor, operation, or state.
- Use a rhetorical question only when it improves the explanation. Prefer a
  direct statement when the answer is a precise tensor, target, or probability.
- When an English sentence is the shared example, keep it unchanged. The course
  uses `The cat sleeps here.` as a stable worked example.

### Italian Style Review Rubric

When reviewing Italian lessons, check the prose in this order:

1. **Technical correctness.** The sentence must describe the implemented code
   path exactly. Do not make the Italian smoother by weakening a shape, runtime,
   loss, checkpoint, or tokenizer contract.
2. **Natural Italian.** If a sentence follows the English structure too closely,
   rewrite it in normal Italian while keeping the technical term unchanged.
3. **Term stability.** Preserve standard AI/programming terms when students will
   see the same word in code, papers, PyTorch, or GitHub.
4. **Section specificity.** A paragraph should be clearly tied to the current
   lesson. If it could be copied into many lessons with only the bold phrase
   changed, rewrite it.
5. **Reader effort.** Split long technical sentences when they contain multiple
   clauses, tensor shapes, and runtime constraints.
6. **Explicit references.** Every pronoun, comparison, and transition must have
   an unambiguous referent in the same paragraph.
7. **Scope clarity.** State what the lesson implements, what it only
   illustrates, and what remains outside the current lesson.

Preferred forms:

| Concept | Preferred Italian form |
|---|---|
| token | `il token`, `i token`, `token ID` |
| tokenizer | `il tokenizer` |
| vocabulary | `il vocabulary` when tied to code/token IDs; `vocabolario` only for ordinary language |
| tensor | `il tensor`, `i tensor` |
| shape | `la shape` |
| loss | `la loss`, `la training loss`, `la validation loss` |
| logits | `i logits` |
| attention | `l'attention`, `la causal self-attention`, `le head di attention` |
| embedding | `gli embedding`, `l'embedding table` |
| training / inference / generation | `il training`, `l'inference`, `la generation` |
| optimizer / checkpoint / resume | `l'optimizer`, `il checkpoint`, `il resume` |
| LayerNorm / Dropout / GELU | keep the library term unchanged |

Use these wording checks:

- After `Obiettivo:`, prefer an infinitive or a compact noun phrase:
  `calcolare`, `allenare`, `normalizzare`, `stimare`, `salvare`, `ripristinare`.
  Avoid switching between imperative, infinitive, and capitalized verbs across
  lessons.
- In `Dove siamo arrivati`, avoid generic template sentences such as "the
  transformation is complete when...". Write what the current lesson now makes
  possible and what is intentionally still out of scope.
- In `Se ricordi una sola cosa`, state the lesson-specific invariant or
  decision. Avoid generic phrasing about the name of the operation being less
  important than the result.
- Avoid false friends and English role labels unless they are code identifiers.
  Prefer `codice chiamante` to `caller`, `componente che usa il modulo` to
  `consumer`, and `informazione` or `segnale` to `evidenza` unless evidence is
  the intended statistical meaning.
- Avoid mild metaphorical verbs when a direct technical verb works better.
  Prefer `il termine può risultare astratto` to `la parola può nascondere`, and
  `viene introdotto qui perché...` to repeated uses of `arriva`.
- Keep `state` in English only inside code names such as `state_dict`. In prose,
  use `stato`.
- Keep conversational directness, but remove filler such as repeated `adesso`,
  `davvero`, `semplicemente`, and `in questo modo` when it does not add meaning.
- Do not smooth away negative scope statements. Sentences like "questa lezione
  non introduce Dropout" or "questo smoke test non dimostra equivalenza
  numerica" are useful and should remain explicit.
- Introduce an abbreviation, axis, or symbol before using it in a derivation.
  Keep the definition near its first use instead of relying on a later panel.
- Keep the execution order visible. If a paragraph describes multiple
  operations, present them in the same order used by the code.
- Avoid unsupported absolutes such as `sempre`, `mai`, `esatto`, or
  `production-ready` unless the implementation and tests establish that claim.

Before accepting an Italian lesson, review it without looking at the English
source and confirm that:

1. the starting input, operation, output, and invariant are identifiable;
2. every paragraph is specific to the current lesson;
3. the prose follows natural Italian word order;
4. English technical terms use consistent Italian articles and prepositions;
5. accents, apostrophes, agreement, and punctuation are correct;
6. rhetorical questions, personifications, filler, and repeated explanations
   have been removed;
7. code, formulas, numeric values, paths, and tensor shapes still match the
   implemented lesson.

## Bilingual Alignment

- `course_en_graphic.md` remains the structural source used for alignment.
- `course_it_graphic.md` must be reviewed whenever the English source changes.
- Code blocks, paths, tensor shapes, formulas, numeric values, and Mermaid
  identifiers must remain stable across languages unless there is a technical
  reason to change them.
- The Italian source hash must be updated only after the Italian file has been
  reviewed against the English source.
- Do not copy text from external books or PDFs. Use them only to improve
  explanations with reworked, original wording.

## Lesson Structure

Each lesson should keep a consistent teaching path:

1. starting state;
2. problem that the current state cannot solve yet;
3. reason for the new operation;
4. concrete transformation in execution order;
5. resulting state;
6. invariant that remains true;
7. what is still intentionally missing.

The center of the lesson owns the readable explanation. It can include the
smallest useful worked example. Complete derivations belong in Mathematics.
Syntax, diffs, and complete code belong in Programming. System position and
dependencies belong in Graph.

## Lists And Layout

- Use bullet lists when the order is not part of the meaning.
- Use numbered lists when the order is part of the procedure or reasoning.
- Use plain indentation only for custom timelines, tensor flows, or UI
  structures that already provide their own sequence.
- Markdown bullet and numbered lists must render with visible markers.

## Code And Repository Links

- Every lesson should map to the relevant source file or snapshot in this
  repository.
- `study/lessons/` contains the executable lesson scripts.
- `study/snapshots/lesson_XX/` contains the complete project state after a
  specific lesson.
- `final_project/` contains the clean current implementation.
- The website may show a lesson-level GitHub link in the main lesson.
- The Programming panel should show a file-specific GitHub link when a file is
  selected.
- Avoid duplicate GitHub buttons in the same panel.

## Validation Before Publishing

For this repository:

```bash
python -B tools/validate_learngpt.py
python -B -m unittest discover -s tests -v
python -B tools/run_all_lessons.py
```

For the web repository:

```bash
npm run verify:aws
```

When publishing the website, build from synchronized course sources, deploy the
AWS static output, wait for the CloudFront invalidation to complete, and verify
that the live artifacts match the generated AWS build.
