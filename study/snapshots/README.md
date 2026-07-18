# Lesson Snapshots

This directory contains one project-code snapshot for each lesson.

Rule:

```text
study/lessons/NN_lesson_name.py -> imports from study/snapshots/lesson_NN/
```

Why this exists:

- older lessons stay runnable;
- the final project can keep evolving in `final_project/`;
- a lesson cannot accidentally depend on a future version of the project.

When a new lesson is created:

1. create a two-digit snapshot directory, for example
   `study/snapshots/lesson_31/`;
2. copy or write only the project files needed by that lesson;
3. update the import paths in the numbered script under `study/lessons`, for
   example `study/lessons/31_generate.py`;
4. update `course_en.md`;
5. update `final_project/` if the final project changes.

Additional rules:

- snapshot `model.py` files should not contain old model classes that are no
  longer used by the current lesson.
- the main model class for each lesson should be named `LanguageModel`.
- support components such as `SelfAttentionHead`, `MultiHeadAttention`,
  `FeedForward`, or `TransformerBlock` should stay in a snapshot only when the
  lesson `LanguageModel` actually needs them.
