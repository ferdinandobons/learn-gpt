"""
Changes compared with the previous files:
- This module is part of the project-code snapshot used by lesson 39.
- It keeps the lesson independent from future changes in `final_project`.

File purpose:
- Provide the code needed by the matching lesson script.
- Preserve a stable reference point for the course examples.
"""

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocabulary_size: int
    context_size: int = 32
    embedding_size: int = 64
    num_heads: int = 4
    num_transformer_blocks: int = 2
    dropout: float = 0.1
    tie_weights: bool = True

    @property
    def head_size(self):
        return self.embedding_size // self.num_heads

    def to_model_kwargs(self):
        payload = asdict(self)
        payload["head_size"] = self.head_size

        return payload

    def to_checkpoint_dict(self):
        return self.to_model_kwargs()


@dataclass
class TrainingConfig:
    batch_size: int = 4
    training_steps: int = 3
    eval_interval: int = 1
    eval_batches: int = 1
    base_learning_rate: float = 0.001
    min_learning_rate: float = 0.0001
    warmup_steps: int = 1
    decay_steps: int = 10
    weight_decay: float = 0.01
    gradient_clip: float | None = 1.0
    gradient_accumulation_steps: int = 2
    resume_from_checkpoint: bool = False

    def to_checkpoint_dict(self):
        return asdict(self)


@dataclass
class GenerationConfig:
    prompt_text: str = "The"
    generated_tokens: int = 40
    temperature: float = 0.9
    top_k: int | None = 50
    num_samples: int = 1

    def to_checkpoint_dict(self):
        return asdict(self)
