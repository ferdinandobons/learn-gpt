"""
Changes compared with the previous files:
- This module belongs to the cleaned final LearnGPT project.
- It uses the English public project layout and the FineWeb-Edu/BPE pipeline.

File purpose:
- Define model, training, generation, and performance configuration.
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
    use_scaled_dot_product_attention: bool = False

    def __post_init__(self):
        if self.vocabulary_size < 1:
            raise ValueError("vocabulary_size must be at least 1.")

        if self.context_size < 1:
            raise ValueError("context_size must be at least 1.")

        if self.embedding_size < 1:
            raise ValueError("embedding_size must be at least 1.")

        if self.num_heads < 1:
            raise ValueError("num_heads must be at least 1.")

        if self.embedding_size % self.num_heads != 0:
            raise ValueError("embedding_size must be divisible by num_heads.")

        if self.num_transformer_blocks < 1:
            raise ValueError("num_transformer_blocks must be at least 1.")

        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0.")

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
    gradient_accumulation_steps: int = 1
    resume_from_checkpoint: bool = False
    compile_model: bool = False
    mixed_precision: bool = False
    precision_dtype: str = "float16"

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if self.training_steps < 1:
            raise ValueError("training_steps must be at least 1.")

        if self.eval_interval < 1:
            raise ValueError("eval_interval must be at least 1.")

        if self.eval_batches < 1:
            raise ValueError("eval_batches must be at least 1.")

        if self.base_learning_rate <= 0:
            raise ValueError("base_learning_rate must be greater than 0.")

        if self.min_learning_rate < 0:
            raise ValueError("min_learning_rate cannot be negative.")

        if self.min_learning_rate > self.base_learning_rate:
            raise ValueError(
                "min_learning_rate must be less than or equal to base_learning_rate."
            )

        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative.")

        if self.decay_steps <= self.warmup_steps:
            raise ValueError("decay_steps must be greater than warmup_steps.")

        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")

        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be greater than 0 when set.")

        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1.")

        if self.precision_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(
                "precision_dtype must be 'float16', 'bfloat16' or 'float32'."
            )

    def to_checkpoint_dict(self):
        return asdict(self)


@dataclass
class GenerationConfig:
    prompt_text: str = "The"
    generated_tokens: int = 40
    temperature: float = 0.9
    top_k: int | None = 50
    num_samples: int = 1

    def __post_init__(self):
        if self.generated_tokens < 0:
            raise ValueError("generated_tokens cannot be negative.")

        if self.temperature <= 0:
            raise ValueError("temperature must be greater than 0.")

        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be greater than 0 when set.")

        if self.num_samples < 1:
            raise ValueError("num_samples must be at least 1.")

    def to_checkpoint_dict(self):
        return asdict(self)
