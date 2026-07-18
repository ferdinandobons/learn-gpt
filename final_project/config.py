"""
Changes compared with the previous files:
- This module belongs to the cleaned final LearnGPT project.
- It uses the English public project layout and the FineWeb-Edu/BPE pipeline.

File purpose:
- Define model, training, generation, and performance configuration.
"""

from dataclasses import asdict, dataclass, fields


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

    @classmethod
    def from_checkpoint_dict(cls, payload):
        config = cls(
            vocabulary_size=payload["vocabulary_size"],
            context_size=payload.get("context_size", 32),
            embedding_size=payload.get("embedding_size", 64),
            num_heads=payload.get("num_heads", 4),
            num_transformer_blocks=payload.get("num_transformer_blocks", 2),
            dropout=payload.get("dropout", 0.1),
            tie_weights=payload.get("tie_weights", True),
            use_scaled_dot_product_attention=payload.get(
                "use_scaled_dot_product_attention",
                False,
            ),
        )
        saved_head_size = payload.get("head_size")
        if saved_head_size is not None and saved_head_size != config.head_size:
            raise ValueError(
                "Checkpoint head_size is inconsistent with embedding_size and num_heads."
            )

        return config


@dataclass
class TrainingConfig:
    seed: int = 1337
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
    context_sensitivity_contexts: int = 0
    min_context_js_divergence: float | None = None
    context_gate_start_step: int = 0
    stop_on_low_context_sensitivity: bool = False
    resume_from_checkpoint: bool = False
    compile_model: bool = False
    mixed_precision: bool = False
    precision_dtype: str = "float16"

    def __post_init__(self):
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")

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

        if self.context_sensitivity_contexts < 0:
            raise ValueError("context_sensitivity_contexts cannot be negative.")

        if self.context_sensitivity_contexts == 1:
            raise ValueError(
                "context_sensitivity_contexts must be 0 or at least 2."
            )

        if (
            self.min_context_js_divergence is not None
            and self.min_context_js_divergence <= 0
        ):
            raise ValueError(
                "min_context_js_divergence must be greater than 0 when set."
            )

        if self.context_gate_start_step < 0:
            raise ValueError("context_gate_start_step cannot be negative.")

        if (
            self.stop_on_low_context_sensitivity
            and self.min_context_js_divergence is None
        ):
            raise ValueError(
                "stop_on_low_context_sensitivity requires min_context_js_divergence."
            )

        if (
            self.min_context_js_divergence is not None
            and self.context_sensitivity_contexts < 2
        ):
            raise ValueError(
                "min_context_js_divergence requires at least two context samples."
            )

        if self.precision_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(
                "precision_dtype must be 'float16', 'bfloat16' or 'float32'."
            )

    def to_checkpoint_dict(self):
        return asdict(self)

    @classmethod
    def from_checkpoint_dict(cls, payload):
        if not payload:
            return cls()

        valid_names = {field.name for field in fields(cls)}
        return cls(
            **{
                key: value
                for key, value in payload.items()
                if key in valid_names
            }
        )


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
