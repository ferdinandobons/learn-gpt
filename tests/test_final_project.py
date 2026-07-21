"""Regression tests for the final LearnGPT runtime."""

import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from final_project.batching import (
    create_dataset_fingerprint,
    load_training_and_validation_data,
)
from final_project.checkpoint import (
    load_checkpoint,
    load_checkpoint_payload,
    restore_checkpoint_rng_state,
    save_checkpoint,
)
from final_project.config import ModelConfig, TrainingConfig
from final_project.generate import (
    generate_samples_from_checkpoint,
    load_model_from_checkpoint,
)
from final_project.model import (
    GPT_INITIALIZATION_STD,
    LanguageModel,
    SelfAttentionHead,
)
from final_project.prepare_subset import prepare_subset
from final_project.quality import estimate_context_sensitivity
from final_project.training import (
    configure_optimizer,
    get_learning_rate,
    get_latest_checkpoint_path,
    train_model,
    validate_checkpoint_start,
    validate_resume_dataset,
)


def make_model_config(vocabulary_size=128):
    return ModelConfig(
        vocabulary_size=vocabulary_size,
        context_size=8,
        embedding_size=32,
        num_heads=4,
        num_transformer_blocks=2,
        dropout=0.0,
        tie_weights=True,
        use_scaled_dot_product_attention=True,
    )


class ModelInitializationTests(unittest.TestCase):
    def test_gpt_initialization_starts_near_random_baseline_loss(self):
        torch.manual_seed(123)
        config = make_model_config()
        model = LanguageModel(**config.to_model_kwargs())
        model.eval()

        embedding_std = model.token_embedding_table.weight.std().item()
        self.assertAlmostEqual(embedding_std, GPT_INITIALIZATION_STD, delta=0.003)

        input_ids = torch.randint(0, config.vocabulary_size, (4, config.context_size))
        target_ids = torch.randint(0, config.vocabulary_size, (4, config.context_size))
        with torch.no_grad():
            _, loss = model(input_ids, target_ids)

        self.assertAlmostEqual(
            loss.item(),
            math.log(config.vocabulary_size),
            delta=0.35,
        )

    def test_gpt2_vocabulary_initial_loss_is_near_log_vocabulary_size(self):
        torch.manual_seed(123)
        config = make_model_config(vocabulary_size=50257)
        model = LanguageModel(**config.to_model_kwargs())
        model.eval()
        input_ids = torch.randint(0, config.vocabulary_size, (1, config.context_size))
        target_ids = torch.randint(0, config.vocabulary_size, (1, config.context_size))

        with torch.no_grad():
            _, loss = model(input_ids, target_ids)

        self.assertAlmostEqual(loss.item(), math.log(50257), delta=0.35)

    def test_residual_projections_use_scaled_initialization(self):
        torch.manual_seed(123)
        config = make_model_config()
        model = LanguageModel(**config.to_model_kwargs())
        expected_std = GPT_INITIALIZATION_STD / math.sqrt(
            2 * config.num_transformer_blocks
        )

        for block in model.transformer_blocks:
            self.assertAlmostEqual(
                block.multi_head_attention.output_projection.weight.std().item(),
                expected_std,
                delta=0.002,
            )
            self.assertAlmostEqual(
                block.feed_forward.project.weight.std().item(),
                expected_std,
                delta=0.002,
            )


class NanoGPTArchitectureTests(unittest.TestCase):
    def test_token_embedding_and_output_head_share_weights(self):
        config = make_model_config()
        model = LanguageModel(**config.to_model_kwargs())

        self.assertIs(model.output_head.weight, model.token_embedding_table.weight)

    def test_default_model_keeps_stable_biases_but_not_qkv_biases(self):
        config = make_model_config()
        model = LanguageModel(**config.to_model_kwargs())

        self.assertTrue(config.bias)
        self.assertIsNotNone(model.output_head.bias)
        self.assertIsNone(model.transformer_blocks[0].multi_head_attention.heads[0].key.bias)

    def test_legacy_checkpoint_config_preserves_biases(self):
        payload = make_model_config().to_checkpoint_dict()
        payload.pop("bias")

        restored = ModelConfig.from_checkpoint_dict(payload)
        model = LanguageModel(**restored.to_model_kwargs())

        self.assertTrue(restored.bias)
        self.assertIsNotNone(model.output_head.bias)

    def test_future_tokens_do_not_change_earlier_logits(self):
        torch.manual_seed(123)
        config = make_model_config()
        model = LanguageModel(**config.to_model_kwargs())
        model.eval()
        shared_prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        first_input = torch.cat(
            (shared_prefix, torch.tensor([[5, 6, 7, 8]], dtype=torch.long)),
            dim=1,
        )
        second_input = torch.cat(
            (shared_prefix, torch.tensor([[9, 10, 11, 12]], dtype=torch.long)),
            dim=1,
        )

        with torch.no_grad():
            first_logits, _ = model(first_input, first_input)
            second_logits, _ = model(second_input, second_input)

        self.assertTrue(
            torch.allclose(
                first_logits[:, : shared_prefix.shape[1]],
                second_logits[:, : shared_prefix.shape[1]],
                atol=1e-6,
                rtol=1e-5,
            )
        )

    def test_manual_and_scaled_dot_product_attention_match_in_eval(self):
        torch.manual_seed(123)
        manual = SelfAttentionHead(
            embedding_size=32,
            head_size=8,
            context_size=8,
            dropout=0.0,
            use_scaled_dot_product_attention=False,
        )
        optimized = SelfAttentionHead(
            embedding_size=32,
            head_size=8,
            context_size=8,
            dropout=0.0,
            use_scaled_dot_product_attention=True,
        )
        optimized.load_state_dict(manual.state_dict())
        manual.eval()
        optimized.eval()
        embeddings = torch.randn(2, 8, 32)

        with torch.no_grad():
            manual_output, _ = manual(embeddings)
            optimized_output, _ = optimized(embeddings)

        self.assertTrue(
            torch.allclose(manual_output, optimized_output, atol=1e-6, rtol=1e-5)
        )

    def test_chunked_and_monolithic_output_projection_match(self):
        torch.manual_seed(123)
        config = make_model_config()
        config.output_chunk_size = 32
        chunked = LanguageModel(**config.to_model_kwargs())
        monolithic_config = ModelConfig.from_checkpoint_dict(
            config.to_checkpoint_dict()
        )
        monolithic_config.output_chunk_size = 0
        monolithic = LanguageModel(**monolithic_config.to_model_kwargs())
        monolithic.load_state_dict(chunked.state_dict())
        input_ids = torch.randint(
            0,
            config.vocabulary_size,
            (2, config.context_size),
        )

        chunked_logits, chunked_loss = chunked(input_ids, input_ids)
        monolithic_logits, monolithic_loss = monolithic(input_ids, input_ids)

        self.assertTrue(torch.equal(chunked_logits, monolithic_logits))
        self.assertEqual(chunked_loss.item(), monolithic_loss.item())
        chunked_loss.backward()
        monolithic_loss.backward()
        for chunked_parameter, monolithic_parameter in zip(
            chunked.parameters(),
            monolithic.parameters(),
        ):
            self.assertTrue(
                torch.allclose(
                    chunked_parameter.grad,
                    monolithic_parameter.grad,
                    atol=1e-7,
                    rtol=1e-6,
                )
            )

    def test_learning_rate_uses_warmup_cosine_decay_and_minimum(self):
        settings = {
            "base_learning_rate": 1e-3,
            "min_learning_rate": 1e-4,
            "warmup_steps": 10,
            "decay_steps": 100,
        }

        self.assertEqual(get_learning_rate(step=0, **settings), 0.0)
        self.assertAlmostEqual(get_learning_rate(step=10, **settings), 1e-3)
        self.assertGreater(
            get_learning_rate(step=50, **settings),
            get_learning_rate(step=90, **settings),
        )
        self.assertAlmostEqual(get_learning_rate(step=100, **settings), 1e-4)
        self.assertAlmostEqual(get_learning_rate(step=101, **settings), 1e-4)


class DatasetValidationTests(unittest.TestCase):
    def write_dataset(self, data_dir, complete=True, encoding_name="gpt2"):
        train = np.arange(128, dtype=np.uint16)
        validation = np.arange(64, dtype=np.uint16)
        train.tofile(data_dir / "train.bin")
        validation.tofile(data_dir / "val.bin")
        metadata = {
            "complete": complete,
            "dtype": "uint16",
            "encoding_name": encoding_name,
            "counters": {
                "train_tokens": len(train),
                "val_tokens": len(validation),
            },
        }
        (data_dir / "meta.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

    def test_dataset_metadata_must_be_complete_and_match_tokenizer(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            data_dir = Path(temporary_dir)
            self.write_dataset(data_dir)

            train, validation = load_training_and_validation_data(
                data_dir,
                encoding_name="gpt2",
            )
            self.assertEqual(len(train), 128)
            self.assertEqual(len(validation), 64)

            with self.assertRaisesRegex(ValueError, "Tokenizer mismatch"):
                load_training_and_validation_data(
                    data_dir,
                    encoding_name="cl100k_base",
                )

            # NumPy memmaps keep the files open. Release them before rewriting
            # the fixture so this lifecycle is valid on Windows as well as on
            # POSIX systems.
            del train, validation
            self.write_dataset(data_dir, complete=False)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_training_and_validation_data(data_dir)

    def test_dataset_fingerprint_is_path_independent_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first_dir = root / "first"
            second_dir = root / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            self.write_dataset(first_dir)
            self.write_dataset(second_dir)

            first = create_dataset_fingerprint(first_dir)
            second = create_dataset_fingerprint(second_dir)
            self.assertEqual(first["value"], second["value"])

            changed = np.memmap(
                second_dir / "train.bin",
                dtype=np.uint16,
                mode="r+",
            )
            changed[17] += 1
            changed.flush()
            del changed

            modified = create_dataset_fingerprint(second_dir)
            self.assertNotEqual(first["value"], modified["value"])


class ExperimentalSubsetTests(unittest.TestCase):
    def write_source_dataset(self, data_dir):
        training_data = np.arange(4096, dtype=np.uint16) % 128
        validation_data = np.arange(2048, dtype=np.uint16) % 128
        training_data.tofile(data_dir / "train.bin")
        validation_data.tofile(data_dir / "val.bin")
        (data_dir / "meta.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "dtype": "uint16",
                    "encoding_name": "gpt2",
                    "dataset_name": "test-dataset",
                    "dataset_config": "test-config",
                    "counters": {
                        "train_tokens": len(training_data),
                        "val_tokens": len(validation_data),
                    },
                }
            ),
            encoding="utf-8",
        )

    def make_arguments(self, source_dir, output_dir):
        target_tokens = 512
        return SimpleNamespace(
            source_data_dir=source_dir,
            output_dir=output_dir,
            target_gb=(target_tokens * np.dtype(np.uint16).itemsize) / 1024**3,
            validation_ratio=0.25,
            seed=123,
            chunk_tokens=32,
            overwrite=False,
        )

    def test_subset_is_complete_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_dir = root / "source"
            first_output_dir = root / "first"
            second_output_dir = root / "second"
            source_dir.mkdir()
            self.write_source_dataset(source_dir)

            first_metadata = prepare_subset(
                self.make_arguments(source_dir, first_output_dir)
            )
            second_metadata = prepare_subset(
                self.make_arguments(source_dir, second_output_dir)
            )

            self.assertTrue(first_metadata["complete"])
            self.assertEqual(first_metadata["counters"]["train_tokens"], 384)
            self.assertEqual(first_metadata["counters"]["val_tokens"], 128)
            self.assertEqual(first_metadata["subset_seed"], 123)
            self.assertEqual(first_metadata["preparation_mode"], "randomized_subset")
            self.assertTrue(second_metadata["complete"])
            self.assertEqual(
                (first_output_dir / "train.bin").read_bytes(),
                (second_output_dir / "train.bin").read_bytes(),
            )
            self.assertEqual(
                (first_output_dir / "val.bin").read_bytes(),
                (second_output_dir / "val.bin").read_bytes(),
            )


class StaticLogitModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.vocabulary_size = vocabulary_size

    def forward(self, input_ids):
        return torch.zeros(
            input_ids.shape[0],
            1,
            self.vocabulary_size,
            device=input_ids.device,
        )


class ContextualLogitModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.vocabulary_size = vocabulary_size

    def forward(self, input_ids):
        logits = torch.full(
            (input_ids.shape[0], 1, self.vocabulary_size),
            -20.0,
            device=input_ids.device,
        )
        next_token_ids = (input_ids[:, -1] + 1).remainder(self.vocabulary_size)
        logits.scatter_(
            2,
            next_token_ids.view(-1, 1, 1),
            20.0,
        )
        return logits


class ContextSensitivityTests(unittest.TestCase):
    def test_context_metric_distinguishes_static_and_contextual_logits(self):
        validation_data = np.arange(256, dtype=np.int64) % 32
        static_metrics = estimate_context_sensitivity(
            model=StaticLogitModel(vocabulary_size=32),
            validation_data=validation_data,
            context_size=8,
            num_contexts=8,
            device="cpu",
        )
        contextual_metrics = estimate_context_sensitivity(
            model=ContextualLogitModel(vocabulary_size=32),
            validation_data=validation_data,
            context_size=8,
            num_contexts=8,
            device="cpu",
        )

        self.assertLess(static_metrics["context_js_divergence"], 1e-8)
        self.assertGreater(contextual_metrics["context_js_divergence"], 1.0)
        self.assertGreater(
            contextual_metrics["context_logit_std"],
            static_metrics["context_logit_std"],
        )
        self.assertAlmostEqual(static_metrics["context_loss_gain"], 0.0, places=6)
        self.assertGreater(contextual_metrics["context_loss_gain"], 1.0)
        self.assertLess(
            contextual_metrics["context_true_loss"],
            contextual_metrics["context_shuffled_loss"],
        )


class GradientIntegrityTests(unittest.TestCase):
    class FakeOverflowGradientScaler:
        def __init__(self, overflows=1):
            self.current_scale = 65_536.0
            self.found_inf = False
            self.overflows_remaining = overflows

        def is_enabled(self):
            return True

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            self.found_inf = False
            if self.overflows_remaining == 0:
                return
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.reshape(-1)[0] = float("inf")
                        self.found_inf = True
                        self.overflows_remaining -= 1
                        return
            raise AssertionError("The overflow test found no gradient.")

        def step(self, optimizer):
            if not self.found_inf:
                optimizer.step()

        def update(self):
            if self.found_inf:
                self.current_scale *= 0.5
            self.found_inf = False

        def get_scale(self):
            return self.current_scale

        def state_dict(self):
            return {"scale": self.current_scale, "growth_tracker": 0}

        def load_state_dict(self, state_dict):
            self.current_scale = state_dict["scale"]

    def test_retry_configuration_requires_a_raw_norm_limit(self):
        with self.assertRaisesRegex(
            ValueError,
            "requires max_grad_norm_before_clip",
        ):
            TrainingConfig(gradient_retry_attempts=1)

    def test_training_rejects_invalid_gradient_before_an_update(self):
        torch.manual_seed(123)
        config = make_model_config(vocabulary_size=32)
        model = LanguageModel(**config.to_model_kwargs())
        initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer = configure_optimizer(
            model,
            learning_rate=1e-3,
            weight_decay=0.1,
            device="cpu",
        )
        training_data = np.arange(512, dtype=np.int64) % config.vocabulary_size
        validation_data = np.arange(256, dtype=np.int64) % config.vocabulary_size

        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaisesRegex(RuntimeError, "Gradient integrity check failed"):
                train_model(
                    model=model,
                    optimizer=optimizer,
                    training_data=training_data,
                    validation_data=validation_data,
                    batch_size=2,
                    context_size=config.context_size,
                    training_steps=1,
                    eval_interval=1,
                    eval_batches=1,
                    checkpoint_path=Path(temporary_dir) / "best.pt",
                    model_config=config.to_checkpoint_dict(),
                    tokenizer_config={"encoding_name": "gpt2"},
                    base_learning_rate=1e-3,
                    min_learning_rate=1e-4,
                    warmup_steps=0,
                    decay_steps=1,
                    gradient_clip=1.0,
                    max_grad_norm_before_clip=1e-12,
                    gradient_retry_attempts=1,
                    device="cpu",
                )

        for initial, current in zip(initial_parameters, model.parameters()):
            self.assertTrue(torch.equal(initial, current))

    def test_training_rejects_nonfinite_loss_before_an_update(self):
        class NonFiniteLossModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))

            def forward(self, input_ids, target_ids=None):
                logits = self.weight * torch.ones(
                    (*input_ids.shape, 32),
                    device=input_ids.device,
                )
                if target_ids is None:
                    return logits
                finite_zero_gradient = self.weight * 0.0
                loss = finite_zero_gradient + torch.tensor(
                    float("inf"),
                    device=input_ids.device,
                )
                return logits, loss

        model = NonFiniteLossModel()
        initial_weight = model.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        training_data = np.arange(128, dtype=np.int64) % 32
        validation_data = np.arange(64, dtype=np.int64) % 32

        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaisesRegex(RuntimeError, "Non-finite training loss"):
                train_model(
                    model=model,
                    optimizer=optimizer,
                    training_data=training_data,
                    validation_data=validation_data,
                    batch_size=1,
                    context_size=8,
                    training_steps=1,
                    eval_interval=1,
                    eval_batches=1,
                    checkpoint_path=Path(temporary_dir) / "best.pt",
                    model_config={"vocabulary_size": 32},
                    tokenizer_config={"encoding_name": "gpt2"},
                    base_learning_rate=1e-3,
                    min_learning_rate=1e-4,
                    warmup_steps=0,
                    decay_steps=1,
                    gradient_clip=None,
                    max_grad_norm_before_clip=None,
                    gradient_retry_attempts=0,
                    device="cpu",
                )

        self.assertTrue(torch.equal(initial_weight, model.weight.detach()))

    def test_training_rejects_nonfinite_evaluation_before_checkpointing(self):
        class NonFiniteEvaluationModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))

            def forward(self, input_ids, target_ids=None):
                logits = self.weight * torch.ones(
                    (*input_ids.shape, 32),
                    device=input_ids.device,
                )
                if target_ids is None:
                    return logits
                loss_value = float("inf") if not self.training else 1.0
                loss = self.weight * 0.0 + torch.tensor(
                    loss_value,
                    device=input_ids.device,
                )
                return logits, loss

        model = NonFiniteEvaluationModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        training_data = np.arange(128, dtype=np.int64) % 32
        validation_data = np.arange(64, dtype=np.int64) % 32

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "best.pt"
            with self.assertRaisesRegex(RuntimeError, "Non-finite training loss"):
                train_model(
                    model=model,
                    optimizer=optimizer,
                    training_data=training_data,
                    validation_data=validation_data,
                    batch_size=1,
                    context_size=8,
                    training_steps=1,
                    eval_interval=1,
                    eval_batches=1,
                    checkpoint_path=checkpoint_path,
                    model_config={"vocabulary_size": 32},
                    tokenizer_config={"encoding_name": "gpt2"},
                    base_learning_rate=1e-3,
                    min_learning_rate=1e-4,
                    warmup_steps=0,
                    decay_steps=1,
                    gradient_clip=None,
                    max_grad_norm_before_clip=None,
                    gradient_retry_attempts=0,
                    device="cpu",
                )

            self.assertFalse(checkpoint_path.exists())
            self.assertFalse(get_latest_checkpoint_path(checkpoint_path).exists())

    def test_cuda_amp_overflow_retries_same_step_and_reduces_scale(self):
        torch.manual_seed(123)
        config = make_model_config(vocabulary_size=32)
        model = LanguageModel(**config.to_model_kwargs())
        initial_parameters = [
            parameter.detach().clone() for parameter in model.parameters()
        ]
        optimizer = configure_optimizer(
            model,
            learning_rate=1e-3,
            weight_decay=0.1,
            device="cpu",
        )
        training_data = np.arange(512, dtype=np.int64) % config.vocabulary_size
        validation_data = np.arange(256, dtype=np.int64) % config.vocabulary_size
        scaler = self.FakeOverflowGradientScaler()

        with tempfile.TemporaryDirectory() as temporary_dir:
            with patch(
                "final_project.training.create_gradient_scaler",
                return_value=scaler,
            ):
                history, _ = train_model(
                    model=model,
                    optimizer=optimizer,
                    training_data=training_data,
                    validation_data=validation_data,
                    batch_size=2,
                    context_size=config.context_size,
                    training_steps=1,
                    eval_interval=1,
                    eval_batches=1,
                    checkpoint_path=Path(temporary_dir) / "best.pt",
                    model_config=config.to_checkpoint_dict(),
                    tokenizer_config={"encoding_name": "gpt2"},
                    base_learning_rate=1e-3,
                    min_learning_rate=1e-4,
                    warmup_steps=0,
                    decay_steps=1,
                    gradient_clip=1.0,
                    max_grad_norm_before_clip=100.0,
                    gradient_retry_attempts=0,
                    mixed_precision=True,
                    precision_dtype="float16",
                    device="cpu",
                )

        self.assertEqual(history[0]["amp_overflow_retries"], 1)
        self.assertEqual(history[0]["amp_overflow_count"], 1)
        self.assertEqual(scaler.get_scale(), 32_768.0)
        self.assertTrue(
            any(
                not torch.equal(initial, current)
                for initial, current in zip(initial_parameters, model.parameters())
            )
        )

    def test_cuda_amp_retry_still_rejects_a_finite_unsafe_gradient(self):
        torch.manual_seed(123)
        config = make_model_config(vocabulary_size=32)
        model = LanguageModel(**config.to_model_kwargs())
        initial_parameters = [
            parameter.detach().clone() for parameter in model.parameters()
        ]
        optimizer = configure_optimizer(
            model,
            learning_rate=1e-3,
            weight_decay=0.1,
            device="cpu",
        )
        training_data = np.arange(512, dtype=np.int64) % config.vocabulary_size
        validation_data = np.arange(256, dtype=np.int64) % config.vocabulary_size
        scaler = self.FakeOverflowGradientScaler()

        with tempfile.TemporaryDirectory() as temporary_dir:
            with patch(
                "final_project.training.create_gradient_scaler",
                return_value=scaler,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Gradient integrity check failed",
                ):
                    train_model(
                        model=model,
                        optimizer=optimizer,
                        training_data=training_data,
                        validation_data=validation_data,
                        batch_size=2,
                        context_size=config.context_size,
                        training_steps=1,
                        eval_interval=1,
                        eval_batches=1,
                        checkpoint_path=Path(temporary_dir) / "best.pt",
                        model_config=config.to_checkpoint_dict(),
                        tokenizer_config={"encoding_name": "gpt2"},
                        base_learning_rate=1e-3,
                        min_learning_rate=1e-4,
                        warmup_steps=0,
                        decay_steps=1,
                        gradient_clip=1.0,
                        max_grad_norm_before_clip=1e-12,
                        gradient_retry_attempts=0,
                        mixed_precision=True,
                        precision_dtype="float16",
                        device="cpu",
                    )

        self.assertEqual(scaler.get_scale(), 32_768.0)
        for initial, current in zip(initial_parameters, model.parameters()):
            self.assertTrue(torch.equal(initial, current))

    def test_cuda_amp_persistent_overflow_fails_without_update(self):
        torch.manual_seed(123)
        config = make_model_config(vocabulary_size=32)
        model = LanguageModel(**config.to_model_kwargs())
        initial_parameters = [
            parameter.detach().clone() for parameter in model.parameters()
        ]
        optimizer = configure_optimizer(
            model,
            learning_rate=1e-3,
            weight_decay=0.1,
            device="cpu",
        )
        training_data = np.arange(512, dtype=np.int64) % config.vocabulary_size
        validation_data = np.arange(256, dtype=np.int64) % config.vocabulary_size
        scaler = self.FakeOverflowGradientScaler(overflows=20)

        with tempfile.TemporaryDirectory() as temporary_dir:
            with patch(
                "final_project.training.create_gradient_scaler",
                return_value=scaler,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "CUDA AMP overflow persisted",
                ):
                    train_model(
                        model=model,
                        optimizer=optimizer,
                        training_data=training_data,
                        validation_data=validation_data,
                        batch_size=2,
                        context_size=config.context_size,
                        training_steps=1,
                        eval_interval=1,
                        eval_batches=1,
                        checkpoint_path=Path(temporary_dir) / "best.pt",
                        model_config=config.to_checkpoint_dict(),
                        tokenizer_config={"encoding_name": "gpt2"},
                        base_learning_rate=1e-3,
                        min_learning_rate=1e-4,
                        warmup_steps=0,
                        decay_steps=1,
                        gradient_clip=1.0,
                        max_grad_norm_before_clip=100.0,
                        gradient_retry_attempts=0,
                        mixed_precision=True,
                        precision_dtype="float16",
                        device="cpu",
                    )

        self.assertEqual(len(optimizer.state), 0)
        for initial, current in zip(initial_parameters, model.parameters()):
            self.assertTrue(torch.equal(initial, current))


class CheckpointTests(unittest.TestCase):
    class FakeGradientScaler:
        def __init__(self, scale):
            self.scale = scale

        def state_dict(self):
            return {"scale": self.scale, "growth_tracker": 7}

        def load_state_dict(self, state_dict):
            self.scale = state_dict["scale"]

    class FakeDisabledGradientScaler:
        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            raise AssertionError("An empty scaler state must not be loaded.")

    def test_cuda_rng_state_is_moved_to_cpu_before_restore(self):
        expected_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
        mapped_state = SimpleNamespace(cpu=lambda: expected_state)

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.set_rng_state_all") as set_rng_state_all,
        ):
            restore_checkpoint_rng_state(
                {"rng_state": {"cuda": [mapped_state]}},
            )

        set_rng_state_all.assert_called_once_with([expected_state])

    def test_fresh_training_protects_existing_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "model.pt"
            checkpoint_path.write_bytes(b"existing")

            with self.assertRaisesRegex(FileExistsError, "would overwrite"):
                validate_checkpoint_start(checkpoint_path)

            validate_checkpoint_start(
                checkpoint_path,
                overwrite_checkpoints=True,
            )

    def test_resume_requires_one_complete_checkpoint_family(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "model.pt"
            latest_path = get_latest_checkpoint_path(checkpoint_path)
            checkpoint_path.write_bytes(b"best")
            latest_path.write_bytes(b"latest")

            validate_checkpoint_start(
                checkpoint_path=checkpoint_path,
                resume_checkpoint_path=latest_path,
            )

            other_latest = Path(temporary_dir) / "other-latest.pt"
            other_latest.write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "same family"):
                validate_checkpoint_start(
                    checkpoint_path=checkpoint_path,
                    resume_checkpoint_path=other_latest,
                )

            checkpoint_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "existing best and latest"):
                validate_checkpoint_start(
                    checkpoint_path=checkpoint_path,
                    resume_checkpoint_path=latest_path,
                )

    def test_resume_matches_uninterrupted_training_with_dropout(self):
        config = make_model_config(vocabulary_size=32)
        config.dropout = 0.2
        training_data = np.arange(512, dtype=np.int64) % config.vocabulary_size
        validation_data = np.arange(256, dtype=np.int64) % config.vocabulary_size

        def create_model_and_optimizer(seed):
            torch.manual_seed(seed)
            model = LanguageModel(**config.to_model_kwargs())
            optimizer = configure_optimizer(
                model,
                learning_rate=1e-3,
                weight_decay=0.1,
                device="cpu",
            )
            return model, optimizer

        def run(model, optimizer, checkpoint_path, training_steps, resume=None):
            return train_model(
                model=model,
                optimizer=optimizer,
                training_data=training_data,
                validation_data=validation_data,
                batch_size=2,
                context_size=config.context_size,
                training_steps=training_steps,
                eval_interval=2,
                eval_batches=2,
                checkpoint_path=checkpoint_path,
                model_config=config.to_checkpoint_dict(),
                tokenizer_config={"encoding_name": "gpt2"},
                base_learning_rate=1e-3,
                min_learning_rate=1e-4,
                warmup_steps=1,
                decay_steps=4,
                gradient_clip=1.0,
                max_grad_norm_before_clip=100.0,
                gradient_retry_attempts=0,
                resume_checkpoint_path=resume,
                device="cpu",
            )

        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_dir = Path(temporary_dir)
            uninterrupted_path = temporary_dir / "uninterrupted.pt"
            split_path = temporary_dir / "split.pt"

            uninterrupted_model, uninterrupted_optimizer = (
                create_model_and_optimizer(123)
            )
            run(
                uninterrupted_model,
                uninterrupted_optimizer,
                uninterrupted_path,
                training_steps=4,
            )

            split_model, split_optimizer = create_model_and_optimizer(123)
            run(
                split_model,
                split_optimizer,
                split_path,
                training_steps=2,
            )
            resumed_model, resumed_optimizer = create_model_and_optimizer(999)
            run(
                resumed_model,
                resumed_optimizer,
                split_path,
                training_steps=4,
                resume=get_latest_checkpoint_path(split_path),
            )

            uninterrupted_payload = load_checkpoint_payload(
                get_latest_checkpoint_path(uninterrupted_path),
                device="cpu",
            )
            resumed_payload = load_checkpoint_payload(
                get_latest_checkpoint_path(split_path),
                device="cpu",
            )

        self.assertEqual(uninterrupted_payload["step"], 4)
        self.assertEqual(resumed_payload["step"], 4)
        for name, parameter in uninterrupted_payload["model_state_dict"].items():
            self.assertTrue(
                torch.equal(parameter, resumed_payload["model_state_dict"][name]),
                msg=f"Parameter differs after resume: {name}",
            )
        self.assertTrue(
            torch.equal(
                uninterrupted_payload["rng_state"]["cpu"],
                resumed_payload["rng_state"]["cpu"],
            )
        )

    def test_resume_rejects_a_different_dataset_fingerprint(self):
        checkpoint = {"dataset_fingerprint": {"value": "first"}}
        current = {"value": "second"}

        with self.assertRaisesRegex(RuntimeError, "different token files"):
            validate_resume_dataset(checkpoint, current)

        self.assertEqual(
            validate_resume_dataset(
                checkpoint,
                current,
                allow_data_change=True,
            ),
            "changed-by-user",
        )

    def test_checkpoint_restores_gradient_scaler_state(self):
        torch.manual_seed(123)
        config = make_model_config()
        model = LanguageModel(**config.to_model_kwargs())
        optimizer = configure_optimizer(
            model,
            learning_rate=1e-3,
            weight_decay=0.1,
            device="cpu",
        )
        saved_scaler = self.FakeGradientScaler(scale=4096.0)

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "scaler.pt"
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                model_config=config.to_checkpoint_dict(),
                step=1,
                losses={"training": 1.0, "validation": 1.0},
                tokenizer_config={"encoding_name": "gpt2"},
                gradient_scaler=saved_scaler,
            )

            restored_scaler = self.FakeGradientScaler(scale=1.0)
            load_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                device="cpu",
                gradient_scaler=restored_scaler,
            )

        self.assertEqual(restored_scaler.scale, 4096.0)

    def test_checkpoint_ignores_disabled_gradient_scaler_state(self):
        torch.manual_seed(123)
        config = make_model_config()
        model = LanguageModel(**config.to_model_kwargs())
        optimizer = configure_optimizer(
            model,
            learning_rate=1e-3,
            weight_decay=0.1,
            device="cpu",
        )
        disabled_scaler = self.FakeDisabledGradientScaler()

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "disabled-scaler.pt"
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                model_config=config.to_checkpoint_dict(),
                step=1,
                losses={"training": 1.0, "validation": 1.0},
                tokenizer_config={"encoding_name": "gpt2"},
                gradient_scaler=disabled_scaler,
            )
            payload = load_checkpoint_payload(checkpoint_path, device="cpu")
            self.assertIsNone(payload["gradient_scaler_state_dict"])
            load_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                device="cpu",
                gradient_scaler=disabled_scaler,
            )

    def test_generation_seed_reproduces_the_same_sample(self):
        torch.manual_seed(123)
        config = make_model_config(vocabulary_size=50_257)
        model = LanguageModel(**config.to_model_kwargs())
        optimizer = configure_optimizer(
            model,
            learning_rate=1e-3,
            weight_decay=0.1,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "generation.pt"
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                model_config=config.to_checkpoint_dict(),
                step=1,
                losses={"training": 1.0, "validation": 1.0},
                tokenizer_config={"encoding_name": "gpt2"},
            )
            first, _ = generate_samples_from_checkpoint(
                checkpoint_path=checkpoint_path,
                prompt_text="The",
                max_new_tokens=8,
                num_samples=2,
                temperature=0.8,
                top_k=40,
                device="cpu",
                seed=777,
            )
            second, _ = generate_samples_from_checkpoint(
                checkpoint_path=checkpoint_path,
                prompt_text="The",
                max_new_tokens=8,
                num_samples=2,
                temperature=0.8,
                top_k=40,
                device="cpu",
                seed=777,
            )

        self.assertEqual(first, second)

    def test_generation_seed_is_independent_of_loader_rng_consumption(self):
        torch.manual_seed(123)
        config = make_model_config(vocabulary_size=50_257)
        model = LanguageModel(**config.to_model_kwargs())
        optimizer = configure_optimizer(
            model,
            learning_rate=1e-3,
            weight_decay=0.1,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "generation-loader-rng.pt"
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                model_config=config.to_checkpoint_dict(),
                step=1,
                losses={"training": 1.0, "validation": 1.0},
                tokenizer_config={"encoding_name": "gpt2"},
            )
            original_loader = load_model_from_checkpoint
            loader_calls = 0

            def noisy_loader(*args, **kwargs):
                nonlocal loader_calls
                loader_calls += 1
                torch.rand(loader_calls * 17)
                return original_loader(*args, **kwargs)

            with patch(
                "final_project.generate.load_model_from_checkpoint",
                side_effect=noisy_loader,
            ):
                first, _ = generate_samples_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    prompt_text="The",
                    max_new_tokens=8,
                    num_samples=1,
                    temperature=0.8,
                    top_k=40,
                    device="cpu",
                    seed=777,
                )
                second, _ = generate_samples_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    prompt_text="The",
                    max_new_tokens=8,
                    num_samples=1,
                    temperature=0.8,
                    top_k=40,
                    device="cpu",
                    seed=777,
                )

        self.assertEqual(first, second)

    def test_legacy_bias_checkpoint_still_loads_for_generation(self):
        torch.manual_seed(123)
        config = make_model_config()
        config.bias = True
        model = LanguageModel(**config.to_model_kwargs())
        optimizer = configure_optimizer(
            model,
            learning_rate=1e-3,
            weight_decay=0.1,
            device="cpu",
        )
        legacy_config = config.to_checkpoint_dict()
        legacy_config.pop("bias")

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "legacy.pt"
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                model_config=legacy_config,
                step=1,
                losses={"training": 1.0, "validation": 1.0},
                tokenizer_config={"encoding_name": "gpt2"},
            )
            loaded_model, _ = load_model_from_checkpoint(
                checkpoint_path,
                device="cpu",
            )

        self.assertIsNotNone(loaded_model.output_head.bias)

    def test_compiled_checkpoint_uses_canonical_model_keys(self):
        torch.manual_seed(123)
        config = make_model_config()
        model = LanguageModel(**config.to_model_kwargs())
        compiled_model = torch.compile(model, backend="eager")
        optimizer = configure_optimizer(
            compiled_model,
            learning_rate=1e-3,
            weight_decay=0.1,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "compiled.pt"
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=compiled_model,
                optimizer=optimizer,
                model_config=config.to_checkpoint_dict(),
                step=1,
                losses={"training": 1.0, "validation": 1.0},
                tokenizer_config={"encoding_name": "gpt2"},
            )
            checkpoint = load_checkpoint_payload(checkpoint_path, device="cpu")
            self.assertFalse(
                any(
                    key.startswith("_orig_mod.")
                    for key in checkpoint["model_state_dict"]
                )
            )

            plain_model = LanguageModel(**config.to_model_kwargs())
            load_checkpoint(
                checkpoint_path,
                model=plain_model,
                device="cpu",
                restore_rng_state=False,
            )
            for expected, actual in zip(model.parameters(), plain_model.parameters()):
                self.assertTrue(torch.equal(expected, actual))

    def test_training_saves_best_and_latest_checkpoints(self):
        torch.manual_seed(123)
        config = make_model_config(vocabulary_size=32)
        model = LanguageModel(**config.to_model_kwargs())
        optimizer = configure_optimizer(
            model,
            learning_rate=1e-3,
            weight_decay=0.1,
            device="cpu",
        )
        training_data = np.arange(512, dtype=np.int64) % config.vocabulary_size
        validation_data = np.arange(256, dtype=np.int64) % config.vocabulary_size

        with tempfile.TemporaryDirectory() as temporary_dir:
            best_path = Path(temporary_dir) / "best.pt"
            history, returned_best_path = train_model(
                model=model,
                optimizer=optimizer,
                training_data=training_data,
                validation_data=validation_data,
                batch_size=2,
                context_size=config.context_size,
                training_steps=2,
                eval_interval=2,
                eval_batches=1,
                checkpoint_path=best_path,
                model_config=config.to_checkpoint_dict(),
                tokenizer_config={"encoding_name": "gpt2"},
                base_learning_rate=1e-3,
                min_learning_rate=1e-4,
                warmup_steps=0,
                decay_steps=2,
                gradient_clip=1.0,
                context_sensitivity_contexts=2,
                device="cpu",
            )

            latest_path = get_latest_checkpoint_path(best_path)
            self.assertEqual(returned_best_path, best_path)
            self.assertTrue(best_path.exists())
            self.assertTrue(latest_path.exists())
            self.assertEqual(len(history), 2)
            self.assertIn("context_js_divergence", history[-1])
            self.assertIn("context_logit_std", history[-1])
            self.assertIn("context_loss_gain", history[-1])
            self.assertIn("gradient_retries", history[-1])
            latest = load_checkpoint_payload(latest_path, device="cpu")
            self.assertEqual(latest["step"], 2)
            self.assertIn("context_js_divergence", latest["losses"])
            self.assertIn("context_logit_std", latest["losses"])
            self.assertIn("context_loss_gain", latest["losses"])
            self.assertIn("gradient_retries", latest["losses"])
            self.assertIn("torch_version", latest["runtime"])


if __name__ == "__main__":
    unittest.main()
