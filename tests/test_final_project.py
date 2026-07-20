"""Regression tests for the final LearnGPT runtime."""

import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from final_project.batching import load_training_and_validation_data
from final_project.checkpoint import load_checkpoint, load_checkpoint_payload, save_checkpoint
from final_project.config import ModelConfig, TrainingConfig
from final_project.generate import load_model_from_checkpoint
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

            self.write_dataset(data_dir, complete=False)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_training_and_validation_data(data_dir)


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


class CheckpointTests(unittest.TestCase):
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
