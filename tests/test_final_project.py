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
from final_project.config import ModelConfig
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
        next_token_ids = input_ids[:, -1].remainder(self.vocabulary_size)
        logits.scatter_(
            2,
            next_token_ids.view(-1, 1, 1),
            20.0,
        )
        return logits


class ContextSensitivityTests(unittest.TestCase):
    def test_context_metric_distinguishes_static_and_contextual_logits(self):
        validation_data = np.arange(256, dtype=np.int64)
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

    def test_training_stops_when_the_context_gate_fails(self):
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
            with self.assertRaisesRegex(RuntimeError, "Context-sensitivity gate failed"):
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
                    context_sensitivity_contexts=2,
                    min_context_js_divergence=100.0,
                    stop_on_low_context_sensitivity=True,
                    device="cpu",
                )

    def test_context_gate_is_deferred_until_the_configured_step(self):
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
                context_sensitivity_contexts=2,
                min_context_js_divergence=100.0,
                context_gate_start_step=2,
                stop_on_low_context_sensitivity=True,
                device="cpu",
            )

        self.assertFalse(history[-1]["context_gate_active"])
        self.assertIsNone(history[-1]["context_gate_passed"])


class CheckpointTests(unittest.TestCase):
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
            latest = load_checkpoint_payload(latest_path, device="cpu")
            self.assertEqual(latest["step"], 2)
            self.assertIn("context_js_divergence", latest["losses"])
            self.assertIn("context_logit_std", latest["losses"])


if __name__ == "__main__":
    unittest.main()
