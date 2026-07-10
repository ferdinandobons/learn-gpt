"""Regression tests for the final LearnGPT runtime."""

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from final_project.batching import load_training_and_validation_data
from final_project.checkpoint import load_checkpoint, load_checkpoint_payload, save_checkpoint
from final_project.config import ModelConfig
from final_project.model import GPT_INITIALIZATION_STD, LanguageModel
from final_project.training import (
    configure_optimizer,
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
                device="cpu",
            )

            latest_path = get_latest_checkpoint_path(best_path)
            self.assertEqual(returned_best_path, best_path)
            self.assertTrue(best_path.exists())
            self.assertTrue(latest_path.exists())
            self.assertEqual(len(history), 2)
            latest = load_checkpoint_payload(latest_path, device="cpu")
            self.assertEqual(latest["step"], 2)


if __name__ == "__main__":
    unittest.main()
