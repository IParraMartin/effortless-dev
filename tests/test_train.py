"""Tests for the training building blocks.

Hermetic and CPU-only: the learning-rate schedule, the decay/no-decay optimizer
split, checkpoint round-tripping, the resumable data cursor, and a real
``evaluate`` pass over a fixture ``.bin``. The full ``main`` is not driven here
because it loads a real tokenizer; its pieces are tested directly instead.
"""

from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from src.config import TrainConfig, TransformerConfig
from src.model import Transformer
from training.data import StatelessBlockSampler, build_dataloader
from training.train import (
    build_optimizer,
    evaluate,
    learning_rate_at,
    save_checkpoint,
)


def tiny_model_config(**updates) -> TransformerConfig:
    values = dict(
        vocab_size=64, d_model=32, n_layers=3, n_heads=4, n_kv_heads=2,
        max_seq_len=8, dropout=0.0,
    )
    values.update(updates)
    return TransformerConfig(**values)


def write_fixture_bin(path: Path, n_tokens: int = 512, vocab: int = 64) -> None:
    """Writes a raw uint16 token file readable by PackedDataset's legacy path."""
    rng = np.random.default_rng(0)
    rng.integers(0, vocab, size=n_tokens, dtype=np.uint16).tofile(path)


class LearningRateScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TrainConfig(
            warmup_steps=10, max_steps=100, learning_rate=1.0, min_lr=0.1
        )

    def test_warmup_ramps_linearly(self) -> None:
        self.assertAlmostEqual(learning_rate_at(0, self.config), 0.1, places=6)
        self.assertAlmostEqual(learning_rate_at(4, self.config), 0.5, places=6)

    def test_peak_at_end_of_warmup(self) -> None:
        self.assertAlmostEqual(learning_rate_at(10, self.config), 1.0, places=6)

    def test_decays_to_min_lr(self) -> None:
        self.assertAlmostEqual(learning_rate_at(100, self.config), 0.1, places=6)

    def test_monotonic_decay_after_warmup(self) -> None:
        values = [learning_rate_at(s, self.config) for s in range(10, 101, 10)]
        self.assertEqual(values, sorted(values, reverse=True))


class OptimizerTests(unittest.TestCase):
    def test_decay_applied_only_to_matrices(self) -> None:
        config = TrainConfig(weight_decay=0.1)
        model = Transformer(tiny_model_config())
        opt = build_optimizer(model, config, torch.device("cpu"))

        self.assertEqual(len(opt.param_groups), 2)
        decayed, plain = opt.param_groups
        self.assertEqual(decayed["weight_decay"], 0.1)
        self.assertEqual(plain["weight_decay"], 0.0)
        # RMSNorm gains are the only 1-D parameters: two per block plus the
        # final norm.
        self.assertEqual(len(plain["params"]), 2 * 3 + 1)
        self.assertTrue(all(p.dim() >= 2 for p in decayed["params"]))


class CheckpointTests(unittest.TestCase):
    def test_roundtrip_restores_weights(self) -> None:
        config = tiny_model_config()
        model = Transformer(config)
        opt = build_optimizer(model, TrainConfig(), torch.device("cpu"))

        tokens = torch.randint(0, config.vocab_size, (2, 8))
        model(tokens[:, :-1], targets=tokens[:, 1:]).loss.backward()
        opt.step()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_checkpoint(path, model, opt, 5, config, TrainConfig())
            blob = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(blob["step"], 5)
        restored = Transformer(blob["model_config"])
        restored.load_state_dict(blob["model"])
        for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
            self.assertTrue(torch.equal(a, b))


class ResumeCursorTests(unittest.TestCase):
    def test_block_order_is_a_pure_function_of_position(self) -> None:
        a = StatelessBlockSampler(50, 4, seed=7)
        b = StatelessBlockSampler(50, 4, seed=7)
        self.assertEqual([a.block_at(p) for p in range(20)],
                         [b.block_at(p) for p in range(20)])

    def test_resume_continues_the_same_stream(self) -> None:
        full = iter(StatelessBlockSampler(50, 4, seed=7, start_micro_batch=0))
        for _ in range(3 * 4):  # skip three micro-batches of four blocks each
            next(full)
        continued = [next(full) for _ in range(4)]

        resumed = iter(StatelessBlockSampler(50, 4, seed=7, start_micro_batch=3))
        self.assertEqual([next(resumed) for _ in range(4)], continued)


class EvaluateTests(unittest.TestCase):
    def test_returns_a_finite_mean_loss(self) -> None:
        config = TrainConfig(
            seq_len=8, batch_size=2, eval_steps=2, num_workers=0
        )
        model = Transformer(tiny_model_config()).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "val.bin"
            write_fixture_bin(path)
            loader = build_dataloader(
                path, config, shuffle=False, start_micro_batch=0
            )
            loss = evaluate(
                model, loader, config, torch.device("cpu"), nullcontext()
            )
        self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
