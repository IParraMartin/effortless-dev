"""Tests for auditing a depth-capped cache against the memory claim."""

from __future__ import annotations

import unittest

import torch

from src.config import TransformerConfig
from src.model import Transformer
from src.modules import KVCache
from utils.costs import AnalyticalCostModel, audit_kv_cache


def config(**updates) -> TransformerConfig:
    values = dict(
        vocab_size=64,
        d_model=32,
        n_layers=8,
        n_heads=4,
        n_kv_heads=2,
        ff_dim=64,
        max_seq_len=32,
        exit_every=2,
        min_exit_layer=1,
    )
    values.update(updates)
    return TransformerConfig(**values)


class Audit(unittest.TestCase):
    """The measurement, the analytical model and the law must agree."""

    def setUp(self) -> None:
        self.config = config()
        self.model = Transformer(self.config)
        self.costs = AnalyticalCostModel.from_config(self.config)

    def cache_at(self, depth: int, seq_len: int = 12) -> KVCache:
        """Runs a real forward pass to depth and returns its cache."""
        cache = KVCache(self.config.n_layers, max_depth=depth)
        ids = torch.randint(0, self.config.vocab_size, (1, seq_len))
        self.model.forward_to_depth(ids, stop_depth=depth, cache=cache)
        return cache

    def test_a_real_capped_cache_is_exact(self) -> None:
        for depth in (2, 4, 6, 8):
            with self.subTest(depth=depth):
                audit = audit_kv_cache(
                    self.cache_at(depth), self.costs, self.config.n_layers
                )
                self.assertTrue(audit.exact, audit.report())
                self.assertEqual(audit.leaked_layers, ())

    def test_saving_follows_the_proportional_law(self) -> None:
        """The formal content of the claim: memory falls exactly 1 - d/L."""
        for depth in (2, 4, 6):
            with self.subTest(depth=depth):
                audit = audit_kv_cache(
                    self.cache_at(depth), self.costs, self.config.n_layers
                )
                self.assertAlmostEqual(
                    audit.measured_saving, 1.0 - depth / 8, places=12
                )

    def test_full_depth_saves_nothing(self) -> None:
        audit = audit_kv_cache(self.cache_at(8), self.costs, self.config.n_layers)
        self.assertEqual(audit.measured_saving, 0.0)
        self.assertEqual(audit.measured_bytes, audit.full_depth_bytes)

    def test_an_upper_layer_entry_is_caught(self) -> None:
        """The failure the audit exists for, which no quality metric shows.

        A cache that materializes above its cap still produces correct tokens.
        Only the memory claim is false, and only this notices.
        """
        cache = self.cache_at(4)
        # Reach past the cap the way a buggy propagation path would.
        rows, heads = 1, self.config.n_kv_heads
        head_dim = self.config.d_model // self.config.n_heads
        cache.keys[5] = torch.zeros(rows, heads, cache.seq_len, head_dim)
        cache.values[5] = torch.zeros(rows, heads, cache.seq_len, head_dim)

        audit = audit_kv_cache(cache, self.costs, self.config.n_layers)
        self.assertFalse(audit.exact)
        self.assertEqual(audit.leaked_layers, (5,))
        self.assertIn("LEAK", audit.report())
        self.assertIn("layers above the cap", audit.report())

    def test_report_names_a_model_disagreement(self) -> None:
        """Measured and analytical bytes differing is its own failure."""
        cache = self.cache_at(4)
        wrong = AnalyticalCostModel.from_config(config(n_kv_heads=4))
        audit = audit_kv_cache(cache, wrong, self.config.n_layers)
        self.assertFalse(audit.exact)
        self.assertEqual(audit.leaked_layers, ())
        self.assertIn("disagrees with the analytical model", audit.report())

    def test_dtype_scales_both_sides_together(self) -> None:
        cache = self.cache_at(4)
        fp32 = audit_kv_cache(cache, self.costs, 8, dtype="fp32")
        bf16 = audit_kv_cache(cache, self.costs, 8, dtype="bf16")
        self.assertEqual(fp32.predicted_bytes, 2 * bf16.predicted_bytes)
        # The saving is a ratio, so precision cancels out of it entirely.
        self.assertAlmostEqual(fp32.predicted_saving, bf16.predicted_saving)


if __name__ == "__main__":
    unittest.main()
