from __future__ import annotations

import unittest

import torch

from src.config import TransformerConfig
from src.model import Transformer


def config(**updates) -> TransformerConfig:
    values = dict(
        vocab_size=64,
        d_model=32,
        n_layers=6,
        n_heads=4,
        n_kv_heads=2,
        ff_dim=64,
        max_seq_len=16,
        min_exit_layer=1,
        learned_kv_propagation=True,
    )
    values.update(updates)
    return TransformerConfig(**values)


class RegressionTests(unittest.TestCase):
    def test_kv_adapters_start_as_exact_identity(self) -> None:
        model = Transformer(config())
        for block in model.blocks:
            adapter = block.kv_adapter
            self.assertEqual(float(adapter.up.weight.detach().abs().max()), 0.0)
            self.assertEqual(
                float(adapter.gap_embed.weight.detach().abs().max()), 0.0
            )

    def test_zero_propagation_weight_skips_auxiliary_loss(self) -> None:
        model = Transformer(config(kv_propagation_weight=0.0))
        ids = torch.randint(0, model.config.vocab_size, (2, 8))
        output = model(ids[:, :-1], targets=ids[:, 1:])
        self.assertIsNone(output.kv_loss)

    def test_final_only_with_exit_sampling_does_not_divide_by_zero(self) -> None:
        model = Transformer(
            config(exit_loss_weighting="final_only", exits_per_step=2)
        )
        ids = torch.randint(0, model.config.vocab_size, (2, 8))
        output = model(ids[:, :-1], targets=ids[:, 1:])
        self.assertTrue(torch.isfinite(output.loss))

    def test_propagation_sampling_uses_only_configured_exits(self) -> None:
        model = Transformer(config(exit_every=2))
        sampled = model._sample_exit_layers(torch.Size([128, 128]), torch.device("cpu"))
        self.assertTrue(set(sampled.unique().tolist()).issubset(model.config.exit_layers))

    def test_all_final_simulation_matches_full_pass_with_dropout(self) -> None:
        torch.manual_seed(11)
        model = Transformer(config(dropout=0.5, learned_kv_propagation=False))
        model.train()
        ids = torch.randint(0, model.config.vocab_size, (2, 8))
        exits = torch.full_like(ids, model.config.n_layers - 1)
        torch.manual_seed(99)
        expected = model._run_blocks(ids)[-1]
        torch.manual_seed(99)
        actual = model.simulate_early_exit(ids, exits)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
