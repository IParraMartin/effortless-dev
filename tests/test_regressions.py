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


class ExitRotationAliasing(unittest.TestCase):
    """The rotation is deterministic in the step, so it aliases.

    ``_select_exits`` chooses from ``step % n_non_final``. Any schedule keyed on
    the step and divisible by that period therefore samples the same position
    forever. In the 38,140-step ``vr-exits`` run, ``eval_every=500`` against
    five non-final exits meant depths 6, 8 and 10 were never once scored on
    held-out data.
    """

    def model(self) -> Transformer:
        # Six exits: five non-final, so the rotation has period five.
        return Transformer(config(n_layers=6, exit_every=1, exits_per_step=2))

    def test_rotation_aliases_against_a_step_multiple(self) -> None:
        model = self.model()
        selections = []
        for step in (500, 1000, 1500, 2000):
            model._step_counter.fill_(step)
            selections.append(tuple(model._select_exits(6)))

        self.assertEqual(len(set(selections)), 1, "expected the bug to reproduce")
        self.assertNotIn(2, selections[0])
        self.assertNotIn(3, selections[0])
        self.assertNotIn(4, selections[0])

    def test_consecutive_steps_still_cover_every_exit(self) -> None:
        """The rotation is not broken -- only its interaction with a schedule."""
        model = self.model()
        seen: set[int] = set()
        for step in range(5):
            model._step_counter.fill_(step)
            seen.update(model._select_exits(6))
        self.assertEqual(seen, set(range(6)))

    def test_score_all_exits_suspends_the_rotation(self) -> None:
        model = self.model()
        model._step_counter.fill_(500)
        with model.score_all_exits():
            self.assertEqual(model._select_exits(6), list(range(6)))
        self.assertNotEqual(model._select_exits(6), list(range(6)))

    def test_score_all_exits_restores_after_a_raise(self) -> None:
        model = self.model()
        with self.assertRaises(RuntimeError):
            with model.score_all_exits():
                raise RuntimeError("boom")
        self.assertFalse(model._score_every_exit)

    def test_every_exit_reports_a_loss_under_the_context(self) -> None:
        model = self.model()
        model._step_counter.fill_(500)
        ids = torch.randint(0, model.config.vocab_size, (2, 8))

        sampled = model(ids[:, :-1], targets=ids[:, 1:])
        with model.score_all_exits():
            complete = model(ids[:, :-1], targets=ids[:, 1:])

        # Two non-final exits plus the final one, against all six.
        self.assertEqual(len(sampled.exit_losses), 3)
        self.assertEqual(
            sorted(complete.exit_losses), list(model.config.exit_layers)
        )

    def test_the_context_does_not_enter_the_checkpoint(self) -> None:
        model = self.model()
        with model.score_all_exits():
            self.assertNotIn("_score_every_exit", model.state_dict())
