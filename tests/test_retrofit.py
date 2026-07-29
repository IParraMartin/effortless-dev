"""Tests for no-regret retrofit.

The claim a retrofit makes is that the parent's full-depth output does not
change. Under a frozen mode that claim is exact and therefore checkable, and
these tests check it the way it can actually fail: **after** optimizer steps, not
at initialization. Every zero-initialized module is bit-identical when
constructed, so an initialization-time check passes even when a parameter that
feeds the full-depth path has been left trainable by mistake.

The subtler failure has the same shape. ``Module.apply`` visits every submodule,
so the model-wide initialization traversal overwrites an adapter's deliberate
zeros with the standard normal. The adapter's own constructor is correct; the
model built from it is not. That is why the identity test here runs on a fully
constructed model rather than on a bare adapter.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from src.config import TransformerConfig
from src.model import Transformer
from src.modules import ExitAdapter
from src.retrofit import (
    EXACT_MODES,
    RETROFIT_MODES,
    LoRALinear,
    RetrofitConfig,
    apply_lora,
    assert_parent_preserved,
    load_parent,
    retrofit,
    set_lora_enabled,
    trainable_parameters,
)


def config(**updates) -> TransformerConfig:
    """Builds a small architecture."""
    values = dict(
        vocab_size=64,
        d_model=32,
        n_layers=6,
        n_heads=4,
        n_kv_heads=2,
        ff_dim=64,
        max_seq_len=32,
        exit_every=6,
        min_exit_layer=1,
    )
    values.update(updates)
    return TransformerConfig(**values)


def parent_model(seed: int = 0) -> Transformer:
    """Builds a final-only parent, frozen and in eval mode."""
    torch.manual_seed(seed)
    model = Transformer(config())
    model.eval().requires_grad_(False)
    return model


def six_exit_config(parent: Transformer, retrofit_config: RetrofitConfig):
    """Derives the retrofitted architecture with exits every two layers."""
    return replace(
        parent.config,
        exit_every=2,
        exit_adapter_rank=(
            retrofit_config.exit_adapter_rank
            if retrofit_config.mode == "frozen_exit_adapter"
            else 0
        ),
        tie_embeddings=not retrofit_config.untie_exit_heads,
    )


def train_briefly(model: Transformer, steps: int = 3, seed: int = 1) -> None:
    """Takes real optimizer steps on whatever the retrofit left trainable.

    Args:
        model: The retrofitted model.
        steps: Number of updates.
        seed: Seed for the synthetic batch.
    """
    parameters = [parameter for _, parameter in trainable_parameters(model)]
    if not parameters:
        return
    optimizer = torch.optim.AdamW(parameters, lr=0.05)
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 64, (2, 12), generator=generator)

    model.train()
    for _ in range(steps):
        with model.score_all_exits():
            model(ids, targets=ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    model.eval()


class FrozenModesPreserveTheParentExactly(unittest.TestCase):
    """The property that makes a frozen retrofit a no-regret lower bound."""

    def setUp(self) -> None:
        self.parent = parent_model()
        generator = torch.Generator().manual_seed(9)
        self.probe = torch.randint(0, 64, (2, 12), generator=generator)

    def _build(self, mode: str) -> tuple[Transformer, object]:
        settings = {"frozen_exit_adapter": {"exit_adapter_rank": 8}}.get(mode, {})
        retrofit_config = RetrofitConfig(mode=mode, **settings)
        return retrofit(
            self.parent,
            retrofit_config,
            model_config=six_exit_config(self.parent, retrofit_config),
        )

    def test_the_parent_is_bit_identical_at_construction(self) -> None:
        for mode in EXACT_MODES:
            with self.subTest(mode=mode):
                model, _ = self._build(mode)
                self.assertEqual(
                    assert_parent_preserved(model, self.parent, self.probe), 0.0
                )

    def test_the_parent_is_bit_identical_after_optimizer_steps(self) -> None:
        """The check that matters. Everything is identical before training."""
        for mode in EXACT_MODES:
            with self.subTest(mode=mode):
                model, _ = self._build(mode)
                train_briefly(model)
                self.assertEqual(
                    assert_parent_preserved(model, self.parent, self.probe),
                    0.0,
                    f"{mode} moved the parent's full-depth logits",
                )

    def test_the_shallow_exits_actually_moved(self) -> None:
        """Otherwise the test above passes because nothing trained at all."""
        for mode in EXACT_MODES:
            with self.subTest(mode=mode):
                model, _ = self._build(mode)
                before = {
                    name: tensor.detach().clone()
                    for name, tensor in trainable_parameters(model)
                }
                train_briefly(model)
                moved = [
                    name
                    for name, tensor in model.named_parameters()
                    if name in before and not torch.equal(tensor, before[name])
                ]
                self.assertTrue(moved, f"{mode} trained nothing")

    def test_no_backbone_parameter_is_trainable(self) -> None:
        for mode in EXACT_MODES:
            with self.subTest(mode=mode):
                _, report = self._build(mode)
                offending = [
                    name
                    for name in report.trainable_names
                    if name.startswith("blocks") or name.startswith("embed")
                ]
                self.assertEqual(offending, [])

    def test_the_final_exit_module_is_not_trainable(self) -> None:
        """It is the parent's own head; training it is what breaks exactness."""
        for mode in EXACT_MODES:
            with self.subTest(mode=mode):
                model, report = self._build(mode)
                last = len(model.exit_modules) - 1
                offending = [
                    name
                    for name in report.trainable_names
                    if name.startswith(f"exit_modules.{last}.")
                ]
                self.assertEqual(offending, [])

    def test_a_shallow_exit_module_is_trainable(self) -> None:
        for mode in EXACT_MODES:
            with self.subTest(mode=mode):
                _, report = self._build(mode)
                self.assertTrue(
                    any(
                        name.startswith("exit_modules.0.")
                        for name in report.trainable_names
                    )
                )

    def test_the_report_says_the_parent_is_exact(self) -> None:
        for mode in EXACT_MODES:
            with self.subTest(mode=mode):
                _, report = self._build(mode)
                self.assertTrue(report.exact)
                self.assertIn("exactly preserved", report.summary())

    def test_a_moved_parent_raises_rather_than_returning_a_number(self) -> None:
        """A caller that ignored the number would publish an unverified claim."""
        model, _ = self._build("frozen_tied_head")
        with torch.no_grad():
            model.blocks[0].attn.q_proj.weight.add_(0.1)

        with self.assertRaisesRegex(AssertionError, "full-depth logits moved"):
            assert_parent_preserved(model, self.parent, self.probe)


class AdaptersAreIdentityAfterTheWholeModelIsBuilt(unittest.TestCase):
    """``Module.apply`` visits every submodule, including zero-initialized ones."""

    def test_a_bare_adapter_is_identity(self) -> None:
        adapter = ExitAdapter(32, 8)
        x = torch.randn(2, 5, 32)
        self.assertTrue(torch.equal(adapter(x), x))

    def test_an_adapter_inside_a_constructed_model_is_still_identity(self) -> None:
        """The regression: the model-wide traversal used to overwrite the zeros."""
        torch.manual_seed(0)
        model = Transformer(config(exit_every=2, exit_adapter_rank=8))
        x = torch.randn(2, 5, 32)

        for position, exit_module in enumerate(model.exit_modules):
            with self.subTest(position=position):
                self.assertIsNotNone(exit_module.adapter)
                self.assertTrue(
                    torch.equal(exit_module.adapter(x), x),
                    "the initialization traversal overwrote the adapter's zeros",
                )

    def test_adding_adapters_does_not_change_any_output(self) -> None:
        torch.manual_seed(0)
        plain = Transformer(config(exit_every=2))
        torch.manual_seed(0)
        adapted = Transformer(config(exit_every=2, exit_adapter_rank=8))
        adapted.load_state_dict(plain.state_dict(), strict=False)

        generator = torch.Generator().manual_seed(3)
        ids = torch.randint(0, 64, (2, 12), generator=generator)
        plain.eval()
        adapted.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(plain(ids).logits, adapted(ids).logits))

    def test_the_down_projection_is_not_degenerate(self) -> None:
        """Zero on both sides would leave the adapter unable to move at all."""
        adapter = ExitAdapter(32, 8)
        self.assertGreater(float(adapter.down.weight.detach().abs().max()), 0.0)
        self.assertEqual(float(adapter.up.weight.detach().abs().max()), 0.0)

    def test_a_non_positive_rank_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank"):
            ExitAdapter(32, 0)

    def test_a_negative_configured_rank_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exit_adapter_rank"):
            config(exit_adapter_rank=-1)


class LoRA(unittest.TestCase):
    """Low-rank updates, and the exact reference they keep available."""

    def setUp(self) -> None:
        self.parent = parent_model()
        generator = torch.Generator().manual_seed(11)
        self.probe = torch.randint(0, 64, (2, 12), generator=generator)

    def _build(self, **settings) -> tuple[Transformer, object]:
        retrofit_config = RetrofitConfig(mode="lora", **settings)
        return retrofit(
            self.parent,
            retrofit_config,
            model_config=six_exit_config(self.parent, retrofit_config),
        )

    def test_a_wrapped_layer_is_the_original_at_construction(self) -> None:
        base = torch.nn.Linear(16, 8, bias=False)
        wrapped = LoRALinear(base, rank=4)
        x = torch.randn(3, 16)

        self.assertTrue(torch.equal(wrapped(x), base(x)))

    def test_the_base_weight_is_frozen(self) -> None:
        base = torch.nn.Linear(16, 8, bias=False)
        wrapped = LoRALinear(base, rank=4)

        self.assertFalse(wrapped.base.weight.requires_grad)
        self.assertTrue(wrapped.lora_a.weight.requires_grad)
        self.assertTrue(wrapped.lora_b.weight.requires_grad)

    def test_target_selection_is_exact(self) -> None:
        model, report = self._build(lora_targets=("q_proj",))
        self.assertEqual(len(report.lora_modules), model.config.n_layers)
        for name in report.lora_modules:
            with self.subTest(name=name):
                self.assertTrue(name.endswith("q_proj"))

    def test_layer_selection_is_exact(self) -> None:
        _, report = self._build(lora_targets=("q_proj", "v_proj"), lora_layers=(0, 2))
        self.assertEqual(
            sorted(report.lora_modules),
            ["blocks.0.attn.q_proj", "blocks.0.attn.v_proj",
             "blocks.2.attn.q_proj", "blocks.2.attn.v_proj"],
        )

    def test_only_the_selected_modules_require_gradients(self) -> None:
        _, report = self._build(lora_targets=("q_proj",), lora_layers=(1,))
        lora_names = [name for name in report.trainable_names if "lora_" in name]

        self.assertTrue(lora_names)
        for name in lora_names:
            with self.subTest(name=name):
                self.assertTrue(name.startswith("blocks.1.attn.q_proj."))

    def test_the_trainable_count_matches_the_configuration(self) -> None:
        rank = 4
        _, report = self._build(
            lora_rank=rank, lora_targets=("q_proj",), lora_layers=(0,)
        )
        d_model = self.parent.config.d_model
        # One down-projection and one up-projection per wrapped layer.
        self.assertEqual(report.trainable_groups["lora"], 2 * rank * d_model)

    def test_disabling_the_updates_recovers_the_parent_exactly(self) -> None:
        model, _ = self._build(lora_rank=4)
        train_briefly(model)

        # With updates on, the parent has moved.
        with self.assertRaises(AssertionError):
            assert_parent_preserved(model, self.parent, self.probe)

        switched = set_lora_enabled(model, False)
        self.assertGreater(switched, 0)
        self.assertEqual(
            assert_parent_preserved(model, self.parent, self.probe), 0.0
        )

    def test_re_enabling_restores_the_adapted_model(self) -> None:
        model, _ = self._build(lora_rank=4)
        train_briefly(model)
        model.eval()
        with torch.no_grad():
            adapted = model(self.probe).logits.clone()

        set_lora_enabled(model, False)
        set_lora_enabled(model, True)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(self.probe).logits, adapted))

    def test_the_scaling_convention_is_alpha_over_rank(self) -> None:
        base = torch.nn.Linear(16, 8, bias=False)
        wrapped = LoRALinear(base, rank=4, alpha=16.0)
        self.assertAlmostEqual(wrapped.scale, 4.0)

    def test_a_non_positive_rank_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank"):
            LoRALinear(torch.nn.Linear(4, 4), rank=0)

    def test_applying_twice_does_not_nest(self) -> None:
        """A double apply would wrap a wrapper and silently double the scale."""
        torch.manual_seed(0)
        model = Transformer(config(exit_every=2))
        first = apply_lora(model, RetrofitConfig(mode="lora", lora_targets=("q_proj",)))
        second = apply_lora(
            model, RetrofitConfig(mode="lora", lora_targets=("q_proj",))
        )

        self.assertEqual(len(first), model.config.n_layers)
        self.assertEqual(
            second, (), "an already-wrapped projection was wrapped again"
        )


class SelectiveUnfreeze(unittest.TestCase):
    """Trainable backbone, and the honesty that has to accompany it."""

    def setUp(self) -> None:
        self.parent = parent_model()

    def _build(self, **settings) -> tuple[Transformer, object]:
        retrofit_config = RetrofitConfig(mode="selective_unfreeze", **settings)
        return retrofit(
            self.parent,
            retrofit_config,
            model_config=six_exit_config(self.parent, retrofit_config),
        )

    def test_only_the_named_blocks_become_trainable(self) -> None:
        _, report = self._build(unfreeze_blocks=(1, 4))
        blocks = {
            int(name.split(".")[1])
            for name in report.trainable_names
            if name.startswith("blocks.")
        }
        self.assertEqual(blocks, {1, 4})

    def test_norms_only_restricts_the_capacity_increase(self) -> None:
        _, wide = self._build(unfreeze_blocks=(1,))
        _, narrow = self._build(unfreeze_blocks=(1,), unfreeze_norms_only=True)

        self.assertLess(narrow.trainable, wide.trainable)
        for name in narrow.trainable_names:
            if name.startswith("blocks."):
                with self.subTest(name=name):
                    self.assertIn("norm", name)

    def test_the_report_does_not_claim_exactness(self) -> None:
        _, report = self._build(unfreeze_blocks=(0,))
        self.assertFalse(report.exact)
        self.assertIn("may drift", report.summary())

    def test_an_unconstrained_parent_is_flagged(self) -> None:
        _, report = self._build(unfreeze_blocks=(0,))
        self.assertTrue(
            any("nothing constrains the parent" in note for note in report.notes)
        )

    def test_a_preservation_weight_removes_the_flag(self) -> None:
        _, report = self._build(unfreeze_blocks=(0,), preservation_weight=0.5)
        self.assertFalse(
            any("nothing constrains the parent" in note for note in report.notes)
        )

    def test_an_out_of_range_block_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "names block"):
            self._build(unfreeze_blocks=(0, 99))

    def test_naming_no_blocks_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unfreeze_blocks"):
            RetrofitConfig(mode="selective_unfreeze")


class WeightTransfer(unittest.TestCase):
    """Exit modules match by depth, not by index."""

    def test_the_parents_head_lands_on_the_matching_depth(self) -> None:
        """Matching by index would install the parent's readout on depth 2."""
        parent = parent_model()
        retrofit_config = RetrofitConfig(mode="frozen_tied_head")
        model, _ = retrofit(
            parent,
            retrofit_config,
            model_config=six_exit_config(parent, retrofit_config),
        )

        self.assertEqual(parent.config.exit_layers, (5,))
        self.assertEqual(model.config.exit_layers, (1, 3, 5))
        last = len(model.exit_modules) - 1
        self.assertTrue(
            torch.equal(
                model.exit_modules[last].norm.weight,
                parent.exit_modules[0].norm.weight,
            )
        )

    def test_new_exits_are_reported_as_uninitialized_from_the_parent(self) -> None:
        parent = parent_model()
        retrofit_config = RetrofitConfig(mode="frozen_tied_head")
        _, report = retrofit(
            parent,
            retrofit_config,
            model_config=six_exit_config(parent, retrofit_config),
        )
        self.assertTrue(
            any("no parent counterpart" in note for note in report.notes)
        )

    def test_the_backbone_is_copied_exactly(self) -> None:
        parent = parent_model()
        retrofit_config = RetrofitConfig(mode="frozen_tied_head")
        model, _ = retrofit(
            parent,
            retrofit_config,
            model_config=six_exit_config(parent, retrofit_config),
        )

        source = parent.state_dict()
        for name, tensor in model.state_dict().items():
            if name.startswith("blocks") or name.startswith("embed"):
                with self.subTest(name=name):
                    self.assertTrue(torch.equal(tensor, source[name]))

    def test_a_mismatched_backbone_is_rejected(self) -> None:
        parent = parent_model()
        retrofit_config = RetrofitConfig(mode="frozen_tied_head")
        with self.assertRaisesRegex(ValueError, "must be the parent's"):
            retrofit(
                parent,
                retrofit_config,
                model_config=replace(parent.config, d_model=64, n_heads=4, ff_dim=64),
            )


class ParentLoading(unittest.TestCase):
    """A checkpoint becomes a frozen reference, or says why it cannot."""

    def test_a_saved_checkpoint_round_trips(self) -> None:
        from src.config import TrainConfig
        from training.train import build_optimizer, save_checkpoint

        model = parent_model()
        train_config = TrainConfig(dtype="fp32", compile_model=False)
        optimizer = build_optimizer(model, train_config, torch.device("cpu"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.pt"
            save_checkpoint(path, model, optimizer, 10, model.config, train_config)
            loaded, loaded_config = load_parent(path)

        self.assertEqual(loaded_config.n_layers, model.config.n_layers)
        self.assertFalse(loaded.training)
        for parameter in loaded.parameters():
            self.assertFalse(parameter.requires_grad)

        generator = torch.Generator().manual_seed(4)
        ids = torch.randint(0, 64, (1, 8), generator=generator)
        with torch.no_grad():
            self.assertTrue(torch.equal(loaded(ids).logits, model(ids).logits))

    def test_a_missing_checkpoint_is_reported(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_parent("nowhere/final.pt")

    def test_a_checkpoint_without_a_config_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "foreign.pt"
            torch.save({"model": {}}, path)
            with self.assertRaisesRegex(KeyError, "model_config"):
                load_parent(path)


class ModeValidation(unittest.TestCase):
    """A mode that degenerates into another mode makes the record wrong."""

    def test_every_mode_is_constructible(self) -> None:
        required = {
            "selective_unfreeze": {"unfreeze_blocks": (0,)},
        }
        for mode in RETROFIT_MODES:
            with self.subTest(mode=mode):
                RetrofitConfig(mode=mode, **required.get(mode, {}))

    def test_an_unknown_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            RetrofitConfig(mode="magic")

    def test_an_adapter_mode_with_zero_rank_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exit_adapter_rank"):
            RetrofitConfig(mode="frozen_exit_adapter", exit_adapter_rank=0)

    def test_an_untied_mode_sets_the_flag_it_implies(self) -> None:
        self.assertTrue(RetrofitConfig(mode="frozen_untied_head").untie_exit_heads)

    def test_lora_needs_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "lora_targets"):
            RetrofitConfig(mode="lora", lora_targets=())

    def test_an_invalid_dropout_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "lora_dropout"):
            RetrofitConfig(mode="lora", lora_dropout=1.0)

    def test_qlora_says_it_is_not_quantized_here(self) -> None:
        """Never let a BF16 control be reported as QLoRA."""
        parent = parent_model()
        retrofit_config = RetrofitConfig(mode="qlora", lora_rank=4)
        _, report = retrofit(
            parent,
            retrofit_config,
            model_config=six_exit_config(parent, retrofit_config),
        )
        self.assertTrue(
            any("must not be reported as QLoRA" in note for note in report.notes)
        )

    def test_recoverability_is_distinguished_from_exactness(self) -> None:
        self.assertTrue(RetrofitConfig(mode="lora").parent_is_recoverable)
        self.assertFalse(RetrofitConfig(mode="lora").preserves_parent_exactly)
        self.assertTrue(
            RetrofitConfig(mode="frozen_tied_head").preserves_parent_exactly
        )
        self.assertFalse(
            RetrofitConfig(
                mode="selective_unfreeze", unfreeze_blocks=(0,)
            ).parent_is_recoverable
        )


class Audit(unittest.TestCase):
    """The report is the audit, so a wrong intention is visible before training."""

    def test_tied_weights_are_counted_once(self) -> None:
        parent = parent_model()
        retrofit_config = RetrofitConfig(mode="full_finetune")
        model, report = retrofit(
            parent,
            retrofit_config,
            model_config=six_exit_config(parent, retrofit_config),
        )
        self.assertEqual(report.trainable, model.num_parameters())
        self.assertEqual(report.frozen, 0)

    def test_an_untied_head_dominates_an_adapter(self) -> None:
        """The cost asymmetry that decides which retrofit is affordable."""
        parent = parent_model()
        reports = {}
        for mode, settings in (
            ("frozen_untied_head", {}),
            ("frozen_exit_adapter", {"exit_adapter_rank": 8}),
        ):
            retrofit_config = RetrofitConfig(mode=mode, **settings)
            _, reports[mode] = retrofit(
                parent,
                retrofit_config,
                model_config=six_exit_config(parent, retrofit_config),
            )

        self.assertGreater(
            reports["frozen_untied_head"].trainable,
            reports["frozen_exit_adapter"].trainable,
        )

    def test_groups_separate_the_costs(self) -> None:
        parent = parent_model()
        retrofit_config = RetrofitConfig(
            mode="frozen_exit_adapter", exit_adapter_rank=8
        )
        _, report = retrofit(
            parent,
            retrofit_config,
            model_config=six_exit_config(parent, retrofit_config),
        )
        self.assertEqual(
            set(report.trainable_groups), {"exit_norms", "exit_adapters"}
        )

    def test_trainable_parameters_deduplicates(self) -> None:
        torch.manual_seed(0)
        model = Transformer(config(exit_every=2))
        model.requires_grad_(True)
        names = [name for name, _ in trainable_parameters(model)]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            sum(parameter.numel() for _, parameter in trainable_parameters(model)),
            model.num_parameters(),
        )

    def test_the_summary_renders(self) -> None:
        parent = parent_model()
        retrofit_config = RetrofitConfig(mode="lora", lora_rank=4)
        _, report = retrofit(
            parent,
            retrofit_config,
            model_config=six_exit_config(parent, retrofit_config),
        )
        text = report.summary()
        self.assertIn("mode        lora", text)
        self.assertIn("layers wrapped", text)


if __name__ == "__main__":
    unittest.main()
