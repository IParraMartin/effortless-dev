"""Tests for the anchored multi-exit objective.

The first two Savio arms reported a worse full endpoint under six-exit training.
That number is real, but it cannot be attributed to parameter sharing, because
the legacy objective normalizes across every exit: at depths 2/4/6/8/10/12 the
final endpoint receives ``12/42 = 0.2857`` of the hard-target coefficient and
the shallow exits receive the rest. A degraded endpoint is the expected outcome
of down-weighting it by a factor of three-and-a-half, not evidence about
capacity.

The anchored objective removes that confound by construction: the full-depth
coefficient is fixed and normalization happens over the shallow exits alone. The
tests below pin the properties that makes it usable as a causal instrument —
that ``shallow_loss_weight=0`` is *exactly* a final-only run, that the endpoint's
coefficient does not move when exits are added, that a capped step is an unbiased
stand-in for scoring all of them, and that the teacher never receives gradient.
"""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from src.config import TransformerConfig
from src.model import Transformer


def config(**updates) -> TransformerConfig:
    """Builds a small architecture with an anchored objective by default."""
    values = dict(
        vocab_size=64,
        d_model=32,
        n_layers=12,
        n_heads=4,
        n_kv_heads=2,
        ff_dim=64,
        max_seq_len=32,
        exit_every=2,
        min_exit_layer=1,
        objective_version="anchored_v1",
    )
    values.update(updates)
    return TransformerConfig(**values)


def batch(seed: int = 0, rows: int = 2, length: int = 16):
    """Returns a deterministic ``(input_ids, targets)`` pair."""
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 64, (rows, length), generator=generator)
    return ids, ids


def gradient_of(
    model: Transformer, term: str, ids: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Differentiates one named term of the objective against all parameters.

    Args:
        model: Model to differentiate.
        term: ``"full"``, ``"shallow"``, or ``"combined"``.
        ids: Input token ids.
        targets: Gold token ids.

    Returns:
        One flat vector, with zeros where a parameter received no gradient, so
        two calls are always comparable coordinate by coordinate.
    """
    model.eval()
    all_hidden = model._run_blocks(ids, 0, None)
    hidden = [all_hidden[layer] for layer in model.config.exit_layers]
    final = model._readout(len(model.exit_modules) - 1, hidden[-1])
    terms = model._objective_terms(ids, targets, hidden, final)

    scalar = {
        "full": terms.full,
        "shallow": terms.shallow,
        "combined": terms.combined(),
    }[term]
    parameters = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(scalar, parameters, allow_unused=True)
    return torch.cat(
        [
            (grad if grad is not None else torch.zeros_like(parameter)).reshape(-1)
            for grad, parameter in zip(grads, parameters)
        ]
    )


class TheEndpointIsAnchored(unittest.TestCase):
    """The property the legacy objective lacks."""

    def test_the_final_coefficient_is_one_at_every_exit_count(self) -> None:
        for exit_every in (1, 2, 3, 4, 6, 12):
            with self.subTest(exit_every=exit_every):
                anchored = config(exit_every=exit_every)
                model = Transformer(anchored)
                out = model(*batch())
                self.assertEqual(out.full_weight, 1.0)

    def test_the_legacy_coefficient_shrinks_as_exits_are_added(self) -> None:
        """Documents the confound, so it stays visible rather than folklore."""
        weights = {}
        for exit_every in (12, 2, 1):
            legacy = config(
                exit_every=exit_every, objective_version="legacy_normalized"
            )
            weights[exit_every] = legacy.exit_weights[-1]

        self.assertEqual(weights[12], 1.0)
        self.assertAlmostEqual(weights[2], 12 / 42, places=6)
        self.assertLess(weights[1], weights[2])

    def test_shallow_weights_normalize_over_shallow_exits_only(self) -> None:
        weights = config(exit_every=2).shallow_weights
        self.assertEqual(weights[-1], 0.0)
        self.assertAlmostEqual(sum(weights), 1.0, places=12)

    def test_a_single_exit_model_has_no_shallow_weights(self) -> None:
        self.assertEqual(config(exit_every=12).shallow_weights, (0.0,))


class ZeroAlphaIsFinalOnly(unittest.TestCase):
    """``shallow_loss_weight=0`` must be the control arm, not an approximation."""

    def setUp(self) -> None:
        self.ids, self.targets = batch(1)

    def test_the_objective_equals_the_full_cross_entropy(self) -> None:
        torch.manual_seed(0)
        model = Transformer(config(shallow_loss_weight=0.0))
        out = model(self.ids, targets=self.targets)

        self.assertAlmostEqual(float(out.loss), out.full_loss, places=6)

    def test_the_gradient_equals_a_final_only_model_from_the_same_parent(self) -> None:
        """The point of P0-2: same seed, same backbone, so this is comparable.

        Only the shared parameters can be compared by name. ``exit_modules.0``
        means the depth-2 exit in the six-exit model and the depth-12 exit in the
        final-only one, so the exit modules are compared by position instead.
        """
        torch.manual_seed(0)
        multi = Transformer(config(shallow_loss_weight=0.0, distill_weight=0.5))
        torch.manual_seed(0)
        single = Transformer(config(exit_every=12))

        backbone = {
            name
            for name, _ in single.named_parameters()
            if not name.startswith("exit_modules")
        }
        for name, parameter in single.named_parameters():
            if name in backbone:
                parameter.data.copy_(dict(multi.named_parameters())[name].data)

        left = self._named_gradient(multi, backbone)
        right = self._named_gradient(single, backbone)

        self.assertTrue(backbone, "no shared parameters to compare")
        for name in sorted(backbone):
            with self.subTest(name=name):
                self.assertTrue(
                    torch.allclose(left[name], right[name], atol=1e-6),
                    f"{name} differs by "
                    f"{float((left[name] - right[name]).abs().max()):.2e}",
                )

    def _named_gradient(self, model, names) -> dict[str, torch.Tensor]:
        """Gradient of the objective, keyed by parameter name."""
        model.eval()
        model.zero_grad(set_to_none=True)
        model(self.ids, targets=self.targets).loss.backward()
        return {
            name: (
                parameter.grad.detach().clone()
                if parameter.grad is not None
                else torch.zeros_like(parameter)
            )
            for name, parameter in model.named_parameters()
            if name in names
        }

    def test_no_shallow_readout_is_computed_at_all(self) -> None:
        """Zero weight has to be free, or it is not a usable control."""
        torch.manual_seed(0)
        model = Transformer(config(shallow_loss_weight=0.0))
        model.reset_head_counters()
        model(self.ids, targets=self.targets)

        self.assertEqual(model.head_calls, 1, "a shallow head ran at alpha=0")

    def test_an_alternating_schedule_skips_shallow_work_on_even_steps(self) -> None:
        torch.manual_seed(0)
        model = Transformer(
            config(shallow_loss_weight=1.0, shallow_loss_schedule="alternating")
        )
        model.train()

        model._step_counter.fill_(0)
        model.reset_head_counters()
        even = model(self.ids, targets=self.targets)
        even_calls = model.head_calls

        model._step_counter.fill_(1)
        model.reset_head_counters()
        odd = model(self.ids, targets=self.targets)

        self.assertEqual(even.shallow_alpha, 0.0)
        self.assertEqual(odd.shallow_alpha, 1.0)
        self.assertEqual(even_calls, 1)
        self.assertGreater(model.head_calls, even_calls)


class SampledShallowExits(unittest.TestCase):
    """A capped step must stand in for the exits it did not score."""

    def setUp(self) -> None:
        self.ids, self.targets = batch(2)

    def _averaged_shallow_gradient(self, model: Transformer, period: int):
        """Averages the shallow gradient over one complete rotation."""
        total = None
        for step in range(period):
            model._step_counter.fill_(step)
            gradient = gradient_of(model, "shallow", self.ids, self.targets)
            total = gradient if total is None else total + gradient
        return total / period

    def test_the_rotation_covers_every_shallow_exit_equally(self) -> None:
        """The premise the unbiased estimator rests on."""
        for exit_every, budget in ((2, 1), (2, 2), (2, 3), (1, 4)):
            with self.subTest(exit_every=exit_every, budget=budget):
                model = Transformer(
                    config(exit_every=exit_every, exits_per_step=budget)
                )
                shallow = len(model.config.exit_layers) - 1
                counts = {position: 0 for position in range(shallow)}
                for step in range(shallow):
                    model._step_counter.fill_(step)
                    for position in model._select_exits(shallow + 1):
                        if position != shallow:
                            counts[position] += 1

                self.assertEqual(
                    len(set(counts.values())),
                    1,
                    f"coverage is uneven: {counts}",
                )

    def test_the_unbiased_estimator_matches_scoring_every_exit(self) -> None:
        torch.manual_seed(0)
        full = Transformer(config(shallow_loss_weight=1.0, distill_weight=0.5))
        with full.score_all_exits():
            target = gradient_of(full, "shallow", self.ids, self.targets)

        torch.manual_seed(0)
        sampled = Transformer(
            config(
                shallow_loss_weight=1.0,
                distill_weight=0.5,
                exits_per_step=2,
                shallow_estimator="unbiased",
            )
        )
        period = len(sampled.config.exit_layers) - 1
        averaged = self._averaged_shallow_gradient(sampled, period)

        relative = float((averaged - target).norm() / target.norm())
        self.assertLess(relative, 1e-5, f"relative gradient error {relative:.2e}")

    def test_the_fixed_total_estimator_is_biased_and_says_so(self) -> None:
        """Kept selectable, but not silently equivalent to the unbiased one."""
        torch.manual_seed(0)
        full = Transformer(config(shallow_loss_weight=1.0))
        with full.score_all_exits():
            target = gradient_of(full, "shallow", self.ids, self.targets)

        torch.manual_seed(0)
        sampled = Transformer(
            config(
                shallow_loss_weight=1.0,
                exits_per_step=2,
                shallow_estimator="fixed_total",
            )
        )
        period = len(sampled.config.exit_layers) - 1
        averaged = self._averaged_shallow_gradient(sampled, period)

        relative = float((averaged - target).norm() / target.norm())
        self.assertGreater(
            relative,
            1e-3,
            "fixed_total is documented as biased; if it has become unbiased the "
            "documentation and the estimator choice both need revisiting",
        )

    def test_the_legacy_objective_keeps_its_fixed_total_behaviour(self) -> None:
        """Reproducibility of the runs on disk does not depend on this default."""
        model = Transformer(
            config(
                objective_version="legacy_normalized",
                exits_per_step=2,
                shallow_estimator="unbiased",
            )
        )
        weights = model.config.exit_weights
        last = len(model.config.exit_layers) - 1

        self.assertAlmostEqual(
            model._shallow_rescale([0, 1], weights, last, legacy=True),
            sum(weights[:last]) / (weights[0] + weights[1]),
            places=12,
        )


class Distillation(unittest.TestCase):
    """The teacher must be a target, never a second student."""

    def setUp(self) -> None:
        self.ids, self.targets = batch(3)

    def test_the_current_full_teacher_receives_no_gradient(self) -> None:
        """Otherwise the endpoint is dragged toward its own shallow exits.

        Asked of the graph rather than of parameters: with tied embeddings the
        student's own readout writes into the same matrix as the teacher's, so a
        parameter-level comparison cannot separate the two paths. The property is
        that the shallow term does not depend on the teacher's logits at all.
        """
        torch.manual_seed(0)
        model = Transformer(config(shallow_loss_weight=1.0, distill_weight=1.0))
        model.eval()

        all_hidden = model._run_blocks(self.ids, 0, None)
        hidden = [all_hidden[layer] for layer in model.config.exit_layers]
        final = model._readout(len(model.exit_modules) - 1, hidden[-1])
        terms = model._objective_terms(self.ids, self.targets, hidden, final)

        (through_teacher,) = torch.autograd.grad(
            terms.shallow, [final], retain_graph=True, allow_unused=True
        )
        self.assertIsNone(
            through_teacher,
            "the shallow term depends on the teacher's logits, so distillation "
            "would pull the endpoint toward its students",
        )
        # And the full term does reach them, so the check above is not vacuous.
        (through_full,) = torch.autograd.grad(
            terms.full, [final], allow_unused=True
        )
        self.assertIsNotNone(through_full)

    def test_a_frozen_parent_teacher_stays_frozen(self) -> None:
        torch.manual_seed(0)
        parent = Transformer(config())
        torch.manual_seed(0)
        child = Transformer(
            config(
                shallow_loss_weight=1.0,
                distill_weight=1.0,
                distill_teacher="frozen_parent",
            )
        )
        child.attach_parent(parent)

        before = {
            name: tensor.detach().clone() for name, tensor in parent.named_parameters()
        }
        child(self.ids, targets=self.targets).loss.backward()

        for name, tensor in parent.named_parameters():
            with self.subTest(name=name):
                self.assertIsNone(tensor.grad, f"{name} accumulated a gradient")
                self.assertTrue(torch.equal(tensor, before[name]))

    def test_a_missing_parent_is_an_error_not_a_skipped_term(self) -> None:
        model = Transformer(
            config(
                shallow_loss_weight=1.0,
                distill_weight=1.0,
                distill_teacher="frozen_parent",
            )
        )
        with self.assertRaisesRegex(ValueError, "frozen parent"):
            model(self.ids, targets=self.targets)

    def test_the_parent_is_not_part_of_this_model(self) -> None:
        """A registered teacher would enter the checkpoint and DDP's reductions."""
        torch.manual_seed(0)
        parent = Transformer(config())
        torch.manual_seed(0)
        child = Transformer(config())
        before = set(child.state_dict())
        child.attach_parent(parent)

        self.assertEqual(set(child.state_dict()), before)
        self.assertEqual(len(list(child.parameters())), len(list(child.parameters())))
        self.assertIs(child.parent, parent)

    def test_temperature_scales_the_term_by_its_square(self) -> None:
        """The convention that keeps the gradient scale temperature-invariant."""
        torch.manual_seed(0)
        model = Transformer(config())
        student = torch.randn(2, 5, 64)
        teacher = torch.randn(2, 5, 64)
        valid = torch.ones(2, 5, dtype=torch.bool)

        for temperature in (1.0, 2.0, 4.0):
            with self.subTest(temperature=temperature):
                measured = float(
                    model._distillation(student, teacher, valid, temperature)
                )
                expected = float(
                    F.kl_div(
                        F.log_softmax(student / temperature, dim=-1),
                        F.log_softmax(teacher / temperature, dim=-1),
                        log_target=True,
                        reduction="none",
                    )
                    .sum(dim=-1)
                    .mean()
                    * temperature**2
                )
                self.assertAlmostEqual(measured, expected, places=5)

    def test_identical_distributions_give_zero_divergence(self) -> None:
        model = Transformer(config())
        logits = torch.randn(2, 5, 64)
        valid = torch.ones(2, 5, dtype=torch.bool)

        self.assertAlmostEqual(
            float(model._distillation(logits, logits, valid, 2.0)), 0.0, places=6
        )

    def test_top_k_pooling_keeps_both_distributions_normalized(self) -> None:
        """Truncating without pooling would let the student escape into the tail."""
        model = Transformer(config())
        teacher = torch.randn(1, 3, 64).log_softmax(dim=-1)
        student = torch.randn(1, 3, 64).log_softmax(dim=-1)
        pooled_teacher, pooled_student = model._pool_tail(teacher, student, 8)

        self.assertEqual(pooled_teacher.shape[-1], 9)
        for name, pooled in (("teacher", pooled_teacher), ("student", pooled_student)):
            with self.subTest(name=name):
                mass = pooled.exp().sum(dim=-1)
                self.assertTrue(torch.allclose(mass, torch.ones_like(mass), atol=1e-4))

    def test_top_k_penalizes_a_student_that_flees_to_the_tail(self) -> None:
        model = Transformer(config())
        valid = torch.ones(1, 1, dtype=torch.bool)
        teacher = torch.full((1, 1, 32), -10.0)
        teacher[..., :4] = 5.0  # all the mass on four tokens

        agreeing = teacher.clone()
        fleeing = torch.full((1, 1, 32), -10.0)
        fleeing[..., 20:] = 5.0  # all the mass outside the teacher's top-4

        close = float(model._distillation(agreeing, teacher, valid, 1.0, top_k=4))
        far = float(model._distillation(fleeing, teacher, valid, 1.0, top_k=4))
        self.assertLess(close, 1e-4)
        self.assertGreater(far, 1.0)

    def test_a_top_k_covering_the_vocabulary_is_a_no_op(self) -> None:
        model = Transformer(config())
        student, teacher = torch.randn(1, 2, 64), torch.randn(1, 2, 64)
        valid = torch.ones(1, 2, dtype=torch.bool)

        self.assertAlmostEqual(
            float(model._distillation(student, teacher, valid, 2.0, top_k=64)),
            float(model._distillation(student, teacher, valid, 2.0)),
            places=5,
        )


class Preservation(unittest.TestCase):
    """The no-regret constraint, expressed in the objective."""

    def setUp(self) -> None:
        self.ids, self.targets = batch(4)

    def test_an_identical_parent_gives_zero_preservation_loss(self) -> None:
        torch.manual_seed(0)
        parent = Transformer(config())
        torch.manual_seed(0)
        child = Transformer(config(preservation_weight=1.0))
        child.attach_parent(parent)
        out = child(self.ids, targets=self.targets)

        self.assertLess(out.preservation_loss, 1e-6)

    def test_a_diverged_parent_gives_a_positive_preservation_loss(self) -> None:
        torch.manual_seed(0)
        parent = Transformer(config())
        torch.manual_seed(1)
        child = Transformer(config(preservation_weight=1.0))
        child.attach_parent(parent)
        out = child(self.ids, targets=self.targets)

        self.assertGreater(out.preservation_loss, 1e-4)

    def test_preservation_enters_the_optimized_scalar(self) -> None:
        torch.manual_seed(0)
        parent = Transformer(config())
        torch.manual_seed(1)
        with_term = Transformer(config(preservation_weight=2.0))
        with_term.attach_parent(parent)
        torch.manual_seed(1)
        without = Transformer(config(preservation_weight=0.0))

        left = with_term(self.ids, targets=self.targets)
        right = without(self.ids, targets=self.targets)
        self.assertAlmostEqual(
            float(left.loss.detach()) - float(right.loss.detach()),
            2.0 * left.preservation_loss,
            places=5,
        )

    def test_preservation_without_a_parent_is_an_error(self) -> None:
        model = Transformer(config(preservation_weight=1.0))
        with self.assertRaisesRegex(ValueError, "frozen parent"):
            model(self.ids, targets=self.targets)

    def test_the_legacy_objective_ignores_preservation(self) -> None:
        """Legacy runs must reproduce, including when new fields are set."""
        torch.manual_seed(0)
        model = Transformer(
            config(objective_version="legacy_normalized", preservation_weight=5.0)
        )
        out = model(self.ids, targets=self.targets)

        self.assertIsNone(out.preservation_loss)


class Schedules(unittest.TestCase):
    """How the shallow coefficient moves, and that it is reported."""

    def test_constant_ignores_the_step(self) -> None:
        anchored = config(shallow_loss_weight=0.3)
        self.assertEqual(anchored.shallow_alpha(0), 0.3)
        self.assertEqual(anchored.shallow_alpha(10_000), 0.3)

    def test_linear_warmup_rises_to_the_target(self) -> None:
        anchored = config(
            shallow_loss_weight=0.4,
            shallow_loss_schedule="linear_warmup",
            shallow_loss_warmup_steps=100,
        )
        self.assertEqual(anchored.shallow_alpha(0), 0.0)
        self.assertAlmostEqual(anchored.shallow_alpha(50), 0.2)
        self.assertAlmostEqual(anchored.shallow_alpha(100), 0.4)
        self.assertAlmostEqual(anchored.shallow_alpha(500), 0.4)

    def test_cosine_ramp_starts_slower_than_linear(self) -> None:
        common = dict(shallow_loss_weight=1.0, shallow_loss_warmup_steps=100)
        cosine = config(shallow_loss_schedule="cosine_ramp", **common)
        linear = config(shallow_loss_schedule="linear_warmup", **common)

        self.assertLess(cosine.shallow_alpha(10), linear.shallow_alpha(10))
        self.assertAlmostEqual(cosine.shallow_alpha(50), 0.5, places=6)
        self.assertAlmostEqual(cosine.shallow_alpha(100), 1.0, places=6)

    def test_alternating_gives_full_only_steps(self) -> None:
        anchored = config(
            shallow_loss_weight=0.5, shallow_loss_schedule="alternating"
        )
        self.assertEqual(
            [anchored.shallow_alpha(step) for step in range(4)], [0.0, 0.5, 0.0, 0.5]
        )

    def test_a_zero_target_stays_zero_under_every_schedule(self) -> None:
        for schedule in ("constant", "alternating"):
            with self.subTest(schedule=schedule):
                anchored = config(
                    shallow_loss_weight=0.0, shallow_loss_schedule=schedule
                )
                self.assertEqual(anchored.shallow_alpha(1), 0.0)


class Logging(unittest.TestCase):
    """Every term the objective contains must be separately readable."""

    def test_the_forward_reports_each_component(self) -> None:
        torch.manual_seed(0)
        parent = Transformer(config())
        torch.manual_seed(1)
        model = Transformer(
            config(
                shallow_loss_weight=0.25,
                distill_weight=0.5,
                preservation_weight=0.1,
            )
        )
        model.attach_parent(parent)
        out = model(*batch(5))

        self.assertIsNotNone(out.full_loss)
        self.assertEqual(out.full_weight, 1.0)
        self.assertEqual(out.shallow_alpha, 0.25)
        self.assertIsNotNone(out.preservation_loss)
        self.assertEqual(len(out.selected_exits), len(model.config.exit_layers))
        self.assertEqual(
            set(out.distill_losses), set(model.config.exit_layers[:-1])
        )
        self.assertIn(model.config.exit_layers[-1], out.exit_losses)

    def test_the_combined_objective_is_the_sum_of_its_parts(self) -> None:
        torch.manual_seed(0)
        model = Transformer(config(shallow_loss_weight=0.25, distill_weight=0.5))
        ids, targets = batch(6)
        model.eval()

        all_hidden = model._run_blocks(ids, 0, None)
        hidden = [all_hidden[layer] for layer in model.config.exit_layers]
        final = model._readout(len(model.exit_modules) - 1, hidden[-1])
        terms = model._objective_terms(ids, targets, hidden, final)

        self.assertAlmostEqual(
            float(terms.combined().detach()),
            terms.full_weight * float(terms.full.detach())
            + terms.alpha * float(terms.shallow.detach()),
            places=5,
        )

    def test_the_reported_full_ce_is_the_endpoint_alone(self) -> None:
        torch.manual_seed(0)
        multi = Transformer(config(shallow_loss_weight=1.0, distill_weight=1.0))
        ids, targets = batch(7)
        out = multi(ids, targets=targets)

        with torch.no_grad():
            logits = multi(ids).logits
            direct = float(
                F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
            )
        self.assertAlmostEqual(out.full_loss, direct, places=5)
        self.assertNotAlmostEqual(out.full_loss, float(out.loss.detach()), places=3)


class Conflict(unittest.TestCase):
    """The diagnostic that has to precede any gradient-surgery machinery."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = Transformer(config(shallow_loss_weight=1.0, distill_weight=0.5))
        self.ids, self.targets = batch(8)

    def test_it_reports_the_statistics_it_claims_to(self) -> None:
        result = self.model.gradient_diagnostics(self.ids, self.targets)

        self.assertGreater(result.full_norm, 0.0)
        self.assertGreater(result.shallow_norm, 0.0)
        self.assertAlmostEqual(
            result.norm_ratio, result.shallow_norm / result.full_norm, places=6
        )
        self.assertTrue(-1.0001 <= result.cosine <= 1.0001)
        self.assertTrue(0.0 <= result.negative_fraction <= 1.0)
        self.assertIn("block-00", result.layer_cosine)
        self.assertIn("embedding", result.layer_cosine)

    def test_a_term_identical_to_itself_has_cosine_one(self) -> None:
        """Calibrates the measure, so a plausible-looking number can be trusted."""
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        named = list(
            (name, p) for name, p in self.model.named_parameters() if p.requires_grad
        )
        self.model.eval()
        all_hidden = self.model._run_blocks(self.ids, 0, None)
        hidden = [all_hidden[layer] for layer in self.model.config.exit_layers]
        final = self.model._readout(len(self.model.exit_modules) - 1, hidden[-1])
        terms = self.model._objective_terms(self.ids, self.targets, hidden, final)
        grads = torch.autograd.grad(
            terms.full, parameters, retain_graph=True, allow_unused=True
        )

        result = self.model._summarize_gradients(named, grads, grads, groups=False)
        self.assertAlmostEqual(result.cosine, 1.0, places=5)
        self.assertAlmostEqual(result.norm_ratio, 1.0, places=5)

    def test_a_negated_term_has_cosine_minus_one(self) -> None:
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        named = list(
            (name, p) for name, p in self.model.named_parameters() if p.requires_grad
        )
        self.model.eval()
        all_hidden = self.model._run_blocks(self.ids, 0, None)
        hidden = [all_hidden[layer] for layer in self.model.config.exit_layers]
        final = self.model._readout(len(self.model.exit_modules) - 1, hidden[-1])
        terms = self.model._objective_terms(self.ids, self.targets, hidden, final)
        grads = torch.autograd.grad(
            terms.full, parameters, retain_graph=True, allow_unused=True
        )
        flipped = tuple(None if g is None else -g for g in grads)

        result = self.model._summarize_gradients(named, grads, flipped, groups=True)
        self.assertAlmostEqual(result.cosine, -1.0, places=5)
        self.assertAlmostEqual(result.negative_fraction, 1.0, places=6)

    def test_it_leaves_no_accumulated_gradient_behind(self) -> None:
        """It runs beside training, so it must not disturb the update."""
        self.model.zero_grad(set_to_none=True)
        self.model.gradient_diagnostics(self.ids, self.targets)

        for name, parameter in self.model.named_parameters():
            with self.subTest(name=name):
                self.assertIsNone(parameter.grad)

    def test_it_restores_the_training_flag(self) -> None:
        self.model.train()
        self.model.gradient_diagnostics(self.ids, self.targets)
        self.assertTrue(self.model.training)

        self.model.eval()
        self.model.gradient_diagnostics(self.ids, self.targets)
        self.assertFalse(self.model.training)

    def test_it_scores_every_exit_regardless_of_the_cap(self) -> None:
        """A rotating subset would make the answer depend on the step it ran on."""
        torch.manual_seed(0)
        capped = Transformer(
            config(shallow_loss_weight=1.0, exits_per_step=1)
        )
        capped._step_counter.fill_(0)
        first = capped.gradient_diagnostics(self.ids, self.targets)
        capped._step_counter.fill_(3)
        second = capped.gradient_diagnostics(self.ids, self.targets)

        self.assertAlmostEqual(first.shallow_norm, second.shallow_norm, places=5)

    def test_a_single_exit_model_is_rejected(self) -> None:
        model = Transformer(config(exit_every=12))
        with self.assertRaisesRegex(ValueError, "shallow exit"):
            model.gradient_diagnostics(self.ids, self.targets)

    def test_the_report_renders(self) -> None:
        text = self.model.gradient_diagnostics(self.ids, self.targets).report()
        self.assertIn("global cosine", text)
        self.assertIn("block-00", text)


class InvalidConfigurations(unittest.TestCase):
    """Misconfiguration must fail loudly rather than train something else."""

    def test_an_unknown_objective_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "objective_version"):
            config(objective_version="anchored_v2")

    def test_custom_weights_must_cover_exactly_the_shallow_depths(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly the shallow depths"):
            config(
                shallow_weighting="custom",
                shallow_custom_weights={2: 1.0, 4: 1.0},
            )

    def test_custom_weights_may_not_name_a_depth_without_an_exit(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly the shallow depths"):
            config(
                shallow_weighting="custom",
                shallow_custom_weights={
                    2: 1.0, 3: 1.0, 4: 1.0, 6: 1.0, 8: 1.0, 10: 1.0
                },
            )

    def test_custom_weights_must_be_requested_to_be_used(self) -> None:
        with self.assertRaisesRegex(ValueError, "silently ignored"):
            config(shallow_custom_weights={2: 1.0})

    def test_custom_weighting_needs_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires shallow_custom_weights"):
            config(shallow_weighting="custom")

    def test_negative_custom_weights_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            config(
                shallow_weighting="custom",
                shallow_custom_weights={2: 1.0, 4: -1.0, 6: 1.0, 8: 1.0, 10: 1.0},
            )

    def test_all_zero_custom_weights_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            config(
                shallow_weighting="custom",
                shallow_custom_weights=dict.fromkeys((2, 4, 6, 8, 10), 0.0),
            )

    def test_string_keyed_custom_weights_are_accepted(self) -> None:
        """They arrive as strings from JSON and from the command line."""
        anchored = config(
            shallow_weighting="custom",
            shallow_custom_weights={"2": 1, "4": 1, "6": 1, "8": 1, "10": 1},
        )
        self.assertEqual(
            [round(w, 6) for w in anchored.shallow_weights],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.0],
        )

    def test_a_ramp_without_a_span_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shallow_loss_warmup_steps"):
            config(
                shallow_loss_weight=0.5,
                shallow_loss_schedule="cosine_ramp",
                shallow_loss_warmup_steps=0,
            )

    def test_negative_weights_are_rejected(self) -> None:
        for name in (
            "full_loss_weight",
            "shallow_loss_weight",
            "distill_weight",
            "preservation_weight",
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    config(**{name: -0.1})

    def test_a_non_positive_temperature_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "distill_temperature"):
            config(distill_temperature=0.0)

    def test_a_non_positive_top_k_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "distill_top_k"):
            config(distill_top_k=0)

    def test_an_unknown_teacher_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "distill_teacher"):
            config(distill_teacher="external_gpt")

    def test_an_unknown_estimator_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shallow_estimator"):
            config(shallow_estimator="importance")


if __name__ == "__main__":
    unittest.main()
