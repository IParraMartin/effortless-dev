"""Tests for the depth targets and the controller fit.

The load-bearing test here is :meth:`SyntheticRecovery.test_controller_learns_a
_known_depth_rule`. A controller that routes *everything* to full depth will
look respectable on any workload where depth rarely matters, so a metric alone
cannot distinguish "the controller learned" from "the controller learned
nothing and the task forgave it". The synthetic problem below has a depth rule
that is exactly recoverable from one feature, so failing to recover it is
unambiguous, and a constant policy is measurably worse than the fit.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from experiments.train_depth_controller import (
    ControllerTrainConfig,
    choose,
    continuation_gain,
    depth_confusion,
    calibration_error,
    earliest_sufficient_index,
    fit_controller,
    normalized_cost,
    per_tier_utility,
    policy_metrics,
    quality_matrix,
)

TIERS = [2, 4, 6]


def synthetic_problem(
    n: int = 600,
    feature_dim: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds a problem whose optimal depth is a known function of feature 0.

    Feature zero is a difficulty flag; the rest is noise the controller must
    learn to ignore. Easy requests are solved at every depth, hard ones only at
    the deepest, so with a price on compute the optimal policy is exactly
    "route on feature zero" and any deviation is measurable.

    Args:
        n: Requests.
        feature_dim: Feature width, including the flag.
        seed: Random seed.

    Returns:
        A tuple ``(features, quality, cost, optimal_index)``.
    """
    rng = np.random.default_rng(seed)
    hard = rng.integers(0, 2, size=n)

    features = rng.normal(0.0, 1.0, size=(n, feature_dim))
    features[:, 0] = hard * 3.0 - 1.5

    quality = np.empty((n, len(TIERS)))
    quality[hard == 0] = [0.9, 0.9, 0.9]
    quality[hard == 1] = [0.2, 0.5, 0.9]

    cost = np.tile(np.array(TIERS, dtype=float) / max(TIERS), (n, 1))
    optimal = np.where(hard == 1, len(TIERS) - 1, 0)
    return features, quality, cost, optimal


class Targets(unittest.TestCase):
    """The three target constructions behave as defined."""

    def test_earliest_sufficient_depth_against_full_depth(self) -> None:
        quality = np.array([[0.9, 0.9, 0.9], [0.2, 0.5, 0.9], [0.85, 0.9, 0.9]])
        index = earliest_sufficient_index(quality, epsilon=0.01)
        self.assertEqual(index.tolist(), [0, 2, 1])

    def test_tolerance_widens_the_sufficient_set(self) -> None:
        quality = np.array([[0.85, 0.9, 0.9]])
        self.assertEqual(
            earliest_sufficient_index(quality, epsilon=0.1).tolist(), [0]
        )
        self.assertEqual(
            earliest_sufficient_index(quality, epsilon=0.01).tolist(), [1]
        )

    def test_best_endpoint_reference_handles_harmful_depth(self) -> None:
        """A deeper endpoint can be worse, and full depth is then wrong."""
        quality = np.array([[0.9, 0.6, 0.5]])
        self.assertEqual(
            earliest_sufficient_index(
                quality, epsilon=0.01, reference="full_depth"
            ).tolist(),
            [0],
        )
        self.assertEqual(
            earliest_sufficient_index(
                quality, epsilon=0.01, reference="best_endpoint"
            ).tolist(),
            [0],
        )

    def test_unknown_reference_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            earliest_sufficient_index(np.zeros((1, 3)), 0.01, reference="middle")

    def test_utility_subtracts_priced_cost(self) -> None:
        quality = np.array([[0.5, 0.8]])
        cost = np.array([[0.5, 1.0]])
        np.testing.assert_allclose(
            per_tier_utility(quality, cost, 0.2), [[0.4, 0.6]]
        )

    def test_continuation_gain_is_zero_at_the_deepest_tier(self) -> None:
        quality = np.array([[0.2, 0.5, 0.9]])
        cost = np.array([[1 / 3, 2 / 3, 1.0]])
        gain = continuation_gain(quality, cost, routing_lambda=0.0)
        self.assertAlmostEqual(gain[0, 2], 0.0)
        self.assertAlmostEqual(gain[0, 0], 0.7)
        self.assertAlmostEqual(gain[0, 1], 0.4)

    def test_continuation_gain_turns_negative_when_depth_is_dear(self) -> None:
        quality = np.array([[0.85, 0.9]])
        cost = np.array([[0.5, 1.0]])
        gain = continuation_gain(quality, cost, routing_lambda=1.0)
        self.assertLess(gain[0, 0], 0.0)


class Columns(unittest.TestCase):
    """Reading quality and cost out of trajectory records."""

    def _records(self) -> list[dict]:
        return [
            {
                "teacher_forced_accuracy": [0.2, 0.6, 0.9],
                "teacher_forced_nll": [2.0, 1.0, 0.5],
                "cost_depth_fraction": [1 / 3, 2 / 3, 1.0],
                "cost_macs": [100.0, 200.0, 300.0],
                "free_running_reward": [],
            }
        ]

    def test_higher_is_better_metrics_pass_through(self) -> None:
        matrix = quality_matrix(self._records(), "teacher_forced_accuracy")
        np.testing.assert_allclose(matrix, [[0.2, 0.6, 0.9]])

    def test_lower_is_better_metrics_are_negated(self) -> None:
        matrix = quality_matrix(self._records(), "teacher_forced_nll")
        np.testing.assert_allclose(matrix, [[-2.0, -1.0, -0.5]])

    def test_costs_normalize_to_the_deepest_tier(self) -> None:
        for metric in ("cost_depth_fraction", "cost_macs"):
            np.testing.assert_allclose(
                normalized_cost(self._records(), metric),
                [[1 / 3, 2 / 3, 1.0]],
            )

    def test_missing_free_running_labels_raise_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            quality_matrix(self._records(), "free_running_reward")

    def test_unknown_metrics_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            quality_matrix(self._records(), "vibes")
        with self.assertRaises(ValueError):
            normalized_cost(self._records(), "vibes")


class Metrics(unittest.TestCase):
    """Policy scoring, regret, and the adaptivity decomposition."""

    def test_oracle_policy_has_zero_regret(self) -> None:
        _, quality, cost, optimal = synthetic_problem(n=200)
        metrics = policy_metrics(optimal, quality, cost, TIERS, 0.2)
        self.assertAlmostEqual(metrics["router_regret"], 0.0, places=9)
        self.assertGreater(metrics["adaptivity_gain"], 0.0)

    def test_constant_policy_regret_matches_the_adaptivity_gain(self) -> None:
        """The best fixed depth loses exactly the oracle's advantage over it."""
        _, quality, cost, _ = synthetic_problem(n=200)
        utility = per_tier_utility(quality, cost, 0.2)
        best = int(utility.mean(axis=0).argmax())

        metrics = policy_metrics(
            np.full(len(quality), best), quality, cost, TIERS, 0.2
        )
        self.assertAlmostEqual(
            metrics["router_regret"], metrics["adaptivity_gain"], places=9
        )

    def test_confusion_counts_oracle_against_chosen(self) -> None:
        matrix = depth_confusion(
            chosen=np.array([0, 2, 2]), oracle=np.array([0, 2, 0]), n_tiers=3
        )
        self.assertEqual(matrix[0][0], 1)
        self.assertEqual(matrix[2][2], 1)
        self.assertEqual(matrix[0][2], 1)

    def test_calibration_error_of_a_perfect_predictor_is_zero(self) -> None:
        probabilities = np.array([0.0, 0.0, 1.0, 1.0])
        outcomes = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(
            calibration_error(probabilities, outcomes), 0.0, places=9
        )

    def test_calibration_error_of_an_inverted_predictor_is_one(self) -> None:
        probabilities = np.array([1.0, 1.0])
        outcomes = np.array([0, 0])
        self.assertAlmostEqual(
            calibration_error(probabilities, outcomes), 1.0, places=9
        )


class SyntheticRecovery(unittest.TestCase):
    """The fit recovers a depth rule that is known to be recoverable."""

    def _config(self, **updates) -> ControllerTrainConfig:
        values = dict(target="utility", epochs=120, hidden=16,
                      learning_rate=3e-3, routing_lambda=0.3)
        values.update(updates)
        return ControllerTrainConfig(**values)

    def test_controller_learns_a_known_depth_rule(self) -> None:
        features, quality, cost, optimal = synthetic_problem(n=600, seed=0)
        train, test = slice(0, 400), slice(400, 600)

        controller, _ = fit_controller(
            features[train], quality[train], cost[train], TIERS,
            self._config(), seed=0,
        )
        chosen = choose(controller, features[test], cost[test], 0.3)

        agreement = float((chosen == optimal[test]).mean())
        self.assertGreater(agreement, 0.95)

    def test_learned_policy_beats_every_fixed_depth(self) -> None:
        features, quality, cost, _ = synthetic_problem(n=600, seed=1)
        train, test = slice(0, 400), slice(400, 600)

        controller, _ = fit_controller(
            features[train], quality[train], cost[train], TIERS,
            self._config(), seed=0,
        )
        chosen = choose(controller, features[test], cost[test], 0.3)
        metrics = policy_metrics(chosen, quality[test], cost[test], TIERS, 0.3)

        self.assertGreater(metrics["utility"], metrics["best_fixed_utility"])
        self.assertLess(metrics["router_regret"], 0.01)

    def test_the_task_is_not_won_by_a_constant_policy(self) -> None:
        """Guards the test above: a fixed depth must genuinely lose."""
        _, quality, cost, _ = synthetic_problem(n=600, seed=1)
        utility = per_tier_utility(quality, cost, 0.3)
        self.assertGreater(
            float(utility.max(axis=1).mean() - utility.mean(axis=0).max()), 0.05
        )

    def test_ordinal_head_also_recovers_the_rule(self) -> None:
        features, quality, cost, optimal = synthetic_problem(n=600, seed=2)
        train, test = slice(0, 400), slice(400, 600)

        controller, _ = fit_controller(
            features[train], quality[train], cost[train], TIERS,
            self._config(target="ordinal", epsilon=0.05), seed=0,
        )
        chosen = choose(controller, features[test], cost[test], 0.3)
        self.assertGreater(float((chosen == optimal[test]).mean()), 0.95)

    def test_lambda_changes_routing_on_a_frozen_controller(self) -> None:
        """One fit, many budgets: the point of the utility formulation."""
        features, quality, cost, _ = synthetic_problem(n=600, seed=3)
        controller, _ = fit_controller(
            features, quality, cost, TIERS, self._config(), seed=0
        )

        depths = [
            float(np.array(TIERS)[choose(controller, features, cost, lam)].mean())
            for lam in (0.0, 0.5, 5.0)
        ]
        self.assertGreaterEqual(depths[0], depths[1])
        self.assertGreaterEqual(depths[1], depths[2])
        self.assertLess(depths[2], depths[0])

    def test_unknown_target_is_rejected(self) -> None:
        features, quality, cost, _ = synthetic_problem(n=32)
        with self.assertRaises(ValueError):
            fit_controller(
                features, quality, cost, TIERS,
                self._config(target="oracle"), seed=0,
            )

    def test_controller_sees_no_labels(self) -> None:
        """Inference takes features only; nothing else is in scope."""
        features, quality, cost, _ = synthetic_problem(n=128)
        controller, _ = fit_controller(
            features, quality, cost, TIERS, self._config(epochs=5), seed=0
        )
        with torch.no_grad():
            scores = controller(torch.tensor(features, dtype=torch.float32))
        self.assertEqual(scores.shape, (128, len(TIERS)))


if __name__ == "__main__":
    unittest.main()
