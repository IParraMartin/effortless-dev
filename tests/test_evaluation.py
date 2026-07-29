"""Tests for the horizontal comparison and the substitution test.

These cover the code path the central claim runs through, and they exist
because it was the *least* exercised part of the repository: the statistics
helpers had tests against synthetic frontiers, but nothing ever loaded a
manifest, aligned the two sides, or ran the substitution test end to end. Two
defects were sitting there.

* The substitution ratio was computed as an estimand and then hardcoded to
  ``FAIL`` in the test, so the report contradicted a number in the same file.
* The sharing tax aligned the two sides **by column order**, so it silently
  produced nothing whenever the manifest had a different number of models than
  the backbone had exits — which is the normal case — and would have compared
  mismatched pairs if the counts had happened to agree.

Both were invisible without executing the path.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.evaluate_vertical_routing import (
    EvaluationConfig,
    best_static_mixture,
    cascade_system,
    complementarity_matrix,
    controller_report_rows,
    estimands,
    fixed_endpoints,
    horizontal_systems,
    load_manifest,
    oracle_system,
)
from utils.statistics import bootstrap_substitution_ratio


def write_manifest(directory: Path, entries: list[dict]) -> Path:
    """Writes a manifest and its per-model result files.

    Args:
        directory: Where to write.
        entries: Dicts with ``tier``, ``cost``, and ``quality`` (a list).

    Returns:
        Path to the manifest.
    """
    manifest = []
    for entry in entries:
        results = directory / f"m{entry['tier']}.json"
        results.write_text(json.dumps(entry["quality"]))
        manifest.append(
            {
                "model_id": f"indep{entry['tier']}",
                "tokenizer_id": "demo",
                "tier": entry["tier"],
                "cost": entry["cost"],
                "results": str(results),
            }
        )
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


class ManifestLoading(unittest.TestCase):
    """A malformed manifest fails loudly rather than comparing nonsense."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_valid_manifest_loads_sorted_by_tier(self) -> None:
        path = write_manifest(
            self.dir,
            [
                {"tier": 6, "cost": 1.0, "quality": [0.9] * 4},
                {"tier": 2, "cost": 0.3, "quality": [0.8] * 4},
            ],
        )
        entries = load_manifest(path)
        self.assertEqual([e["tier"] for e in entries], [2, 6])

    def test_missing_field_names_what_is_missing(self) -> None:
        path = self.dir / "bad.json"
        path.write_text(json.dumps([{"model_id": "a", "tier": 1}]))
        with self.assertRaisesRegex(ValueError, "cost"):
            load_manifest(path)

    def test_mixed_tokenizers_are_rejected(self) -> None:
        path = self.dir / "mixed.json"
        path.write_text(
            json.dumps(
                [
                    {"model_id": "a", "tokenizer_id": "x", "tier": 1,
                     "cost": 0.2, "results": "a.json"},
                    {"model_id": "b", "tokenizer_id": "y", "tier": 6,
                     "cost": 1.0, "results": "b.json"},
                ]
            )
        )
        with self.assertRaisesRegex(ValueError, "tokeniz"):
            load_manifest(path)

    def test_request_count_mismatch_is_rejected(self) -> None:
        """Paired comparison is meaningless if the rows are not the same rows."""
        path = write_manifest(
            self.dir, [{"tier": 6, "cost": 1.0, "quality": [0.9] * 3}]
        )
        with self.assertRaisesRegex(ValueError, "same requests"):
            horizontal_systems(load_manifest(path), n_requests=8, lambdas=(0.0,))


class StaticMixtureBaseline(unittest.TestCase):
    """The matched-cost baseline is an exact expectation, not one noisy draw."""

    def test_exact_expectation_hits_target_without_coin_flip_noise(self) -> None:
        quality = np.array([[0.0, 1.0], [1.0, 0.0]])
        cost = np.array([[0.2, 0.8], [0.2, 0.8]])

        first = best_static_mixture(quality, cost, target_cost=0.5, seed=0)
        second = best_static_mixture(quality, cost, target_cost=0.5, seed=999)

        self.assertIsNotNone(first)
        self.assertIsNone(first.choices)
        np.testing.assert_allclose(first.quality, [0.5, 0.5])
        np.testing.assert_allclose(first.cost, [0.5, 0.5])
        np.testing.assert_allclose(first.quality, second.quality)


class ControllerReportingSplit(unittest.TestCase):
    def test_uses_only_the_persisted_reporting_request_ids(self) -> None:
        records = [
            {"request_id": 10, "split": "validation"},
            {"request_id": 11, "split": "validation"},
            {"request_id": 12, "split": "validation"},
        ]
        blob = {"split_request_ids": {"report": [12, 10]}}
        self.assertEqual(controller_report_rows(records, blob), [2, 0])

    def test_old_checkpoint_is_rejected_instead_of_leaking_calibration(self) -> None:
        records = [{"request_id": 0, "split": "validation"}]
        with self.assertRaisesRegex(ValueError, "schema 2"):
            controller_report_rows(records, {})


class SharingTaxAlignment(unittest.TestCase):
    """The two sides are matched on the tier a model claims, not on order."""

    def _estimands(self, tiers, horizontal_tiers, horizontal_quality):
        n = 64
        rng = np.random.default_rng(0)
        quality = rng.uniform(0.5, 0.9, size=(n, len(tiers)))
        cost = np.tile(
            np.array(tiers, dtype=float) / max(tiers), (n, 1)
        )
        h_cost = np.tile(
            np.array(horizontal_tiers, dtype=float) / max(horizontal_tiers),
            (n, 1),
        )
        systems = fixed_endpoints(quality, cost, tiers)
        return estimands(
            systems, quality, cost, horizontal_quality, h_cost, tiers,
            EvaluationConfig(resamples=100), horizontal_tiers,
        )

    def test_fewer_models_than_tiers_still_produces_a_tax(self) -> None:
        """The case that previously returned None without saying so."""
        tiers = [1, 2, 3, 4, 5, 6]
        horizontal_tiers = [1, 3, 6]
        quality = np.full((64, 3), 0.8)

        result = self._estimands(tiers, horizontal_tiers, quality)
        taxes = result["sharing_tax"]
        self.assertIsNotNone(taxes)
        self.assertEqual([row["tier"] for row in taxes], [1, 3, 6])

    def test_tax_is_measured_against_the_matching_endpoint(self) -> None:
        tiers = [1, 2, 3]
        n = 64
        quality = np.zeros((n, 3))
        quality[:, 0], quality[:, 1], quality[:, 2] = 0.1, 0.5, 0.9
        cost = np.tile(np.array([1 / 3, 2 / 3, 1.0]), (n, 1))

        # One independent model, claiming tier 3, scoring a flat 1.0.
        horizontal = np.full((n, 1), 1.0)
        result = estimands(
            fixed_endpoints(quality, cost, tiers), quality, cost,
            horizontal, np.ones((n, 1)), tiers,
            EvaluationConfig(resamples=100), [3],
        )
        row = result["sharing_tax"][0]
        self.assertEqual(row["tier"], 3)
        # Against tier 3's endpoint (0.9), not tier 1's (0.1).
        self.assertAlmostEqual(row["endpoint_quality"], 0.9, places=6)
        self.assertAlmostEqual(row["sharing_tax"]["estimate"], 0.1, places=6)

    def test_unmatched_tiers_are_reported_not_dropped(self) -> None:
        tiers = [1, 2, 3]
        n = 64
        quality = np.full((n, 3), 0.5)
        cost = np.tile(np.array([1 / 3, 2 / 3, 1.0]), (n, 1))

        result = estimands(
            fixed_endpoints(quality, cost, tiers), quality, cost,
            np.full((n, 1), 0.6), np.ones((n, 1)), tiers,
            EvaluationConfig(resamples=100), [9],
        )
        self.assertIsNone(result["sharing_tax"])
        self.assertEqual(result["sharing_tax_unmatched_tiers"], [9])

    def test_absent_manifest_reports_absence(self) -> None:
        tiers = [1, 2]
        n = 32
        quality = np.full((n, 2), 0.5)
        cost = np.tile(np.array([0.5, 1.0]), (n, 1))
        result = estimands(
            fixed_endpoints(quality, cost, tiers), quality, cost,
            None, None, tiers, EvaluationConfig(resamples=100), None,
        )
        self.assertIsNone(result["horizontal_frontier"])
        self.assertIsNone(result["sharing_tax"])
        self.assertIn("unavailable, not zero", result["horizontal_note"])


class SubstitutionRatioInterval(unittest.TestCase):
    """The ratio is tested, and its interval rebuilds the frontier per replicate."""

    def _matrices(self, n: int, vertical_top: float, horizontal_top: float):
        quality = np.zeros((n, 2))
        quality[:, 0], quality[:, 1] = 0.5, vertical_top
        horizontal = np.zeros((n, 2))
        horizontal[:, 0], horizontal[:, 1] = 0.5, horizontal_top
        cost = np.tile(np.array([0.5, 1.0]), (n, 1))
        return quality, cost, horizontal, cost.copy()

    def test_identical_systems_give_a_ratio_of_one(self) -> None:
        v_q, v_c, h_q, h_c = self._matrices(256, 0.9, 0.9)
        interval = bootstrap_substitution_ratio(
            v_q, v_c, h_q, h_c, cost=1.0, baseline_quality=0.5,
            resamples=200, seed=0,
        )
        self.assertIsNotNone(interval)
        self.assertAlmostEqual(interval.estimate, 1.0, places=6)

    def test_half_the_gain_gives_half_the_ratio(self) -> None:
        v_q, v_c, h_q, h_c = self._matrices(256, 0.7, 0.9)
        interval = bootstrap_substitution_ratio(
            v_q, v_c, h_q, h_c, cost=1.0, baseline_quality=0.5,
            resamples=200, seed=0,
        )
        self.assertAlmostEqual(interval.estimate, 0.5, places=6)

    def test_interval_brackets_the_estimate(self) -> None:
        rng = np.random.default_rng(0)
        n = 256
        v_q = np.column_stack([rng.normal(0.5, 0.1, n), rng.normal(0.8, 0.1, n)])
        h_q = np.column_stack([rng.normal(0.5, 0.1, n), rng.normal(0.9, 0.1, n)])
        cost = np.tile(np.array([0.5, 1.0]), (n, 1))

        interval = bootstrap_substitution_ratio(
            v_q, cost, h_q, cost.copy(), cost=1.0, baseline_quality=0.5,
            resamples=300, seed=1,
        )
        self.assertLessEqual(interval.low, interval.estimate)
        self.assertGreaterEqual(interval.high, interval.estimate)

    def test_withheld_when_the_horizontal_system_gained_nothing(self) -> None:
        v_q, v_c, h_q, h_c = self._matrices(128, 0.9, 0.5001)
        self.assertIsNone(
            bootstrap_substitution_ratio(
                v_q, v_c, h_q, h_c, cost=1.0, baseline_quality=0.5,
                resamples=100, seed=0,
            )
        )

    def test_withheld_when_the_cost_is_outside_both_frontiers(self) -> None:
        v_q, v_c, h_q, h_c = self._matrices(128, 0.9, 0.9)
        self.assertIsNone(
            bootstrap_substitution_ratio(
                v_q, v_c, h_q, h_c, cost=5.0, baseline_quality=0.5,
                resamples=100, seed=0,
            )
        )


class Cascades(unittest.TestCase):
    """A reusable prefix must not be given away for free."""

    def test_vertical_cascade_pays_the_prefix_once(self) -> None:
        quality = np.array([[0.4, 0.9], [0.4, 0.9]])
        cost = np.array([[0.3, 1.0], [0.3, 1.0]])
        escalate = np.array([True, False])

        vertical = cascade_system(quality, cost, escalate, True, "v")
        horizontal = cascade_system(quality, cost, escalate, False, "h")

        # Escalated row: vertical continues (1.0), horizontal reruns (0.3+1.0).
        self.assertAlmostEqual(float(vertical.cost[0]), 1.0)
        self.assertAlmostEqual(float(horizontal.cost[0]), 1.3)
        # Unescalated row is the cheap path for both.
        self.assertAlmostEqual(float(vertical.cost[1]), 0.3)
        self.assertAlmostEqual(float(horizontal.cost[1]), 0.3)
        self.assertLess(vertical.mean_cost, horizontal.mean_cost)

    def test_quality_follows_the_escalation_decision(self) -> None:
        quality = np.array([[0.4, 0.9], [0.4, 0.9]])
        cost = np.array([[0.3, 1.0], [0.3, 1.0]])
        result = cascade_system(
            quality, cost, np.array([True, False]), True, "v"
        )
        self.assertAlmostEqual(float(result.quality[0]), 0.9)
        self.assertAlmostEqual(float(result.quality[1]), 0.4)


class Complementarity(unittest.TestCase):
    """Nested versus complementary errors, which bound what routing can buy."""

    def test_nested_errors_show_zeros_above_the_diagonal(self) -> None:
        # Every request the shallow candidate solves, the deep one solves too.
        quality = np.array([[0.9, 0.9], [0.1, 0.9], [0.1, 0.1]])
        matrix = complementarity_matrix(quality, [1, 2])["rescue_rate"]
        # Deep rescues shallow on one of three; shallow never rescues deep.
        self.assertAlmostEqual(matrix[0][1], 1 / 3)
        self.assertAlmostEqual(matrix[1][0], 0.0)

    def test_complementary_errors_are_detected(self) -> None:
        quality = np.array([[0.9, 0.1], [0.1, 0.9]])
        matrix = complementarity_matrix(quality, [1, 2])["rescue_rate"]
        self.assertAlmostEqual(matrix[0][1], 0.5)
        self.assertAlmostEqual(matrix[1][0], 0.5)


class OracleBound(unittest.TestCase):
    """The oracle bounds every policy over the same candidates."""

    def test_oracle_never_loses_to_a_fixed_candidate(self) -> None:
        rng = np.random.default_rng(0)
        quality = rng.uniform(0, 1, size=(200, 4))
        cost = np.tile(np.array([0.25, 0.5, 0.75, 1.0]), (200, 1))

        for lam in (0.0, 0.3, 1.0):
            oracle = oracle_system(quality, cost, lam)
            utility = (oracle.quality - lam * oracle.cost).mean()
            for column in range(4):
                fixed = (quality[:, column] - lam * cost[:, column]).mean()
                self.assertGreaterEqual(utility + 1e-9, fixed)


if __name__ == "__main__":
    unittest.main()


class ReachableCeiling(unittest.TestCase):
    """The conditional oracle separates learnable headroom from luck.

    The plain oracle maximizes over candidates per request, which requires the
    outcome. When per-request quality carries noise the router cannot predict,
    that maximization manufactures headroom nobody can reach — and a router
    judged against it is reported as failing when it is optimal.
    """

    def test_recovers_gain_that_is_predictable_from_features(self) -> None:
        from experiments.evaluate_vertical_routing import conditional_oracle

        rng = np.random.default_rng(0)
        n = 400
        # Feature 0 says which tier is better; nothing else matters.
        flag = rng.integers(0, 2, n)
        features = rng.normal(0, 1, (n, 6))
        features[:, 0] = flag * 4.0 - 2.0

        quality = np.where(flag[:, None] == 1, [0.2, 0.9], [0.9, 0.2]).astype(float)
        cost = np.tile(np.array([0.5, 1.0]), (n, 1))

        ceiling = conditional_oracle(features, quality, cost, 0.0, steps=250)
        best_fixed = quality.mean(0).max()
        # Predictable structure: the ceiling should capture most of it.
        self.assertGreater(ceiling.mean_quality, best_fixed + 0.2)

    def test_does_not_manufacture_gain_from_pure_noise(self) -> None:
        from experiments.evaluate_vertical_routing import (
            conditional_oracle, oracle_system,
        )

        rng = np.random.default_rng(1)
        n = 400
        features = rng.normal(0, 1, (n, 6))
        # Quality is independent of the features: nothing is learnable.
        quality = rng.uniform(0, 1, (n, 3))
        cost = np.tile(np.array([1 / 3, 2 / 3, 1.0]), (n, 1))

        plain = oracle_system(quality, cost, 0.0)
        ceiling = conditional_oracle(features, quality, cost, 0.0, steps=250)
        best_fixed = quality.mean(0).max()

        # The plain oracle invents a large gain out of the noise...
        self.assertGreater(plain.mean_quality - best_fixed, 0.15)
        # ...while the reachable ceiling stays near the best fixed candidate.
        self.assertLess(abs(ceiling.mean_quality - best_fixed), 0.08)
