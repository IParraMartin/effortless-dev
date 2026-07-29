"""Tests for depth-capped execution and request-level vertical routing.

The tests are grouped by what could plausibly break, and several exist because
a plausible implementation would pass a weaker check while being wrong:

* Depth is an *executed block count*, never a layer index, and the two differ
  by one everywhere. :class:`DepthSemantics` pins that down.
* A depth-capped cache must not merely avoid *reading* upper layers, it must
  not allocate them, since that is the entire memory claim.
* Routing must give the same tokens whether a request is batched with others or
  run alone. Any test where every row picks the same depth exercises only the
  easy path, so the mixed cases here script the depths rather than hoping an
  untrained controller produces variety.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.config import RoutingConfig, TransformerConfig
from src.model import Transformer
from src.modules import KVCache
from src.routing import DepthController, pool_prompt_features
from utils.costs import AnalyticalCostModel, CostCounters
from utils.statistics import (
    common_cost_range,
    integrated_substitution_ratio,
    interpolate_frontier,
    non_inferiority_test,
    paired_bootstrap,
    pareto_frontier,
    vertical_substitution_ratio,
)


def tiny_config(**updates) -> TransformerConfig:
    """Builds a small model configuration with grouped-query attention on.

    GQA is enabled by default rather than in one dedicated test, so every path
    below exercises the ``enable_gqa`` branch instead of the multi-head one.

    Args:
        **updates: Fields to override.

    Returns:
        The configuration.
    """
    values = dict(
        vocab_size=64,
        d_model=32,
        n_layers=6,
        n_heads=4,
        n_kv_heads=2,
        ff_dim=64,
        max_seq_len=48,
        min_exit_layer=1,
        exit_every=2,
    )
    values.update(updates)
    return TransformerConfig(**values)


def tiny_model(seed: int = 0, **updates) -> Transformer:
    """Builds a deterministic small model in eval mode.

    Args:
        seed: Seed applied before construction.
        **updates: Configuration overrides.

    Returns:
        The model.
    """
    torch.manual_seed(seed)
    return Transformer(tiny_config(**updates)).eval()


class ScriptedController(torch.nn.Module):
    """A controller that returns a fixed plan, one depth per request.

    Exists so mixed-depth batching can be tested without depending on what an
    untrained network happens to prefer. A uniformly confident controller sends
    every row to the same depth and exercises only the path where bucketing is
    a no-op, which is precisely the path that cannot fail.

    Args:
        plan: Depths to hand out, consumed in request order.
        n_tiers: Number of tiers, for shaping the returned scores.
    """

    feature_dim = 0
    hidden_dim = 0
    output = "utility"

    def __init__(self, plan: list[int], n_tiers: int = 3) -> None:
        super().__init__()
        self.plan = list(plan)
        self.n_tiers = n_tiers

    def select(self, features, tiers, **_):
        """Pops one planned depth per row.

        Args:
            features: Pooled features, read only for its row count.
            tiers: Candidate depths, read only for the score width.

        Returns:
            A tuple ``(depths, scores)``.
        """
        rows = features.size(0)
        depths = torch.tensor([self.plan.pop(0) for _ in range(rows)])
        return depths, torch.zeros(rows, len(tiers))


class DepthSemantics(unittest.TestCase):
    """Depth counts executed blocks, and conversions happen in one place."""

    def test_depth_and_layer_conversions_are_inverse(self) -> None:
        model = tiny_model()
        for layer in range(model.config.n_layers):
            depth = model.depth_of_layer(layer)
            self.assertEqual(model.layer_of_depth(depth), layer)
            self.assertEqual(depth, layer + 1)

    def test_exit_depths_end_at_full_depth(self) -> None:
        config = tiny_config()
        self.assertEqual(config.exit_depths[-1], config.n_layers)
        self.assertEqual(
            config.exit_depths, tuple(l + 1 for l in config.exit_layers)
        )

    def test_depth_out_of_range_is_rejected(self) -> None:
        model = tiny_model()
        ids = torch.randint(0, 64, (2, 4))
        with self.assertRaises(ValueError):
            model.forward_to_depth(ids, model.config.n_layers + 1)


class PrefixExecution(unittest.TestCase):
    """Requirement 3: forward_to_depth matches running the first d blocks."""

    def test_matches_reference_execution_at_every_depth(self) -> None:
        model = tiny_model()
        ids = torch.randint(0, 64, (3, 7))
        reference = model._run_blocks(ids)

        for depth in range(1, model.config.n_layers + 1):
            state = model.forward_to_depth(ids, depth)
            torch.testing.assert_close(state.hidden, reference[depth - 1])
            self.assertEqual(state.depth, depth)

    def test_incremental_decode_matches_one_shot(self) -> None:
        """The same prefix, fed one token at a time, must land in one place."""
        model = tiny_model()
        ids = torch.randint(0, 64, (2, 6))
        depth = 4

        one_shot = model.forward_to_depth(ids, depth)

        cache = KVCache(model.config.n_layers, max_depth=depth)
        state = None
        for position in range(ids.size(1)):
            state = model.forward_to_depth(
                ids[:, position : position + 1], depth, cache=cache
            )
        torch.testing.assert_close(
            state.hidden[:, -1], one_shot.hidden[:, -1], atol=1e-5, rtol=1e-5
        )


class SuffixContinuation(unittest.TestCase):
    """Requirement 4: continue_from_depth composes with the prefix."""

    def test_composition_reproduces_direct_execution(self) -> None:
        model = tiny_model()
        ids = torch.randint(0, 64, (3, 5))
        full = model.forward_to_depth(ids, model.config.n_layers)

        for split in (1, 2, 4):
            prefix = model.forward_to_depth(
                ids, split, return_boundary_state=True
            )
            suffix = model.continue_from_depth(
                prefix.boundary, split, model.config.n_layers
            )
            torch.testing.assert_close(suffix.hidden, full.hidden)
            self.assertEqual(suffix.backfill_tokens, ids.size(1))
            self.assertEqual(
                suffix.backfill_blocks, model.config.n_layers - split
            )

    def test_partial_boundary_state_is_rejected(self) -> None:
        """Escalating from the last token alone would silently lose attention."""
        model = tiny_model()
        ids = torch.randint(0, 64, (2, 6))
        cache = KVCache(model.config.n_layers)
        prefix = model.forward_to_depth(
            ids, 2, cache=cache, return_boundary_state=True
        )

        with self.assertRaises(ValueError) as raised:
            model.continue_from_depth(
                prefix.boundary[:, -1:], 2, 4, cache=cache, offset=0
            )
        self.assertIn("boundary activation", str(raised.exception))


class DepthCappedCache(unittest.TestCase):
    """Requirement 6: the cap is enforced, not merely respected."""

    def test_no_entries_above_the_cap(self) -> None:
        model = tiny_model()
        ids = torch.randint(0, 64, (2, 5))
        cache = KVCache(model.config.n_layers, max_depth=3)
        model.forward_to_depth(ids, 3, cache=cache)

        self.assertEqual(
            cache.layer_presence, (True, True, True, False, False, False)
        )
        self.assertEqual(cache.active_depth, 3)
        self.assertEqual(cache.seq_len, 5)

    def test_writing_above_the_cap_raises(self) -> None:
        model = tiny_model()
        cache = KVCache(model.config.n_layers, max_depth=2)
        with self.assertRaises(ValueError):
            model.forward_to_depth(torch.randint(0, 64, (1, 3)), 4, cache=cache)

    def test_bytes_match_the_analytical_formula(self) -> None:
        model = tiny_model()
        cost = AnalyticalCostModel.from_config(model.config)
        ids = torch.randint(0, 64, (2, 5))

        for depth in (1, 3, 6):
            cache = KVCache(model.config.n_layers, max_depth=depth)
            model.forward_to_depth(ids, depth, cache=cache)
            expected = 2 * cache.seq_len * depth * cost.kv_width * 4 * 2
            self.assertEqual(cache.bytes_allocated, expected)

    def test_bytes_are_proportional_to_depth(self) -> None:
        model = tiny_model()
        ids = torch.randint(0, 64, (2, 5))

        per_depth = set()
        for depth in (1, 2, 3, 6):
            cache = KVCache(model.config.n_layers, max_depth=depth)
            model.forward_to_depth(ids, depth, cache=cache)
            per_depth.add(cache.bytes_allocated // depth)
        self.assertEqual(len(per_depth), 1)

    def test_full_cache_still_holds_every_layer(self) -> None:
        """The token-level propagation path must keep working unchanged."""
        model = tiny_model()
        cache = KVCache(model.config.n_layers)
        model.generate(torch.randint(0, 64, (2, 3)), max_new_tokens=3)
        model.forward_to_depth(torch.randint(0, 64, (2, 3)), 6, cache=cache)
        self.assertTrue(all(cache.layer_presence))

    def test_select_rows_produces_independent_caches(self) -> None:
        model = tiny_model()
        ids = torch.randint(0, 64, (4, 5))
        cache = KVCache(model.config.n_layers, max_depth=4)
        model.forward_to_depth(ids, 4, cache=cache)

        picked = cache.select_rows(torch.tensor([0, 2]), max_depth=4)
        self.assertEqual(picked.keys[0].size(0), 2)
        torch.testing.assert_close(picked.keys[0][0], cache.keys[0][0])

        picked.keys[0].zero_()
        self.assertGreater(float(cache.keys[0].detach().abs().max()), 0.0)


class EndpointReadout(unittest.TestCase):
    """Requirement 7: the vocabulary is projected once per generated token."""

    def test_one_head_call_per_generated_token(self) -> None:
        model = tiny_model()
        model.attach_router(
            RoutingConfig(routing_mode="fixed", fixed_depth=4, probe_depth=1)
        )
        ids = torch.randint(0, 64, (3, 5))

        model.reset_head_counters()
        model.generate_routed(ids, max_new_tokens=7, temperature=0.0)

        # One readout per step, over the whole bucket: seven calls, and three
        # token positions in each. Nothing is projected for the prompt beyond
        # its final position, and no shallower exit is evaluated on the way.
        self.assertEqual(model.head_calls, 7)
        self.assertEqual(model.head_tokens, 7 * 3)

    def test_intermediate_tiers_are_not_projected(self) -> None:
        """Routing to depth 6 must not read the exits at depths 2 and 4."""
        model = tiny_model()
        model.attach_router(
            RoutingConfig(
                routing_mode="request", probe_depth=2, depth_tiers=(2, 4, 6)
            )
        )
        model.depth_controller = ScriptedController([6, 6])

        model.reset_head_counters()
        model.generate_routed(
            torch.randint(0, 64, (2, 4)), max_new_tokens=3, temperature=0.0
        )
        self.assertEqual(model.head_calls, 3)

    def test_readout_without_an_exit_module_raises(self) -> None:
        model = tiny_model()
        state = model.forward_to_depth(torch.randint(0, 64, (1, 3)), 3)
        # exit_every=2 puts exits at depths 2, 4, 6; depth 3 has none.
        with self.assertRaises(ValueError):
            model.endpoint_logits(state.last_token, 3)


class BackwardCompatibility(unittest.TestCase):
    """Requirements 1 and 2: the old paths are untouched."""

    def test_full_depth_forward_is_unchanged_without_routing(self) -> None:
        model = tiny_model()
        ids = torch.randint(0, 64, (2, 8))
        self.assertIsNone(model.routing)

        out = model(ids[:, :-1], targets=ids[:, 1:])
        reference = model._run_blocks(ids[:, :-1])[-1]
        torch.testing.assert_close(
            out.logits, model.exit_modules[-1](reference)
        )

    def test_routed_full_depth_matches_legacy_generation(self) -> None:
        model = tiny_model()
        ids = torch.randint(0, 64, (3, 5))

        legacy = model.generate(
            ids, max_new_tokens=8, temperature=0.0, threshold=0.0
        )
        model.attach_router(
            RoutingConfig(
                routing_mode="fixed",
                fixed_depth=model.config.n_layers,
                probe_depth=1,
            )
        )
        routed = model.generate_routed(ids, max_new_tokens=8, temperature=0.0)

        self.assertTrue(torch.equal(routed.sequences, legacy.sequences))

    def test_routing_mode_none_runs_full_depth(self) -> None:
        model = tiny_model()
        model.attach_router(RoutingConfig(routing_mode="none"))
        out = model.generate_routed(
            torch.randint(0, 64, (2, 4)), max_new_tokens=3, temperature=0.0
        )
        self.assertTrue(bool((out.depths == model.config.n_layers).all()))


class FixedDepthGeneration(unittest.TestCase):
    """Requirement 5: fixed-depth decoding really uses that endpoint."""

    def test_first_token_matches_the_endpoint_readout(self) -> None:
        model = tiny_model()
        ids = torch.randint(0, 64, (2, 6))

        for depth in model.config.exit_depths:
            model.attach_router(
                RoutingConfig(
                    routing_mode="fixed", fixed_depth=depth, probe_depth=1
                )
            )
            routed = model.generate_routed(
                ids, max_new_tokens=1, temperature=0.0
            )
            expected = model.endpoint_logits(
                model.forward_to_depth(ids, depth).last_token, depth
            ).argmax(dim=-1)
            self.assertTrue(
                torch.equal(routed.sequences[:, ids.size(1)], expected)
            )

    def test_depth_changes_the_endpoint_distribution(self) -> None:
        """Otherwise the depth argument is not reaching execution at all.

        Compares logits rather than sampled tokens. An untrained model's argmax
        can easily coincide across depths — the output head is shared, so a
        degenerate initialization sends every depth to the same token — and a
        test that happened to pass because of that would be checking nothing.
        """
        model = tiny_model()
        ids = torch.randint(0, 64, (4, 6))

        with torch.no_grad():
            readouts = [
                model.endpoint_logits(
                    model.forward_to_depth(ids, depth).last_token, depth
                )
                for depth in model.config.exit_depths
            ]
        for shallow, deep in zip(readouts, readouts[1:]):
            self.assertGreater(float((shallow - deep).abs().max()), 1e-4)


class BatchedRouting(unittest.TestCase):
    """Requirement 13: batching must not change any request's output."""

    def test_mixed_depths_match_individual_routing(self) -> None:
        model = tiny_model()
        model.attach_router(
            RoutingConfig(
                routing_mode="request", probe_depth=2, depth_tiers=(2, 4, 6)
            )
        )
        ids = torch.randint(0, 64, (4, 5))
        plan = [2, 6, 4, 2]

        model.depth_controller = ScriptedController(list(plan))
        batched = model.generate_routed(ids, max_new_tokens=6, temperature=0.0)
        self.assertEqual(batched.depths.tolist(), plan)

        for row, depth in enumerate(plan):
            model.depth_controller = ScriptedController([depth])
            single = model.generate_routed(
                ids[row : row + 1], max_new_tokens=6, temperature=0.0
            )
            self.assertTrue(
                torch.equal(single.sequences[0], batched.sequences[row]),
                f"row {row} at depth {depth} changed when batched",
            )

    def test_mixed_prompt_lengths_and_depths(self) -> None:
        model = tiny_model()
        model.attach_router(
            RoutingConfig(
                routing_mode="request", probe_depth=1, depth_tiers=(2, 4, 6)
            )
        )
        ids = torch.randint(1, 64, (4, 6))
        lengths = torch.tensor([3, 6, 4, 6])

        # Groups run in ascending length order: 3, then 4, then 6 (two rows).
        model.depth_controller = ScriptedController([2, 4, 6, 6])
        out = model.generate_routed(
            ids,
            max_new_tokens=4,
            prompt_lengths=lengths,
            temperature=0.0,
            pad_token_id=0,
        )
        self.assertEqual(out.depths.tolist(), [2, 6, 4, 6])

        for row in range(4):
            length = int(lengths[row])
            model.attach_router(
                RoutingConfig(
                    routing_mode="fixed",
                    fixed_depth=int(out.depths[row]),
                    probe_depth=1,
                    depth_tiers=(2, 4, 6),
                )
            )
            single = model.generate_routed(
                ids[row : row + 1, :length], max_new_tokens=4, temperature=0.0
            )
            self.assertTrue(
                torch.equal(single.completions()[0], out.completions()[row]),
                f"row {row} (length {length}) changed when batched",
            )

    def test_every_completion_is_full_length(self) -> None:
        model = tiny_model()
        model.attach_router(
            RoutingConfig(routing_mode="fixed", fixed_depth=4, probe_depth=1)
        )
        ids = torch.randint(1, 64, (3, 5))
        out = model.generate_routed(
            ids,
            max_new_tokens=5,
            prompt_lengths=torch.tensor([2, 5, 3]),
            temperature=0.0,
        )
        for completion in out.completions():
            self.assertEqual(completion.numel(), 5)


class ControllerBehaviour(unittest.TestCase):
    """Requirements 8, 9, and 12."""

    def test_controller_sees_only_probe_features(self) -> None:
        """Requirement 8: no targets, no labels, no final-layer state."""
        model = tiny_model()
        routing = RoutingConfig(
            routing_mode="request", probe_depth=2, depth_tiers=(2, 4, 6)
        )
        controller = model.attach_router(routing)

        seen: list[torch.Tensor] = []
        original = controller.forward
        controller.forward = lambda features: seen.append(features) or original(
            features
        )

        ids = torch.randint(0, 64, (3, 5))
        model.generate_routed(ids, max_new_tokens=2, temperature=0.0)

        self.assertEqual(len(seen), 1)
        # The only tensor reaching the controller has the width of pooled probe
        # features, so it cannot be carrying logits, targets, or a deep state.
        self.assertEqual(seen[0].shape, (3, controller.feature_dim))

        probe = model.forward_to_depth(ids, routing.probe_depth)
        expected = pool_prompt_features(
            probe.hidden,
            pooling=routing.controller_pooling,
            include_length=routing.controller_use_length,
            max_seq_len=model.config.max_seq_len,
        )
        torch.testing.assert_close(seen[0], expected)

    def test_probe_prefix_is_reused_instead_of_recomputed(self) -> None:
        """Every prompt token should traverse each selected block at most once."""
        model = tiny_model()
        routing = RoutingConfig(
            routing_mode="request", probe_depth=2, depth_tiers=(2, 4, 6)
        )
        model.attach_router(routing, ScriptedController([2, 4, 6]))

        token_visits = [0] * model.config.n_layers
        handles = []
        for layer, block in enumerate(model.blocks):
            def count(_module, args, _output, layer=layer):
                hidden = args[0]
                token_visits[layer] += hidden.size(0) * hidden.size(1)

            handles.append(block.register_forward_hook(count))

        try:
            prompt_len = 5
            out = model.generate_routed(
                torch.randint(0, 64, (3, prompt_len)),
                max_new_tokens=1,
                temperature=0.0,
            )
        finally:
            for handle in handles:
                handle.remove()

        # All three requests run the two-block probe, two continue to depth 4,
        # and one continues to depth 6. A recomputed probe would double the
        # first two entries while the counters still reported these values.
        self.assertEqual(
            token_visits,
            [3 * prompt_len, 3 * prompt_len,
             2 * prompt_len, 2 * prompt_len,
             1 * prompt_len, 1 * prompt_len],
        )
        self.assertEqual(out.counters.total_block_executions, sum(token_visits))

    def test_selection_is_deterministic_in_eval_mode(self) -> None:
        """Requirement 9."""
        model = tiny_model()
        model.attach_router(
            RoutingConfig(
                routing_mode="request",
                probe_depth=2,
                depth_tiers=(2, 4, 6),
                deterministic_routing=True,
            )
        )
        ids = torch.randint(0, 64, (5, 6))

        first = model.generate_routed(ids, max_new_tokens=2, temperature=0.0)
        for _ in range(3):
            again = model.generate_routed(ids, max_new_tokens=2, temperature=0.0)
            self.assertTrue(torch.equal(first.depths, again.depths))

    def test_lambda_shifts_routing_shallower(self) -> None:
        """A price on compute must actually change the decision."""
        torch.manual_seed(3)
        controller = DepthController(d_model=8, n_tiers=3, hidden_dim=8)
        features = torch.randn(64, controller.feature_dim)
        tiers = (2, 4, 6)

        cheap, _ = controller.select(features, tiers, routing_lambda=0.0)
        dear, _ = controller.select(features, tiers, routing_lambda=100.0)
        self.assertLessEqual(float(dear.float().mean()), float(cheap.float().mean()))
        self.assertEqual(int(dear.max()), 2)

    def test_ordinal_head_probabilities_are_monotone(self) -> None:
        torch.manual_seed(5)
        controller = DepthController(
            d_model=8, n_tiers=4, hidden_dim=8, output="ordinal"
        )
        with torch.no_grad():
            controller.cut_increments.normal_()
            controller.head.weight.normal_()

        with torch.no_grad():
            probabilities = controller(torch.randn(32, controller.feature_dim))
        differences = probabilities[:, 1:] - probabilities[:, :-1]
        self.assertGreaterEqual(float(differences.min()), -1e-6)
        torch.testing.assert_close(probabilities[:, -1], torch.ones(32))

    def test_pooling_respects_padding(self) -> None:
        """Requirement 12: a mean over padding is not a mean over the prompt."""
        torch.manual_seed(0)
        hidden = torch.randn(2, 6, 4)
        lengths = torch.tensor([3, 6])

        pooled = pool_prompt_features(hidden, lengths, pooling="last_mean")
        torch.testing.assert_close(pooled[0, :4], hidden[0, 2])
        torch.testing.assert_close(pooled[0, 4:], hidden[0, :3].mean(dim=0))
        torch.testing.assert_close(pooled[1, :4], hidden[1, 5])
        torch.testing.assert_close(pooled[1, 4:], hidden[1].mean(dim=0))

    def test_padding_beyond_the_length_is_ignored(self) -> None:
        torch.manual_seed(0)
        hidden = torch.randn(1, 6, 4)
        lengths = torch.tensor([3])
        before = pool_prompt_features(hidden, lengths, pooling="last_mean")

        hidden[:, 3:] = 999.0
        after = pool_prompt_features(hidden, lengths, pooling="last_mean")
        torch.testing.assert_close(before, after)

    def test_length_feature_is_normalized(self) -> None:
        hidden = torch.zeros(2, 8, 4)
        pooled = pool_prompt_features(
            hidden,
            torch.tensor([2, 8]),
            pooling="last",
            include_length=True,
            max_seq_len=16,
        )
        torch.testing.assert_close(pooled[:, -1], torch.tensor([0.125, 0.5]))

    def test_bad_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pool_prompt_features(torch.zeros(1, 4, 2), torch.tensor([9]))
        with self.assertRaises(ValueError):
            pool_prompt_features(torch.zeros(1, 4, 2), torch.tensor([0]))


class TierValidation(unittest.TestCase):
    """Requirement 10: bad candidate sets fail loudly and specifically."""

    def setUp(self) -> None:
        self.config = tiny_config()  # exits at depths 2, 4, 6

    def _resolve(self, **updates):
        return RoutingConfig(**updates).resolve(self.config)

    def test_unsorted_tiers(self) -> None:
        with self.assertRaisesRegex(ValueError, "ascending"):
            self._resolve(depth_tiers=(4, 2, 6))

    def test_duplicate_tiers(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            self._resolve(depth_tiers=(2, 2, 6))

    def test_out_of_range_tiers(self) -> None:
        with self.assertRaisesRegex(ValueError, r"outside \[1, 6\]"):
            self._resolve(depth_tiers=(2, 6, 9))

    def test_tier_without_an_exit_module(self) -> None:
        with self.assertRaisesRegex(ValueError, "No exit module"):
            self._resolve(depth_tiers=(2, 3, 6))

    def test_full_depth_must_be_reachable(self) -> None:
        with self.assertRaisesRegex(ValueError, "fallback"):
            self._resolve(depth_tiers=(2, 4))

    def test_probe_deeper_than_shallowest_tier(self) -> None:
        with self.assertRaisesRegex(ValueError, "probe_depth"):
            self._resolve(depth_tiers=(2, 4, 6), probe_depth=3)

    def test_fixed_depth_must_be_a_tier(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed_depth"):
            self._resolve(routing_mode="fixed", fixed_depth=3)

    def test_safety_depth_must_be_a_tier(self) -> None:
        with self.assertRaisesRegex(ValueError, "safety_depth"):
            self._resolve(safety_depth=5)

    def test_unknown_enumerations(self) -> None:
        for field, value in (
            ("routing_mode", "sideways"),
            ("controller_pooling", "median"),
            ("controller_output", "regression"),
        ):
            with self.assertRaises(ValueError):
                RoutingConfig(**{field: value})

    def test_defaults_resolve_to_the_model_exits(self) -> None:
        resolved = self._resolve()
        self.assertEqual(resolved.depth_tiers, self.config.exit_depths)
        self.assertEqual(resolved.selectable_tiers, self.config.exit_depths)

    def test_min_depth_narrows_the_selectable_set(self) -> None:
        resolved = self._resolve(depth_tiers=(2, 4, 6), min_depth=4)
        self.assertEqual(resolved.selectable_tiers, (4, 6))

    def test_safety_depth_raises_shallow_routes(self) -> None:
        model = tiny_model()
        model.attach_router(
            RoutingConfig(
                routing_mode="request",
                probe_depth=1,
                depth_tiers=(2, 4, 6),
                safety_depth=4,
            )
        )
        model.depth_controller = ScriptedController([2, 2])
        out = model.generate_routed(
            torch.randint(0, 64, (2, 4)), max_new_tokens=2, temperature=0.0
        )
        self.assertEqual(out.depths.tolist(), [4, 4])
        self.assertEqual(
            out.trace.fallback_reasons, ["safety_depth_floor"] * 2
        )


class GroupedQueryAttention(unittest.TestCase):
    """Requirement 11: GQA holds in prefix, suffix, and capped-cache paths."""

    def test_all_paths_agree_under_gqa(self) -> None:
        for n_kv_heads in (1, 2, 4):
            with self.subTest(n_kv_heads=n_kv_heads):
                model = tiny_model(n_kv_heads=n_kv_heads)
                self.assertEqual(model.config.n_kv_heads, n_kv_heads)
                ids = torch.randint(0, 64, (2, 6))

                reference = model.forward_to_depth(ids, 6)
                prefix = model.forward_to_depth(
                    ids, 2, return_boundary_state=True
                )
                suffix = model.continue_from_depth(prefix.boundary, 2, 6)
                torch.testing.assert_close(suffix.hidden, reference.hidden)

                cache = KVCache(model.config.n_layers, max_depth=4)
                cached = model.forward_to_depth(ids[:, :5], 4, cache=cache)
                step = model.forward_to_depth(ids[:, 5:], 4, cache=cache)
                direct = model.forward_to_depth(ids, 4)
                torch.testing.assert_close(
                    step.hidden[:, -1], direct.hidden[:, -1],
                    atol=1e-5, rtol=1e-5,
                )
                self.assertEqual(cached.depth, 4)


class Escalation(unittest.TestCase):
    """Phase 6: escalation is exact when the retained state is sufficient."""

    def test_escalation_matches_full_depth(self) -> None:
        model = tiny_model()
        ids = torch.randint(0, 64, (2, 5))

        cache = KVCache(model.config.n_layers, max_depth=2)
        shallow = model.forward_to_depth(
            ids, 2, cache=cache, return_boundary_state=True
        )
        state, widened = model.escalate(cache, shallow.boundary, 2, 6)

        reference = model.forward_to_depth(ids, 6)
        torch.testing.assert_close(state.hidden, reference.hidden)
        self.assertEqual(widened.active_depth, 6)
        self.assertTrue(all(widened.layer_presence))

    def test_escalated_cache_continues_correctly(self) -> None:
        """The widened cache must decode as though it had always been deep."""
        model = tiny_model()
        ids = torch.randint(0, 64, (2, 5))
        follow = torch.randint(0, 64, (2, 1))

        cache = KVCache(model.config.n_layers, max_depth=2)
        shallow = model.forward_to_depth(
            ids, 2, cache=cache, return_boundary_state=True
        )
        _, widened = model.escalate(cache, shallow.boundary, 2, 6)
        after = model.forward_to_depth(follow, 6, cache=widened)

        deep_cache = KVCache(model.config.n_layers)
        model.forward_to_depth(ids, 6, cache=deep_cache)
        expected = model.forward_to_depth(follow, 6, cache=deep_cache)

        torch.testing.assert_close(
            after.hidden, expected.hidden, atol=1e-5, rtol=1e-5
        )

    def test_escalation_must_go_deeper(self) -> None:
        model = tiny_model()
        cache = KVCache(model.config.n_layers, max_depth=4)
        state = model.forward_to_depth(
            torch.randint(0, 64, (1, 3)), 4, cache=cache,
            return_boundary_state=True,
        )
        with self.assertRaises(ValueError):
            model.escalate(cache, state.boundary, 4, 4)

    def test_original_cache_is_left_intact(self) -> None:
        model = tiny_model()
        ids = torch.randint(0, 64, (2, 4))
        cache = KVCache(model.config.n_layers, max_depth=2)
        state = model.forward_to_depth(
            ids, 2, cache=cache, return_boundary_state=True
        )
        model.escalate(cache, state.boundary, 2, 6)

        self.assertEqual(cache.active_depth, 2)
        self.assertEqual(
            cache.layer_presence, (True, True, False, False, False, False)
        )


class CostAccounting(unittest.TestCase):
    """Requirement 14: counters agree with hand-calculated values."""

    def test_block_and_head_formulas(self) -> None:
        config = TransformerConfig(
            vocab_size=100, d_model=10, n_layers=4, n_heads=2, n_kv_heads=1,
            ff_dim=20, max_seq_len=16,
        )
        cost = AnalyticalCostModel.from_config(config)

        # rho = 1/2, so projections are (2 + 2*0.5) * 100 = 300.
        self.assertEqual(cost.projection_macs, 300.0)
        self.assertEqual(cost.ffn_macs, 3 * 10 * 20)
        self.assertEqual(cost.attention_macs(7), 2 * 7 * 10)
        self.assertEqual(cost.block_macs(7), 300.0 + 600.0 + 140.0)
        self.assertEqual(cost.head_macs, 10 * 100)
        self.assertEqual(cost.kv_projection_macs, 100.0)

    def test_kv_bytes_formula(self) -> None:
        config = TransformerConfig(
            vocab_size=100, d_model=10, n_layers=4, n_heads=2, n_kv_heads=1,
            ff_dim=20, max_seq_len=16,
        )
        cost = AnalyticalCostModel.from_config(config)
        # 2 tensors * 2 bytes * 8 tokens * 3 layers * (1 head * 5 dims)
        self.assertEqual(cost.kv_bytes(3, 8, "bf16"), 2 * 2 * 8 * 3 * 5)

    def test_prefill_attention_is_quadratic(self) -> None:
        config = TransformerConfig(
            vocab_size=100, d_model=10, n_layers=4, n_heads=2, n_kv_heads=1,
            ff_dim=20, max_seq_len=16,
        )
        cost = AnalyticalCostModel.from_config(config)
        # depth 1, prompt 3: per-token 900 each, context 10 * 3 * 4 = 120.
        self.assertEqual(cost.prefill_macs(1, 3), 3 * 900 + 120)

    def test_counters_reproduce_the_formula(self) -> None:
        counters = CostCounters()
        counters.record_blocks(depth=3, tokens=2, context_len=5)
        self.assertEqual(counters.total_block_executions, 6)
        self.assertEqual(counters.attention_position_sum, 3 * 2 * 5)

        counters.record_prefill(depth=2, prompt_len=3, rows=1)
        self.assertEqual(counters.total_block_executions, 6 + 6)
        # Positions 1 + 2 + 3 = 6, over two blocks.
        self.assertEqual(counters.attention_position_sum, 30 + 12)

    def test_suffix_continuation_charges_only_its_own_blocks(self) -> None:
        counters = CostCounters()
        counters.record_prefill(depth=6, prompt_len=4, rows=1, start_depth=2)
        self.assertEqual(counters.total_block_executions, 4 * 4)
        self.assertEqual(sorted(counters.block_executions), [2, 3, 4, 5])

    def test_routed_generation_counters_match_by_hand(self) -> None:
        model = tiny_model()
        model.attach_router(
            RoutingConfig(routing_mode="fixed", fixed_depth=4, probe_depth=1)
        )
        prompt_len, rows, new_tokens = 5, 2, 3
        out = model.generate_routed(
            torch.randint(0, 64, (rows, prompt_len)),
            max_new_tokens=new_tokens,
            temperature=0.0,
        )

        # Prefill runs 4 blocks over 5 positions for 2 rows; decode runs 4
        # blocks for 2 rows on each of the 2 continuation steps.
        expected = 4 * prompt_len * rows + 4 * rows * (new_tokens - 1)
        self.assertEqual(out.counters.total_block_executions, expected)
        self.assertEqual(out.counters.head_tokens, rows * new_tokens)
        self.assertEqual(out.counters.decode_tokens, rows * (new_tokens - 1))

    def test_estimated_macs_sums_to_its_parts(self) -> None:
        model = tiny_model()
        model.attach_router(
            RoutingConfig(routing_mode="fixed", fixed_depth=2, probe_depth=1)
        )
        out = model.generate_routed(
            torch.randint(0, 64, (2, 4)), max_new_tokens=3, temperature=0.0
        )
        parts = {k: v for k, v in out.estimated_macs.items() if k != "total"}
        self.assertAlmostEqual(
            sum(parts.values()), out.estimated_macs["total"], places=3
        )

    def test_shallow_routing_costs_less(self) -> None:
        model = tiny_model()
        ids = torch.randint(0, 64, (2, 6))

        totals = {}
        for depth in (2, 4, 6):
            model.attach_router(
                RoutingConfig(
                    routing_mode="fixed", fixed_depth=depth, probe_depth=1
                )
            )
            out = model.generate_routed(ids, max_new_tokens=4, temperature=0.0)
            totals[depth] = out.estimated_macs["total"]
        self.assertLess(totals[2], totals[4])
        self.assertLess(totals[4], totals[6])


class StatisticsUtilities(unittest.TestCase):
    """Requirement 15: the inference helpers behave on known arrays."""

    def test_bootstrap_recovers_a_known_mean(self) -> None:
        values = np.full(200, 0.25)
        interval = paired_bootstrap(values, resamples=500, seed=0)
        self.assertAlmostEqual(interval.estimate, 0.25, places=9)
        self.assertAlmostEqual(interval.low, 0.25, places=9)
        self.assertAlmostEqual(interval.high, 0.25, places=9)

    def test_bootstrap_brackets_a_noisy_mean(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(0.5, 1.0, size=2000)
        interval = paired_bootstrap(values, resamples=500, seed=1)
        self.assertLess(interval.low, 0.5)
        self.assertGreater(interval.high, 0.5)
        self.assertFalse(interval.excludes(0.5))

    def test_bootstrap_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap([])

    def test_pareto_frontier_drops_dominated_points(self) -> None:
        points = [(1.0, 0.5), (2.0, 0.4), (2.0, 0.7), (3.0, 0.7), (4.0, 0.9)]
        self.assertEqual(
            pareto_frontier(points), [(1.0, 0.5), (2.0, 0.7), (4.0, 0.9)]
        )

    def test_frontier_interpolation_refuses_to_extrapolate(self) -> None:
        frontier = [(1.0, 0.0), (3.0, 1.0)]
        self.assertAlmostEqual(interpolate_frontier(frontier, 2.0), 0.5)
        self.assertIsNone(interpolate_frontier(frontier, 0.5))
        self.assertIsNone(interpolate_frontier(frontier, 3.5))

    def test_common_cost_range(self) -> None:
        a = [(1.0, 0.0), (5.0, 1.0)]
        b = [(2.0, 0.0), (9.0, 1.0)]
        self.assertEqual(common_cost_range(a, b), (2.0, 5.0))
        self.assertIsNone(common_cost_range(a, [(6.0, 0.0), (9.0, 1.0)]))

    def test_substitution_ratio_of_identical_frontiers_is_one(self) -> None:
        frontier = [(1.0, 0.2), (3.0, 0.8)]
        ratio = vertical_substitution_ratio(frontier, frontier, 0.2, 2.0)
        self.assertAlmostEqual(ratio, 1.0)

    def test_substitution_ratio_halves_with_half_the_gain(self) -> None:
        horizontal = [(1.0, 0.0), (3.0, 1.0)]
        vertical = [(1.0, 0.0), (3.0, 0.5)]
        self.assertAlmostEqual(
            vertical_substitution_ratio(vertical, horizontal, 0.0, 3.0), 0.5
        )

    def test_substitution_ratio_withheld_without_a_gain(self) -> None:
        """A horizontal system that gained nothing has nothing to substitute."""
        flat = [(1.0, 0.4), (3.0, 0.4001)]
        self.assertIsNone(vertical_substitution_ratio(flat, flat, 0.4, 2.0))
        self.assertIsNone(integrated_substitution_ratio(flat, flat, 0.4))

        # The threshold is declared, not hard-coded: lowering it below the
        # observed gain brings the ratio back.
        self.assertAlmostEqual(
            vertical_substitution_ratio(flat, flat, 0.4, 2.0, min_gain=1e-9),
            1.0,
        )

    def test_integrated_ratio_over_the_common_range(self) -> None:
        horizontal = [(1.0, 0.0), (5.0, 1.0)]
        vertical = [(1.0, 0.0), (5.0, 0.5)]
        ratio, span = integrated_substitution_ratio(vertical, horizontal, 0.0)
        self.assertAlmostEqual(ratio, 0.5, places=6)
        self.assertEqual(span, (1.0, 5.0))

    def test_non_inferiority_passes_on_an_identical_system(self) -> None:
        rng = np.random.default_rng(0)
        quality = rng.normal(0.7, 0.1, size=400)
        cost = rng.normal(1.0, 0.05, size=400)
        result = non_inferiority_test(
            quality, quality, cost, cost,
            quality_margin=0.01, cost_tolerance=0.0,
            resamples=400,
        )
        self.assertTrue(result.substitution_supported)

    def test_non_inferiority_fails_on_a_worse_system(self) -> None:
        rng = np.random.default_rng(1)
        horizontal = rng.normal(0.7, 0.05, size=400)
        vertical = horizontal - 0.2
        cost = np.ones(400)
        result = non_inferiority_test(
            vertical, horizontal, cost, cost,
            quality_margin=0.01, cost_tolerance=0.0,
            resamples=400,
        )
        self.assertFalse(result.quality_passes)
        self.assertFalse(result.substitution_supported)

    def test_underpowered_study_does_not_support_substitution(self) -> None:
        """An interval containing zero is not evidence of equivalence."""
        rng = np.random.default_rng(2)
        horizontal = rng.normal(0.7, 0.5, size=8)
        vertical = rng.normal(0.7, 0.5, size=8)
        result = non_inferiority_test(
            vertical, horizontal, np.ones(8), np.ones(8),
            quality_margin=0.001, cost_tolerance=0.0,
            resamples=400,
        )
        self.assertFalse(result.quality_passes)

    def test_unpaired_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            non_inferiority_test([1.0, 2.0], [1.0], [1.0, 2.0], [1.0, 2.0],
                                 quality_margin=0.1, cost_tolerance=0.0)


if __name__ == "__main__":
    unittest.main()
