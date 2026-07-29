"""Tests for the independent model family and the cross-tokenizer comparison.

The horizontal side is the half of the central comparison that did not exist. The
hard part is not scoring the models; it is that they use a different tokenizer
from the backbone they are being compared against, which makes per-token loss
incomparable. A tokenizer that splits text more finely earns a lower average loss
per piece without predicting anything better.

So the tests here concentrate on the guards rather than the arithmetic:

* quality must be stated in a tokenizer-independent unit, and a mismatch between
  the manifest's unit and the vertical side's must be refused, not averaged;
* bits per byte is lower-is-better, so it must be negated exactly once — a sign
  error inverts every policy that reads it;
* the two sides must be paired by *content*, because the vertical side reports on
  a frozen subset while the family is scored on every drawn request;
* cost must be shape-aware, since prefill scales with the prompt and decode with
  the growing context.

No test here downloads a model. The scoring path is exercised with a stub whose
tokenizer and loss are chosen so the expected bits-per-byte is computable by hand.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.evaluate_vertical_routing import (
    check_units_comparable,
    horizontal_systems,
    load_manifest,
)
from experiments.horizontal_family import (
    PYTHIA_FAMILY,
    PYTHIA_SUITE,
    ModelResult,
    measure_cost_profile,
    score_bits_per_byte,
    tokenizer_fingerprint,
    write_manifest,
    _check_one_tokenizer,
    _check_same_requests,
)
from experiments.workloads import RequestShape, Workload


class StubTokenizer:
    """A whitespace tokenizer with a controllable identity.

    Args:
        vocab_size: Reported vocabulary size.
        offset: Added to every id, so two instances with different offsets encode
            the same text differently and must fingerprint differently.
    """

    def __init__(self, vocab_size: int = 100, offset: int = 0) -> None:
        self.vocab_size = vocab_size
        self.offset = offset

    def __call__(self, text: str, return_tensors=None) -> dict:
        """Encodes text as one id per whitespace-separated word."""
        words = text.split()
        return {"input_ids": [len(word) + self.offset for word in words]}

    def __len__(self) -> int:
        return self.vocab_size


class StubModel(torch.nn.Module):
    """A maximally uninformed model: uniform logits at every position.

    Cross-entropy against *any* target is then exactly ``ln(vocab_size)``, which
    makes the expected bits-per-byte computable by hand and independent of which
    ids the tokenizer produced. An earlier version of this stub boosted the logit
    of the current token to hit a configurable loss; it appeared to work only
    because every word in the fixture has two letters, so the whitespace
    tokenizer emitted a constant id and the boosted position happened to be the
    gold one. That is the kind of accident a stub must not contain.

    Attributes:
        nll: Per-token cross-entropy this model produces, ``ln(vocab_size)``.
    """

    def __init__(self, vocab_size: int = 100) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(4))

        class Config:
            num_hidden_layers = 2
            hidden_size = 8

        self.config = Config()
        self.config.vocab_size = vocab_size
        self.nll = math.log(vocab_size)

    def forward(self, input_ids: torch.Tensor):
        """Emits uniform logits, so every token costs ``ln(vocab_size)`` nats."""
        rows, length = input_ids.shape

        class Output:
            pass

        out = Output()
        out.logits = torch.zeros(rows, length, self.config.vocab_size)
        return out


def stub_workload(
    continuations: list[str] | None = None, prompt: str = "aa bb cc"
) -> Workload:
    """Builds a workload carrying decoded text and byte lengths."""
    continuations = continuations or ["dd ee", "ff gg hh"]
    rows = len(continuations)
    return Workload(
        prompts=torch.zeros(rows, 3, dtype=torch.long),
        references=torch.zeros(rows, 2, dtype=torch.long),
        difficulty=["unknown"] * rows,
        source_ids=list(range(rows)),
        document_ids=list(range(rows)),
        token_hashes=[f"tok{index}" for index in range(rows)],
        text_hashes=[f"txt{index}" for index in range(rows)],
        continuation_bytes=[len(text.encode("utf-8")) for text in continuations],
        texts=[(prompt, " " + text) for text in continuations],
        spec={"shape": "p3c2", "prompt_len": 3, "continuation_len": 2},
    )


class BitsPerByteScoring(unittest.TestCase):
    """The tokenizer-independent unit, and the reason it is used."""

    def test_the_value_matches_the_definition(self) -> None:
        """Computed by hand, so a change in convention cannot pass unnoticed."""
        model = StubModel()
        workload = stub_workload(["dd ee"])
        scored = score_bits_per_byte(
            model, StubTokenizer(), workload, batch_size=4
        )

        tokens = scored["scored_tokens"][0]
        expected = (
            tokens * model.nll / (math.log(2.0) * workload.continuation_bytes[0])
        )
        self.assertAlmostEqual(scored["bits_per_byte"][0], expected, places=4)

    def test_the_denominator_does_not_depend_on_the_tokenizer(self) -> None:
        """Why bytes are the right denominator, stated precisely.

        Tokenizer granularity changes the *number of terms* in the sum, so a
        per-token average is a quantity about the tokenizer as much as about the
        model. It does not change the number of UTF-8 bytes in the text.

        This is the mechanical half of the argument and all a stub can show. The
        substantive half is the chain rule: for a model that assigns probability
        to strings, the total log-probability of a string is the same however the
        string is segmented, so a *total* divided by a byte count is comparable
        across tokenizers while a per-token mean is not. This stub is uniform
        rather than calibrated, so it cannot demonstrate that half — real Pythia
        does, and the 70m-versus-160m gap on the same text is the evidence.
        """
        workload = stub_workload(["dd ee ff gg"])
        coarse = score_bits_per_byte(
            StubModel(), StubTokenizer(), workload, batch_size=4
        )

        class Finer(StubTokenizer):
            def __call__(self, text, return_tensors=None):
                ids = []
                for word in text.split():
                    ids.extend([len(word), len(word) + 50])
                return {"input_ids": ids}

        fine = score_bits_per_byte(StubModel(), Finer(), workload, batch_size=4)

        self.assertGreater(
            fine["scored_tokens"][0],
            coarse["scored_tokens"][0],
            "the finer tokenizer should produce more tokens",
        )
        # The denominator is the same for both, being a property of the text.
        for scored in (coarse, fine):
            self.assertAlmostEqual(
                scored["bits_per_byte"][0],
                scored["nll_sum"][0]
                / (math.log(2.0) * workload.continuation_bytes[0]),
                places=6,
            )
        # And a per-token mean would have differed purely from segmentation.
        coarse_mean = coarse["nll_sum"][0] / coarse["scored_tokens"][0]
        fine_mean = fine["nll_sum"][0] / fine["scored_tokens"][0]
        self.assertAlmostEqual(coarse_mean, fine_mean, places=6)
        self.assertNotAlmostEqual(
            coarse["nll_sum"][0], fine["nll_sum"][0], places=3
        )

    def test_an_undecoded_workload_is_refused(self) -> None:
        bare = stub_workload()
        bare.texts = None
        with self.assertRaisesRegex(ValueError, "no decoded text"):
            score_bits_per_byte(StubModel(), StubTokenizer(), bare)

    def test_a_merged_boundary_is_recorded_as_missing_not_as_zero(self) -> None:
        """Zero loss on an unscorable request would look like a perfect score."""
        workload = stub_workload([""])
        scored = score_bits_per_byte(
            StubModel(), StubTokenizer(), workload, batch_size=1
        )
        self.assertTrue(math.isnan(scored["bits_per_byte"][0]))
        self.assertEqual(scored["scored_tokens"][0], 0)


class TokenizerIdentity(unittest.TestCase):
    """Identity by behaviour, not by the path a tokenizer was loaded from."""

    def test_the_same_tokenizer_fingerprints_alike(self) -> None:
        self.assertEqual(
            tokenizer_fingerprint(StubTokenizer()),
            tokenizer_fingerprint(StubTokenizer()),
        )

    def test_a_different_encoding_fingerprints_differently(self) -> None:
        self.assertNotEqual(
            tokenizer_fingerprint(StubTokenizer(offset=0)),
            tokenizer_fingerprint(StubTokenizer(offset=7)),
        )

    def test_a_different_vocabulary_size_fingerprints_differently(self) -> None:
        self.assertNotEqual(
            tokenizer_fingerprint(StubTokenizer(vocab_size=100)),
            tokenizer_fingerprint(StubTokenizer(vocab_size=200)),
        )

    def test_a_family_with_one_tokenizer_is_accepted(self) -> None:
        shared = tokenizer_fingerprint(StubTokenizer())
        _check_one_tokenizer(
            [
                ModelResult("a", shared, 1, 1, [0.5], ["h"]),
                ModelResult("b", shared, 2, 2, [0.4], ["h"]),
            ]
        )

    def test_a_family_mixing_tokenizers_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "mixes tokenizers"):
            _check_one_tokenizer(
                [
                    ModelResult("a", "tokenizer-one", 1, 1, [0.5], ["h"]),
                    ModelResult("b", "tokenizer-two", 2, 2, [0.4], ["h"]),
                ]
            )

    def test_models_scoring_different_requests_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "different requests"):
            _check_same_requests(
                [
                    ModelResult("a", "t", 1, 1, [0.5], ["h1"]),
                    ModelResult("b", "t", 2, 2, [0.4], ["h2"]),
                ]
            )


class FamilyMetadata(unittest.TestCase):
    """What is held constant decides what a difference between tiers means."""

    def test_the_suite_is_ordered_by_capacity(self) -> None:
        sizes = []
        for name in PYTHIA_SUITE:
            tail = name.rsplit("-", 1)[1]
            value = float(tail.rstrip("mb"))
            sizes.append(value * (1000 if tail.endswith("b") else 1))
        self.assertEqual(sizes, sorted(sizes))

    def test_the_family_records_what_it_controls(self) -> None:
        for field in ("family", "held_constant", "varies", "note"):
            with self.subTest(field=field):
                self.assertIn(field, PYTHIA_FAMILY)
        self.assertIn("tokenizer", PYTHIA_FAMILY["held_constant"])
        self.assertIn("training corpus", PYTHIA_FAMILY["held_constant"])

    def test_the_note_states_the_uncontrolled_comparison(self) -> None:
        """The vertical-versus-Pythia comparison is not a controlled one."""
        self.assertIn("not* controlled", PYTHIA_FAMILY["note"])
        self.assertIn("upper bound", PYTHIA_FAMILY["note"])


class CostProfile(unittest.TestCase):
    """Cost is a function of request shape, not a scalar per model."""

    def test_a_longer_shape_costs_more(self) -> None:
        model, tokenizer = StubModel(), StubTokenizer()
        short = measure_cost_profile(
            model, tokenizer, RequestShape(32, 16), 1000, 1.0, 10
        )
        long = measure_cost_profile(
            model, tokenizer, RequestShape(256, 128), 1000, 1.0, 10
        )
        self.assertGreater(long["analytical_macs"], short["analytical_macs"])

    def test_measured_time_is_kept_apart_from_the_estimate(self) -> None:
        """A MAC count is not a latency and must not stand in for one."""
        profile = measure_cost_profile(
            StubModel(), StubTokenizer(), RequestShape(64, 32), 1000, 4.0, 8
        )
        self.assertIn("analytical_macs", profile)
        self.assertIn("seconds_per_request", profile)
        self.assertAlmostEqual(profile["seconds_per_request"], 0.5)

    def test_the_shape_is_recorded_with_the_cost(self) -> None:
        profile = measure_cost_profile(
            StubModel(), StubTokenizer(), RequestShape(64, 32), 1000, 1.0, 1
        )
        self.assertEqual(profile["prompt_len"], 64.0)
        self.assertEqual(profile["continuation_len"], 32.0)


class ManifestContract(unittest.TestCase):
    """What the evaluation requires of a manifest before it will use one."""

    def build(self, directory: Path, **overrides) -> Path:
        results = [
            ModelResult(
                model_id="fam/small",
                tokenizer_id="stub:100:abcd",
                parameters=1_000,
                resident_bytes=4_000,
                bits_per_byte=[1.2, 1.1, 1.3],
                text_hashes=["a", "b", "c"],
                cost_profile={"p3c2": {"analytical_macs": 100.0}},
                tier=0,
            ),
            ModelResult(
                model_id="fam/large",
                tokenizer_id="stub:100:abcd",
                parameters=4_000,
                resident_bytes=16_000,
                bits_per_byte=[0.9, 0.8, 1.0],
                text_hashes=["a", "b", "c"],
                cost_profile={"p3c2": {"analytical_macs": 400.0}},
                tier=1,
            ),
        ]
        metadata = {
            "quality_unit": "bits_per_byte",
            "quality_direction": "lower_is_better",
            "family": PYTHIA_FAMILY,
            "hardware": {"device_type": "cpu"},
        }
        metadata.update(overrides)
        return write_manifest(results, metadata, directory)

    def test_a_manifest_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entries = load_manifest(self.build(Path(directory)))
        self.assertEqual([entry["tier"] for entry in entries], [0, 1])
        self.assertEqual(entries[0]["quality_unit"], "bits_per_byte")
        self.assertIn("cost_profile", entries[0])
        self.assertIn("parameters", entries[0])

    def test_matching_units_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entries = load_manifest(self.build(Path(directory)))
        self.assertEqual(check_units_comparable(entries, "bits_per_byte"), "bits_per_byte")

    def test_a_unit_mismatch_is_refused(self) -> None:
        """The guard that stops an invalid cross-tokenizer comparison."""
        with tempfile.TemporaryDirectory() as directory:
            entries = load_manifest(self.build(Path(directory)))
        with self.assertRaisesRegex(ValueError, "different quantities"):
            check_units_comparable(entries, "teacher_forced_accuracy")

    def test_an_unstated_unit_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.build(Path(directory))
            entries = json.loads(path.read_text())
            for entry in entries:
                entry.pop("quality_unit")
            path.write_text(json.dumps(entries))
            loaded = load_manifest(path)
        with self.assertRaisesRegex(ValueError, "does not state its quality unit"):
            check_units_comparable(loaded, "bits_per_byte")

    def test_lower_is_better_quality_is_negated_once(self) -> None:
        """A sign error here inverts every policy that reads the column."""
        with tempfile.TemporaryDirectory() as directory:
            entries = load_manifest(self.build(Path(directory)))
            systems, quality, cost = horizontal_systems(
                entries, n_requests=3, lambdas=(0.0,), shapes=["p3c2"] * 3
            )

        small = next(s for s in systems if s.name.endswith("fam/small"))
        large = next(s for s in systems if s.name.endswith("fam/large"))
        # 0.9 bits/byte is better than 1.2, so after orientation the larger model
        # must have the higher quality.
        self.assertGreater(large.mean_quality, small.mean_quality)
        self.assertAlmostEqual(small.mean_quality, -(1.2 + 1.1 + 1.3) / 3, places=6)

    def test_the_shape_aware_cost_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entries = load_manifest(self.build(Path(directory)))
            _, _, cost = horizontal_systems(
                entries, n_requests=3, lambdas=(0.0,), shapes=["p3c2"] * 3
            )
        # Normalized against the deepest tier, so the ratio is 100/400.
        self.assertAlmostEqual(cost[0, 0] / cost[0, 1], 0.25, places=6)

    def test_a_missing_shape_in_the_profile_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entries = load_manifest(self.build(Path(directory)))
            with self.assertRaisesRegex(ValueError, "no cost profile"):
                horizontal_systems(
                    entries, n_requests=3, lambdas=(0.0,), shapes=["p999c1"] * 3
                )

    def test_pairing_is_by_content_not_position(self) -> None:
        """The vertical side reports on a frozen subset of what was scored."""
        with tempfile.TemporaryDirectory() as directory:
            entries = load_manifest(self.build(Path(directory)))
            _, quality, _ = horizontal_systems(
                entries,
                n_requests=2,
                lambdas=(0.0,),
                shapes=["p3c2"] * 2,
                request_hashes=["c", "a"],
            )
        # Requested in the order c, a -> qualities 1.3 and 1.2, negated.
        self.assertAlmostEqual(quality[0, 0], -1.3, places=6)
        self.assertAlmostEqual(quality[1, 0], -1.2, places=6)

    def test_an_unscored_request_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entries = load_manifest(self.build(Path(directory)))
            with self.assertRaisesRegex(ValueError, "did not score"):
                horizontal_systems(
                    entries,
                    n_requests=2,
                    lambdas=(0.0,),
                    shapes=["p3c2"] * 2,
                    request_hashes=["a", "unknown-digest"],
                )

    def test_a_count_mismatch_without_hashes_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.build(Path(directory))
            entries = json.loads(path.read_text())
            for entry in entries:
                entry.pop("request_hashes")
            path.write_text(json.dumps(entries))
            loaded = load_manifest(path)
            with self.assertRaisesRegex(ValueError, "same requests"):
                horizontal_systems(loaded, n_requests=9, lambdas=(0.0,))


if __name__ == "__main__":
    unittest.main()
