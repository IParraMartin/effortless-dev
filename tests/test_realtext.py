"""Tests for real-text collection, clustered intervals, and the no-regret test.

These cover the machinery that turns a checkpoint into evidence about language
rather than about a synthetic token pattern. Four properties are load-bearing and
each has a way of failing silently:

* a request must not straddle a document boundary, or its continuation is
  unrelated to its prompt and every endpoint is depressed equally;
* a batch must never contain padding, because the model's attention is
  causal-only and real positions would attend to pad positions;
* a corpus NLL is a ratio of totals, so averaging per-request means is wrong
  whenever requests differ in length — and the error is the same order as the
  effects being measured;
* an interval must resample documents, because requests from one document are
  correlated and an unclustered interval is too narrow, which for a one-sided
  non-inferiority test biases toward *passing*.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.collect_depth_trajectories import (
    CORPORA,
    SCHEMA_VERSION,
    check_corpus_compatible,
    parse_shapes,
    tier_costs,
)
from experiments.evaluate_vertical_routing import request_clusters
from experiments.no_regret import compare, test_preservation
from experiments.workloads import (
    DEFAULT_SHAPES,
    RequestShape,
    real_text_corpus,
)
from src.config import TransformerConfig
from src.model import Transformer
from src.routing import DepthController
from utils.statistics import paired_bootstrap

EOS = 63
VOCAB = 64


def write_corpus(
    directory: Path,
    documents: int = 30,
    low: int = 200,
    high: int = 700,
    seed: int = 0,
    tokenizer_size: int = VOCAB,
) -> Path:
    """Writes an EOS-separated token file with a sidecar.

    Args:
        directory: Where to write.
        documents: Number of documents.
        low: Shortest document.
        high: Longest document.
        seed: Seed for the token values.
        tokenizer_size: Recorded vocabulary size.

    Returns:
        Path to the written ``.bin``.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(documents):
        length = int(rng.integers(low, high))
        parts.append(rng.integers(0, EOS, size=length, dtype=np.uint16))
        parts.append(np.array([EOS], dtype=np.uint16))
    tokens = np.concatenate(parts)

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "val.bin"
    tokens.tofile(path)
    Path(str(path) + ".meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dtype": "uint16",
                "n_tokens": int(tokens.size),
                "tokenizer_name": "fixture",
                "tokenizer_size": tokenizer_size,
                "dataset_name": "fixture-corpus",
                "dataset_config": None,
                "split": "validation",
            }
        )
    )
    return path


def model_config(**updates) -> TransformerConfig:
    """Builds a small architecture sized for the fixture corpus."""
    values = dict(
        vocab_size=VOCAB, d_model=32, n_layers=6, n_heads=4, n_kv_heads=2,
        ff_dim=64, max_seq_len=128, exit_every=2, min_exit_layer=1,
    )
    values.update(updates)
    return TransformerConfig(**values)


class RealTextCorpus(unittest.TestCase):
    """Requests drawn from a tokenized corpus."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.path = write_corpus(Path(self._temporary.name))

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def draw(self, **kwargs):
        settings = dict(
            shapes=(RequestShape(32, 16), RequestShape(48, 24)),
            n_requests=40,
            eos_id=EOS,
            seed=1,
        )
        settings.update(kwargs)
        return real_text_corpus(self.path, **settings)

    def test_no_request_straddles_a_document_boundary(self) -> None:
        """Otherwise the continuation is unrelated to the prompt."""
        buckets, _ = self.draw()
        for bucket in buckets:
            with self.subTest(shape=bucket.spec["shape"]):
                self.assertFalse(
                    bool((bucket.sequences() == EOS).any()),
                    "a request spans an end-of-text token",
                )

    def test_every_batch_is_rectangular_without_padding(self) -> None:
        """The model's attention has no padding mask; a pad token would corrupt."""
        buckets, _ = self.draw()
        for bucket in buckets:
            with self.subTest(shape=bucket.spec["shape"]):
                self.assertEqual(
                    bucket.prompts.shape[1], bucket.spec["prompt_len"]
                )
                self.assertEqual(
                    bucket.references.shape[1], bucket.spec["continuation_len"]
                )

    def test_shapes_are_separated_into_their_own_buckets(self) -> None:
        buckets, metadata = self.draw()
        self.assertEqual(len(buckets), 2)
        self.assertEqual(
            sorted(bucket.spec["shape"] for bucket in buckets),
            ["p32c16", "p48c24"],
        )
        self.assertEqual(metadata["requests_drawn"], 40)

    def test_requests_are_spread_evenly_across_shapes(self) -> None:
        buckets, _ = self.draw(n_requests=40)
        self.assertEqual({len(bucket) for bucket in buckets}, {20})

    def test_document_identity_is_recorded(self) -> None:
        buckets, metadata = self.draw()
        self.assertEqual(metadata["documents_found"], 30)
        for bucket in buckets:
            with self.subTest(shape=bucket.spec["shape"]):
                self.assertIsNotNone(bucket.document_ids)
                self.assertEqual(len(bucket.document_ids), len(bucket))
                self.assertEqual(bucket.source_ids, bucket.document_ids)

    def test_several_requests_share_a_document(self) -> None:
        """The reason clustering is needed at all."""
        buckets, _ = self.draw(n_requests=60)
        documents = [d for bucket in buckets for d in bucket.document_ids]
        self.assertLess(
            len(set(documents)),
            len(documents),
            "no document was sampled twice, so this fixture cannot exercise "
            "clustering",
        )

    def test_token_hashes_identify_content(self) -> None:
        buckets, _ = self.draw()
        hashes = [h for bucket in buckets for h in bucket.token_hashes]
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertTrue(all(len(h) == 32 for h in hashes))

    def test_the_draw_is_reproducible_from_the_seed(self) -> None:
        left, _ = self.draw(seed=7)
        right, _ = self.draw(seed=7)
        for a, b in zip(left, right):
            self.assertTrue(torch.equal(a.prompts, b.prompts))
        other, _ = self.draw(seed=8)
        self.assertFalse(torch.equal(left[0].prompts, other[0].prompts))

    def test_offsets_and_domains_are_carried(self) -> None:
        buckets, _ = self.draw()
        for bucket in buckets:
            with self.subTest(shape=bucket.spec["shape"]):
                self.assertTrue(all(o >= 0 for o in bucket.offsets))
                self.assertEqual(set(bucket.domains), {"fixture-corpus"})

    def test_select_carries_the_new_metadata(self) -> None:
        buckets, _ = self.draw()
        subset = buckets[0].select([0, 2])
        self.assertEqual(len(subset.document_ids), 2)
        self.assertEqual(len(subset.token_hashes), 2)
        self.assertEqual(subset.document_ids, [buckets[0].document_ids[i] for i in (0, 2)])

    def test_absent_eos_is_reported_as_degenerate_clustering(self) -> None:
        """Silence here would produce intervals nobody knew were unclustered."""
        _, metadata = self.draw(eos_id=None)
        self.assertEqual(metadata["documents_found"], 1)
        self.assertIn("clustering_note", metadata)
        self.assertIn("too narrow", metadata["clustering_note"])

    def test_truncation_is_reported(self) -> None:
        _, metadata = self.draw(
            shapes=(RequestShape(32, 16), RequestShape(600, 300)), n_requests=20
        )
        self.assertIn("truncation_note", metadata)
        self.assertLess(metadata["requests_drawn"], 20)

    def test_a_corpus_with_no_long_enough_document_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no document"):
            self.draw(shapes=(RequestShape(900, 400),))

    def test_a_missing_corpus_is_reported(self) -> None:
        with self.assertRaises(FileNotFoundError):
            real_text_corpus("nowhere/val.bin")

    def test_non_positive_request_counts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_requests"):
            self.draw(n_requests=0)

    def test_the_metadata_names_the_corpus_and_tokenizer(self) -> None:
        _, metadata = self.draw()
        self.assertEqual(metadata["corpus"], "real_text")
        self.assertEqual(metadata["dataset_name"], "fixture-corpus")
        self.assertEqual(metadata["tokenizer_name"], "fixture")
        self.assertEqual(metadata["eos_id"], EOS)


class ShapeParsing(unittest.TestCase):
    """Shape specifications, and the ways they fail."""

    def test_valid_specifications_parse(self) -> None:
        self.assertEqual(
            parse_shapes(("64:32", "128:64")),
            (RequestShape(64, 32), RequestShape(128, 64)),
        )

    def test_the_defaults_differ_materially_in_cost(self) -> None:
        totals = [shape.total for shape in DEFAULT_SHAPES]
        self.assertEqual(len(set(totals)), len(totals))
        self.assertGreater(max(totals) / min(totals), 2.0)

    def test_a_malformed_specification_is_rejected(self) -> None:
        for bad in ("64", "64:32:16", "a:b", ""):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "prompt:continuation"):
                    parse_shapes((bad,))

    def test_a_zero_length_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_shapes(("64:0",))

    def test_an_empty_list_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one shape"):
            parse_shapes(())

    def test_the_corpora_are_named(self) -> None:
        self.assertEqual(set(CORPORA), {"synthetic", "real_text"})


class Preflight(unittest.TestCase):
    """Incompatibilities must fail at startup, not mid-collection."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.path = write_corpus(Path(self._temporary.name))

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_a_request_longer_than_the_context_is_rejected(self) -> None:
        buckets, metadata = real_text_corpus(
            self.path, shapes=(RequestShape(100, 60),), n_requests=4, eos_id=EOS
        )
        with self.assertRaisesRegex(ValueError, "exceed the model's context"):
            check_corpus_compatible(
                model_config(max_seq_len=64),
                [("report", bucket) for bucket in buckets],
                metadata,
            )

    def test_a_larger_tokenizer_than_the_model_is_rejected(self) -> None:
        buckets, metadata = real_text_corpus(
            self.path, shapes=(RequestShape(32, 16),), n_requests=4, eos_id=EOS
        )
        metadata["tokenizer_size"] = 50257
        with self.assertRaisesRegex(ValueError, "tokenized with a vocabulary"):
            check_corpus_compatible(
                model_config(),
                [("report", bucket) for bucket in buckets],
                metadata,
            )

    def test_an_out_of_range_token_is_rejected(self) -> None:
        buckets, metadata = real_text_corpus(
            self.path, shapes=(RequestShape(32, 16),), n_requests=4, eos_id=EOS
        )
        buckets[0].prompts[0, 0] = 200
        metadata.pop("tokenizer_size")
        with self.assertRaisesRegex(ValueError, "outside the model's"):
            check_corpus_compatible(
                model_config(),
                [("report", bucket) for bucket in buckets],
                metadata,
            )

    def test_a_compatible_corpus_passes(self) -> None:
        buckets, metadata = real_text_corpus(
            self.path, shapes=(RequestShape(32, 16),), n_requests=4, eos_id=EOS
        )
        check_corpus_compatible(
            model_config(), [("report", b) for b in buckets], metadata
        )


class ClusteredIntervals(unittest.TestCase):
    """Resampling documents rather than requests."""

    def correlated(self, clusters: int = 20, per_cluster: int = 5, seed: int = 0):
        """Builds observations that are identical within a cluster.

        Returns:
            A tuple ``(values, labels)``. Within-cluster correlation is total, so
            the effective sample size is the number of clusters, not observations.
        """
        rng = np.random.default_rng(seed)
        effects = rng.normal(0.0, 1.0, size=clusters)
        values = np.repeat(effects, per_cluster)
        labels = np.repeat(np.arange(clusters), per_cluster)
        return values, labels

    def test_clustering_widens_the_interval_on_correlated_data(self) -> None:
        """The whole point: an unclustered interval here is too narrow."""
        values, labels = self.correlated()
        naive = paired_bootstrap(values, resamples=800, seed=0)
        clustered = paired_bootstrap(
            values, resamples=800, seed=0, clusters=labels
        )

        naive_width = naive.high - naive.low
        clustered_width = clustered.high - clustered.low
        self.assertGreater(
            clustered_width,
            naive_width * 1.5,
            f"clustered width {clustered_width:.4f} should greatly exceed the "
            f"unclustered {naive_width:.4f} when observations repeat within a "
            f"cluster",
        )

    def test_the_estimate_is_unchanged(self) -> None:
        values, labels = self.correlated()
        self.assertAlmostEqual(
            paired_bootstrap(values, resamples=200, seed=0).estimate,
            paired_bootstrap(
                values, resamples=200, seed=0, clusters=labels
            ).estimate,
            places=12,
        )

    def test_singleton_clusters_match_the_unclustered_estimate(self) -> None:
        rng = np.random.default_rng(3)
        values = rng.normal(size=200)
        labels = np.arange(200)
        naive = paired_bootstrap(values, resamples=600, seed=1)
        clustered = paired_bootstrap(values, resamples=600, seed=1, clusters=labels)

        self.assertAlmostEqual(naive.estimate, clustered.estimate, places=12)
        self.assertAlmostEqual(
            naive.high - naive.low, clustered.high - clustered.low, delta=0.05
        )

    def test_the_interval_records_its_resampling_unit(self) -> None:
        values, labels = self.correlated()
        clustered = paired_bootstrap(values, resamples=100, seed=0, clusters=labels)
        naive = paired_bootstrap(values, resamples=100, seed=0)

        self.assertTrue(clustered.clustered)
        self.assertEqual(clustered.n_clusters, 20)
        self.assertIn("20 clusters", str(clustered))
        self.assertFalse(naive.clustered)
        self.assertIsNone(naive.n_clusters)

    def test_mismatched_labels_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "every observation needs"):
            paired_bootstrap([1.0, 2.0, 3.0], clusters=[0, 1])

    def test_schema_one_records_report_no_clusters(self) -> None:
        """Silently substituting request_id would mislabel the interval."""
        self.assertIsNone(request_clusters([{"request_id": 0}, {"request_id": 1}]))
        self.assertIsNone(request_clusters([]))

    def test_schema_two_records_yield_document_clusters(self) -> None:
        clusters = request_clusters(
            [{"document_id": 4}, {"document_id": 4}, {"document_id": 9}]
        )
        self.assertIsNotNone(clusters)
        self.assertEqual(clusters.tolist(), [4, 4, 9])


class CorpusAggregation(unittest.TestCase):
    """A corpus NLL is a ratio of totals."""

    def test_mean_of_means_differs_from_the_corpus_value(self) -> None:
        # Two requests: 8 tokens at NLL 5.0, 32 tokens at NLL 1.0.
        sums = np.array([40.0, 32.0])
        counts = np.array([8.0, 32.0])
        mean_of_means = float(np.mean(sums / counts))
        corpus = float(sums.sum() / counts.sum())

        self.assertAlmostEqual(mean_of_means, 3.0, places=12)
        self.assertAlmostEqual(corpus, 1.8, places=12)
        self.assertGreater(abs(mean_of_means - corpus), 1.0)

    def test_the_record_schema_carries_sums_and_counts(self) -> None:
        from experiments.collect_depth_trajectories import RequestRecord

        record = RequestRecord(
            request_id=0, source_id=0, split="validation", difficulty="unknown",
            prompt_len=32, continuation_len=16, tiers=[2, 4],
            teacher_forced_nll=[1.0, 0.9],
            teacher_forced_accuracy=[0.1, 0.2],
            teacher_forced_top1_agreement=[0.5, 1.0],
        )
        for name in (
            "teacher_forced_nll_sum", "teacher_forced_top1_count", "valid_tokens",
            "document_id", "domain", "token_hash", "corpus_offset", "shape",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(record, name))

    def test_the_schema_version_advanced(self) -> None:
        self.assertGreaterEqual(SCHEMA_VERSION, 2)


class RetrofitCostAccounting(unittest.TestCase):
    """Modules a retrofit adds are part of the endpoint's cost."""

    def setUp(self) -> None:
        self.base = TransformerConfig(
            vocab_size=52000, d_model=768, n_layers=12, n_heads=12, max_seq_len=2048
        )

    def test_an_exit_adapter_is_charged(self) -> None:
        from dataclasses import replace

        plain = tier_costs(self.base, (2, 12), 128, 64)
        adapted = tier_costs(
            replace(self.base, exit_adapter_rank=32), (2, 12), 128, 64
        )
        for index in range(2):
            with self.subTest(tier=index):
                self.assertGreater(adapted["macs"][index], plain["macs"][index])

    def test_the_adapter_charge_matches_the_arithmetic(self) -> None:
        from dataclasses import replace

        from utils.costs import AnalyticalCostModel

        plain = tier_costs(self.base, (12,), 128, 64)
        adapted = tier_costs(
            replace(self.base, exit_adapter_rank=32), (12,), 128, 64
        )
        model = AnalyticalCostModel.from_config(self.base)
        # One adapter per generated token, alongside the vocabulary head.
        self.assertAlmostEqual(
            adapted["macs"][0] - plain["macs"][0],
            64 * model.exit_adapter_macs(32),
            places=3,
        )

    def test_lora_is_charged_per_executed_block(self) -> None:
        plain = tier_costs(self.base, (2, 12), 128, 64)
        wrapped = tier_costs(
            self.base, (2, 12), 128, 64, lora_rank=8, lora_targets_per_block=4
        )
        shallow = wrapped["macs"][0] / plain["macs"][0] - 1
        deep = wrapped["macs"][1] / plain["macs"][1] - 1

        self.assertGreater(shallow, 0.0)
        self.assertGreater(deep, shallow, "a per-block cost must grow with depth")

    def test_no_retrofit_means_no_extra_charge(self) -> None:
        left = tier_costs(self.base, (2, 12), 128, 64)
        right = tier_costs(self.base, (2, 12), 128, 64, lora_rank=0)
        self.assertEqual(left["macs"].tolist(), right["macs"].tolist())


class ControllerCostParity(unittest.TestCase):
    """A controller must not deploy against a cost it was not selected under."""

    def setUp(self) -> None:
        self.controller = DepthController(d_model=16, n_tiers=3, input_dim=8)
        self.features = torch.randn(2, 8)

    def test_a_recorded_metric_makes_the_fallback_an_error(self) -> None:
        self.controller.cost_metric = "cost_macs"
        with self.assertRaisesRegex(ValueError, "different objectives"):
            self.controller.select(self.features, (2, 4, 6))

    def test_supplying_the_cost_vector_is_accepted(self) -> None:
        self.controller.cost_metric = "cost_macs"
        self.controller.select(
            self.features, (2, 4, 6), tier_costs=torch.tensor([0.2, 0.6, 1.0])
        )

    def test_a_depth_fraction_controller_may_use_the_fallback(self) -> None:
        self.controller.cost_metric = "cost_depth_fraction"
        self.controller.select(self.features, (2, 4, 6))

    def test_an_unrecorded_metric_keeps_the_old_behaviour(self) -> None:
        self.assertIsNone(self.controller.cost_metric)
        self.controller.select(self.features, (2, 4, 6))


class NoRegretTest(unittest.TestCase):
    """The instrument H1 needs for every mode above the frozen ones."""

    def paired(
        self,
        gap: float,
        clusters: int = 30,
        per_cluster: int = 2,
        seed=0,
        gap_spread: float = 0.0,
    ):
        """Builds a paired comparison with a known endpoint gap in nats.

        Args:
            gap: Mean NLL the candidate loses.
            clusters: Documents.
            per_cluster: Requests per document.
            seed: Seed.
            gap_spread: Per-document standard deviation of the gap. Zero makes
                the difference constant and the interval degenerate, which is
                useful for checking the estimate but cannot exercise a verdict
                that depends on interval *width*.
        """
        rng = np.random.default_rng(seed)
        n = clusters * per_cluster
        parent = rng.normal(3.2, 0.05, size=n)
        offsets = (
            np.repeat(rng.normal(0.0, gap_spread, size=clusters), per_cluster)
            if gap_spread
            else 0.0
        )
        return {
            "parent_nll": parent,
            "candidate_nll": parent + gap + offsets,
            "documents": np.repeat(np.arange(clusters), per_cluster),
            "shapes": ["p32c16"] * n,
            "valid_tokens": np.full(n, 16.0),
            "parent_corpus_nll": float(parent.mean()),
            "candidate_corpus_nll": float((parent + gap + offsets).mean()),
        }

    def test_an_identical_candidate_passes(self) -> None:
        result = test_preservation(self.paired(0.0), 0.01, 400, 0)
        self.assertTrue(result["passes"])
        self.assertAlmostEqual(result["nll_difference"]["estimate"], 0.0, places=9)
        self.assertAlmostEqual(result["perplexity_ratio"], 1.0, places=6)

    def test_a_regression_beyond_the_margin_fails(self) -> None:
        """The failure path, which the toy checkpoint could not express."""
        result = test_preservation(self.paired(0.05), 0.01, 400, 0)
        self.assertFalse(result["passes"])
        self.assertAlmostEqual(result["nll_difference"]["estimate"], 0.05, places=6)

    def test_a_regression_inside_the_margin_passes(self) -> None:
        result = test_preservation(self.paired(0.002), 0.01, 400, 0)
        self.assertTrue(result["passes"])

    def test_an_improvement_passes(self) -> None:
        result = test_preservation(self.paired(-0.02), 0.01, 400, 0)
        self.assertTrue(result["passes"])
        self.assertLess(result["nll_difference"]["estimate"], 0.0)

    def test_the_verdict_is_one_sided_not_an_overlap_check(self) -> None:
        """An interval containing zero must not by itself support preservation.

        This is the case that distinguishes the two rules. The estimate is
        centred near zero but the spread is wide, so the interval straddles zero
        — an "is there a detected regression?" reading would pass it — while the
        lower bound sits well below the margin, so non-inferiority is not
        supported. An underpowered study must not be able to conclude
        preservation.
        """
        paired = self.paired(0.0, gap_spread=0.08, seed=4)
        result = test_preservation(paired, 0.01, 600, 0)

        interval = result["nll_difference"]
        self.assertLess(interval["low"], 0.0)
        self.assertGreater(interval["high"], 0.0)
        self.assertFalse(
            result["passes"],
            "an interval spanning zero was treated as evidence of preservation",
        )

    def test_the_resampling_unit_is_the_document(self) -> None:
        result = test_preservation(self.paired(0.0, clusters=30, per_cluster=2), 0.01, 200, 0)
        self.assertEqual(result["n_documents"], 30)
        self.assertEqual(result["n_requests"], 60)
        self.assertTrue(result["nll_difference"]["clustered"])

    def test_the_margin_is_reported_with_the_verdict(self) -> None:
        result = test_preservation(self.paired(0.0), 0.0123, 200, 0)
        self.assertEqual(result["quality_margin"], 0.0123)

    def test_compare_scores_both_models_on_the_same_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_corpus(Path(directory))
            buckets, _ = real_text_corpus(
                path,
                shapes=(RequestShape(32, 16), RequestShape(48, 24)),
                n_requests=20,
                eos_id=EOS,
                seed=2,
            )
            torch.manual_seed(0)
            parent = Transformer(model_config()).eval()
            paired = compare(parent, parent, [("report", b) for b in buckets])

        self.assertEqual(paired["parent_nll"].shape, paired["candidate_nll"].shape)
        self.assertEqual(len(paired["documents"]), len(paired["parent_nll"]))
        # Same model twice, so the endpoint is preserved exactly.
        self.assertTrue(
            np.array_equal(paired["parent_nll"], paired["candidate_nll"])
        )
        self.assertEqual(len(set(paired["shapes"])), 2)

    def test_corpus_nll_uses_summed_totals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_corpus(Path(directory))
            buckets, _ = real_text_corpus(
                path,
                shapes=(RequestShape(32, 16), RequestShape(48, 24)),
                n_requests=20,
                eos_id=EOS,
                seed=2,
            )
            torch.manual_seed(0)
            parent = Transformer(model_config()).eval()
            paired = compare(parent, parent, [("report", b) for b in buckets])

        expected = float(
            (paired["parent_nll"] * paired["valid_tokens"]).sum()
            / paired["valid_tokens"].sum()
        )
        self.assertAlmostEqual(paired["parent_corpus_nll"], expected, places=5)
        # The shapes differ in length, so the naive average is a different number.
        self.assertNotAlmostEqual(
            paired["parent_corpus_nll"], float(paired["parent_nll"].mean()), places=9
        )


class LoRACheckpointRoundTrip(unittest.TestCase):
    """Wrapping a projection renames its weight; the loader must replay that."""

    def test_a_lora_checkpoint_reloads(self) -> None:
        from dataclasses import replace

        from src.retrofit import RetrofitConfig, restore, retrofit

        torch.manual_seed(0)
        parent = Transformer(model_config(exit_every=6)).eval().requires_grad_(False)
        settings = RetrofitConfig(mode="lora", lora_rank=4)
        model, _ = retrofit(
            parent, settings, model_config=replace(parent.config, exit_every=2)
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retrofit.pt"
            from experiments.retrofit_parent import save

            save(path, model, model.config, settings, "parent.pt")
            reloaded = restore(path)

        ids = torch.randint(0, VOCAB, (2, 24))
        with torch.no_grad():
            self.assertTrue(
                torch.equal(reloaded(ids).logits, model(ids).logits),
                "the reloaded LoRA model does not reproduce the saved one",
            )

    def test_a_plain_checkpoint_still_reloads(self) -> None:
        from src.config import TrainConfig
        from src.retrofit import restore
        from training.train import build_optimizer, save_checkpoint

        torch.manual_seed(0)
        model = Transformer(model_config())
        train_config = TrainConfig(dtype="fp32", compile_model=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.pt"
            save_checkpoint(
                path,
                model,
                build_optimizer(model, train_config, torch.device("cpu")),
                5,
                model.config,
                train_config,
            )
            reloaded = restore(path)

        ids = torch.randint(0, VOCAB, (1, 16))
        model.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(reloaded(ids).logits, model(ids).logits))

    def test_a_disagreeing_record_is_rejected(self) -> None:
        from src.retrofit import restore

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.pt"
            torch.save(
                {
                    "model_config": model_config(),
                    "model": {},
                    "retrofit": {"mode": "frozen_tied_head"},
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "after replaying"):
                restore(path)


if __name__ == "__main__":
    unittest.main()
