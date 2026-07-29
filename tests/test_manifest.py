"""Tests for building a horizontal manifest from trajectory collections."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.build_manifest import build
from experiments.evaluate_vertical_routing import load_manifest
from utils.provenance import RunRecord


def write_trajectories(
    directory: Path,
    request_ids: list[int],
    tiers: list[int],
    accuracy: dict[int, list[float]],
    splits: list[str] | None = None,
    vocab_size: int = 50304,
) -> Path:
    """Writes a minimal trajectory directory of the shape ``load`` expects.

    Args:
        directory: Destination, created if absent.
        request_ids: Request identifier per row.
        tiers: Candidate depths recorded on every row.
        accuracy: Teacher-forced accuracy per request id, one value per tier.
        splits: Split per row; all ``"validation"`` when omitted.
        vocab_size: Recorded in ``model_config``, the tokenizer surrogate.

    Returns:
        The directory written to.
    """
    directory.mkdir(parents=True, exist_ok=True)
    splits = splits or ["validation"] * len(request_ids)

    with (directory / "trajectories.jsonl").open("w") as handle:
        for row, (request_id, split) in enumerate(zip(request_ids, splits)):
            handle.write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "source_id": request_id % 3,
                        "split": split,
                        "difficulty": "easy",
                        "prompt_len": 8,
                        "continuation_len": 4,
                        "tiers": tiers,
                        "teacher_forced_nll": [1.0] * len(tiers),
                        "teacher_forced_accuracy": accuracy[request_id],
                        "teacher_forced_top1_agreement": [1.0] * len(tiers),
                        "final_nll": 1.0,
                        "cost_macs": [100.0 * t for t in tiers],
                        "cost_depth_fraction": [t / tiers[-1] for t in tiers],
                        "kv_bytes": [64 * t for t in tiers],
                    }
                )
                + "\n"
            )

    np.savez_compressed(
        directory / "features.npz",
        features=np.zeros((len(request_ids), 4), dtype=np.float32),
    )
    RunRecord.create(script="test").write(
        directory / "run.json",
        payload={
            "metadata": {
                "schema_version": 1,
                "tiers": tiers,
                "model_config": {"vocab_size": vocab_size},
            }
        },
    )
    return directory


class BuildManifest(unittest.TestCase):
    """The happy path and the schema the evaluation expects back."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

        self.vertical = write_trajectories(
            self.root / "vertical",
            request_ids=[0, 1, 2, 3],
            tiers=[4, 8, 12],
            accuracy={
                0: [0.1, 0.3, 0.5],
                1: [0.2, 0.4, 0.6],
                2: [0.0, 0.2, 0.4],
                3: [0.3, 0.5, 0.7],
            },
        )
        self.independent = write_trajectories(
            self.root / "noexits",
            request_ids=[0, 1, 2, 3],
            tiers=[12],
            accuracy={0: [0.55], 1: [0.62], 2: [0.41], 3: [0.74]},
        )

    def test_manifest_loads_back(self) -> None:
        path = build(
            self.vertical,
            {"noexits": str(self.independent)},
            self.root / "horizontal",
        )
        entries = load_manifest(path)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["model_id"], "noexits")
        self.assertEqual(entries[0]["tier"], 12)

    def test_quality_is_copied_not_recomputed(self) -> None:
        build(
            self.vertical,
            {"noexits": str(self.independent)},
            self.root / "horizontal",
        )
        values = json.loads((self.root / "horizontal" / "noexits.json").read_text())
        self.assertEqual(values, [0.55, 0.62, 0.41, 0.74])

    def test_cost_is_the_independent_models_own(self) -> None:
        path = build(
            self.vertical,
            {"noexits": str(self.independent)},
            self.root / "horizontal",
        )
        entry = json.loads(path.read_text())[0]
        self.assertAlmostEqual(entry["cost"], 1200.0)

    def test_deepest_tier_is_the_default(self) -> None:
        """A collection with several tiers stands at its deepest."""
        path = build(
            self.vertical,
            {"shared_as_independent": str(self.vertical)},
            self.root / "horizontal",
        )
        self.assertEqual(json.loads(path.read_text())[0]["tier"], 12)

    def test_explicit_tier_selects_its_column(self) -> None:
        path = build(
            self.vertical,
            {"mid": str(self.vertical)},
            self.root / "horizontal",
            tiers={"mid": 8},
        )
        entry = json.loads(path.read_text())[0]
        self.assertEqual(entry["tier"], 8)
        values = json.loads(Path(entry["results"]).read_text())
        self.assertEqual(values, [0.3, 0.4, 0.2, 0.5])

    def test_only_validation_rows_are_exported(self) -> None:
        mixed = write_trajectories(
            self.root / "mixed",
            request_ids=[0, 1, 2, 3, 4],
            tiers=[12],
            accuracy={i: [0.5] for i in range(5)},
            splits=["train", "validation", "validation", "validation", "validation"],
        )
        # The vertical side must be filtered the same way, or the counts differ.
        vertical = write_trajectories(
            self.root / "vertical_mixed",
            request_ids=[0, 1, 2, 3, 4],
            tiers=[4, 12],
            accuracy={i: [0.1, 0.5] for i in range(5)},
            splits=["train", "validation", "validation", "validation", "validation"],
        )
        path = build(vertical, {"m": str(mixed)}, self.root / "horizontal")
        self.assertEqual(json.loads(path.read_text())[0]["n_requests"], 4)


class Alignment(unittest.TestCase):
    """The guard the module exists for.

    A paired bootstrap over misaligned rows still produces a confident-looking
    interval, so the mismatch has to be refused rather than reported.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.vertical = write_trajectories(
            self.root / "vertical",
            request_ids=[0, 1, 2],
            tiers=[6, 12],
            accuracy={i: [0.2, 0.5] for i in range(3)},
        )

    def test_different_request_count_is_refused(self) -> None:
        other = write_trajectories(
            self.root / "short",
            request_ids=[0, 1],
            tiers=[12],
            accuracy={0: [0.5], 1: [0.6]},
        )
        with self.assertRaises(ValueError) as caught:
            build(self.vertical, {"m": str(other)}, self.root / "out")
        self.assertIn("same requests", str(caught.exception))

    def test_reordered_requests_are_refused(self) -> None:
        """Same count, same ids, different order -- the silent case."""
        other = write_trajectories(
            self.root / "shuffled",
            request_ids=[2, 0, 1],
            tiers=[12],
            accuracy={0: [0.5], 1: [0.6], 2: [0.7]},
        )
        with self.assertRaises(ValueError) as caught:
            build(self.vertical, {"m": str(other)}, self.root / "out")
        self.assertIn("validation row 0", str(caught.exception))

    def test_mixed_vocabularies_are_refused(self) -> None:
        other = write_trajectories(
            self.root / "other_vocab",
            request_ids=[0, 1, 2],
            tiers=[12],
            accuracy={i: [0.5] for i in range(3)},
            vocab_size=32000,
        )
        with self.assertRaises(ValueError) as caught:
            build(self.vertical, {"m": str(other)}, self.root / "out")
        self.assertIn("tokenizations", str(caught.exception))

    def test_declared_tokenizer_overrides_the_surrogate(self) -> None:
        """Vocabulary size is a surrogate, so it must be overridable."""
        other = write_trajectories(
            self.root / "other_vocab",
            request_ids=[0, 1, 2],
            tiers=[12],
            accuracy={i: [0.5] for i in range(3)},
            vocab_size=32000,
        )
        path = build(
            self.vertical,
            {"m": str(other)},
            self.root / "out",
            tokenizer_id="gpt2",
        )
        self.assertEqual(json.loads(path.read_text())[0]["tokenizer_id"], "gpt2")

    def test_no_validation_rows_is_refused(self) -> None:
        other = write_trajectories(
            self.root / "train_only",
            request_ids=[0, 1, 2],
            tiers=[12],
            accuracy={i: [0.5] for i in range(3)},
            splits=["train"] * 3,
        )
        with self.assertRaises(ValueError) as caught:
            build(self.vertical, {"m": str(other)}, self.root / "out")
        self.assertIn("no validation requests", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
