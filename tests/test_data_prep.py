"""Tests for corpus preparation: split resolution, holdout carving, and reuse.

These exist because of a failure that cost two hours of cluster time. FineWeb-Edu
ships only a ``train`` split; the streaming path trusted the requested split name
and died with ``Bad split: validation`` *after* writing the whole training file,
which was then thrown away on the retry. Both halves of that are covered here:
deriving a held-out set when the corpus has none, and reusing a split that is
already complete on disk.

The hub is stubbed rather than contacted. The logic under test is entirely about
which documents end up where, and a test that needs the network to prove it is a
test that does not get run.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.config import TrainConfig
from training import data as prep


class FakeStream:
    """An ``IterableDataset`` stand-in supporting the calls carving makes."""

    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs

    def skip(self, n: int) -> FakeStream:
        return self.__class__(self.docs[n:])

    def take(self, n: int) -> FakeStream:
        return self.__class__(self.docs[:n])

    def __iter__(self):
        return iter(self.docs)


class FakeDataset:
    """A map-style ``Dataset`` stand-in."""

    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs

    def __len__(self) -> int:
        return len(self.docs)

    def select(self, indices) -> FakeDataset:
        return self.__class__([self.docs[i] for i in indices])

    def __iter__(self):
        return iter(self.docs)


def corpus(n: int, streaming: bool = True):
    """Builds a stand-in corpus whose documents are identifiable by index."""
    docs = [{"text": f"doc-{i}"} for i in range(n)]
    return FakeStream(docs) if streaming else FakeDataset(docs)


def texts(dataset) -> list[str]:
    """Reads a stand-in corpus back as plain strings."""
    return [d["text"] for d in dataset]


def stub_hub(splits: list[str], docs: int = 100, streaming: bool = True):
    """Patches the two ``datasets`` entry points ``_load_split`` reaches for."""
    def load_dataset(*args, split=None, streaming=False, **kwargs):
        if splits is not None and split not in splits:
            raise ValueError(f"Bad split: {split}. Available splits: {splits}")
        return corpus(docs, streaming=streaming)

    def get_split_names(*args, **kwargs):
        if splits is None:
            raise ConnectionError("offline")
        return splits

    return mock.patch.multiple(
        "datasets",
        load_dataset=load_dataset,
        get_dataset_split_names=get_split_names,
    )


class SplitResolution(unittest.TestCase):
    """Which split gets opened, and whether one has to be carved."""

    def test_train_only_corpus_derives_validation(self) -> None:
        """The FineWeb-Edu case: no validation split, do not fail."""
        config = TrainConfig(dataset_name="org/corpus", streaming=True)
        with stub_hub(["train"]):
            _, derived = prep._load_split(config, "validation")
        self.assertTrue(derived)

    def test_real_validation_split_is_used_as_is(self) -> None:
        config = TrainConfig(dataset_name="org/corpus", streaming=True)
        with stub_hub(["train", "validation"]):
            _, derived = prep._load_split(config, "validation")
        self.assertFalse(derived)

    def test_test_split_substitutes_for_validation(self) -> None:
        config = TrainConfig(dataset_name="org/corpus", streaming=True)
        with stub_hub(["train", "test"]):
            _, derived = prep._load_split(config, "test")
        self.assertFalse(derived)

    def test_unprobeable_hub_trusts_the_requested_name(self) -> None:
        """Offline or private: fall through rather than guessing wrongly."""
        config = TrainConfig(dataset_name="org/corpus", streaming=True)
        with stub_hub(None):
            _, derived = prep._load_split(config, "train")
        self.assertFalse(derived)

    def test_corpus_without_any_usable_split_raises(self) -> None:
        config = TrainConfig(dataset_name="org/corpus", streaming=True)
        with stub_hub(["nonsense"]):
            with self.assertRaisesRegex(ValueError, "no 'train' to derive"):
                prep._load_split(config, "validation")

    def test_local_files_always_carve(self) -> None:
        config = TrainConfig(dataset_name="/tmp/corpus.jsonl", streaming=True)
        with stub_hub(["train"]):
            _, derived = prep._load_split(config, "validation")
        self.assertTrue(derived)


class Carving(unittest.TestCase):
    """Where the held-out documents come from, and that they are disjoint."""

    def test_capped_run_takes_validation_from_beyond_the_cap(self) -> None:
        config = TrainConfig(streaming=True, max_train_docs=50)
        train = texts(prep._carve(corpus(100), "train", config))
        val = texts(prep._carve(corpus(100), "validation", config))

        self.assertFalse(set(train[:50]) & set(val))
        self.assertEqual(val[0], "doc-50")

    def test_capped_run_leaves_the_training_stream_untouched(self) -> None:
        """So a train.bin written before the holdout existed stays valid."""
        config = TrainConfig(streaming=True, max_train_docs=50)
        carved = texts(prep._carve(corpus(100), "train", config))
        self.assertEqual(carved, texts(corpus(100)))

    def test_uncapped_run_takes_validation_off_the_front(self) -> None:
        size = prep.HELD_OUT_DOCS * 3
        config = TrainConfig(streaming=True, max_train_docs=None)
        train = texts(prep._carve(corpus(size), "train", config))
        val = texts(prep._carve(corpus(size), "validation", config))

        self.assertEqual(val[0], "doc-0")
        self.assertEqual(len(val), prep.HELD_OUT_DOCS)
        self.assertEqual(train[0], f"doc-{prep.HELD_OUT_DOCS}")
        self.assertFalse(set(train) & set(val))

    def test_uncapped_run_on_a_tiny_corpus_reports_the_holdout(self) -> None:
        """Skipping the holdout can consume a small corpus entirely."""
        config = TrainConfig(
            dataset_name="org/corpus", tokenizer_name="gpt2",
            streaming=True, max_train_docs=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            config.data_dir = tmp
            with stub_hub(["train"], docs=10):
                with self.assertRaisesRegex(ValueError, "held out"):
                    prep.prepare(config)

    def test_map_style_capped_is_disjoint(self) -> None:
        config = TrainConfig(streaming=False, max_train_docs=50)
        train = texts(prep._carve(corpus(100, False), "train", config))
        val = texts(prep._carve(corpus(100, False), "validation", config))
        self.assertFalse(set(train[:50]) & set(val))

    def test_map_style_cap_covering_everything_falls_back_to_the_front(self) -> None:
        """Nothing lies beyond a cap larger than the corpus."""
        config = TrainConfig(streaming=False, max_train_docs=1000)
        train = texts(prep._carve(corpus(100, False), "train", config))
        val = texts(prep._carve(corpus(100, False), "validation", config))

        self.assertTrue(val)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(val[0], "doc-0")

    def test_validation_is_bounded_by_held_out_docs(self) -> None:
        config = TrainConfig(streaming=True, max_train_docs=10)
        val = texts(prep._carve(corpus(100_000), "validation", config))
        self.assertEqual(len(val), prep.HELD_OUT_DOCS)


class Reuse(unittest.TestCase):
    """A finished split is not rebuilt; a mismatched one is."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.config = TrainConfig(
            dataset_name="org/corpus", dataset_config="c", text_column="text",
            tokenizer_name="gpt2", max_train_docs=50, data_dir=str(self.dir),
        )

    def _write(self, name: str, n_tokens: int = 8, **overrides) -> Path:
        path = self.dir / name
        path.write_bytes(np.zeros(n_tokens, dtype=np.uint16).tobytes())
        meta = {
            "schema_version": prep.META_SCHEMA_VERSION,
            "dtype": "uint16",
            "n_tokens": n_tokens,
            "tokenizer_name": "gpt2",
            "tokenizer_size": 50257,
            "dataset_name": "org/corpus",
            "dataset_config": "c",
            "text_column": "text",
            "max_train_docs": 50,
        }
        meta.update(overrides)
        Path(str(path) + prep.META_SUFFIX).write_text(json.dumps(meta))
        return path

    def test_matching_split_is_reusable(self) -> None:
        path = self._write("train.bin")
        self.assertEqual(prep._completed(path, self.config, 50257), 8)

    def test_missing_sidecar_means_unfinished(self) -> None:
        """A job killed mid-write leaves the .bin without its sidecar."""
        path = self.dir / "train.bin"
        path.write_bytes(np.zeros(8, dtype=np.uint16).tobytes())
        self.assertIsNone(prep._completed(path, self.config, 50257))

    def test_absent_file_is_not_reusable(self) -> None:
        self.assertIsNone(
            prep._completed(self.dir / "nothing.bin", self.config, 50257)
        )

    def test_different_corpus_is_not_reused(self) -> None:
        path = self._write("train.bin", dataset_name="other/corpus")
        self.assertIsNone(prep._completed(path, self.config, 50257))

    def test_different_tokenizer_is_not_reused(self) -> None:
        """Mixing two tokenizations would train without error and mean nothing."""
        path = self._write("train.bin", tokenizer_name="EleutherAI/gpt-neox-20b")
        self.assertIsNone(prep._completed(path, self.config, 50257))

    def test_different_document_cap_is_not_reused(self) -> None:
        path = self._write("train.bin", max_train_docs=999)
        self.assertIsNone(prep._completed(path, self.config, 50257))

    def test_resized_tokenizer_is_not_reused(self) -> None:
        path = self._write("train.bin")
        self.assertIsNone(prep._completed(path, self.config, 50432))

    def test_truncated_file_is_not_reused(self) -> None:
        """The sidecar says 8 tokens; the file holds 4."""
        path = self._write("train.bin", n_tokens=8)
        path.write_bytes(np.zeros(4, dtype=np.uint16).tobytes())
        self.assertIsNone(prep._completed(path, self.config, 50257))

    def test_unreadable_sidecar_is_not_reused(self) -> None:
        path = self._write("train.bin")
        Path(str(path) + prep.META_SUFFIX).write_text("{not json")
        self.assertIsNone(prep._completed(path, self.config, 50257))


class EndToEnd(unittest.TestCase):
    """`prepare` over a stubbed train-only corpus, which is the failing case."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _config(self, **overrides) -> TrainConfig:
        values = dict(
            dataset_name="org/corpus", dataset_config="c", text_column="text",
            tokenizer_name="gpt2", streaming=True, max_train_docs=40,
            data_dir=str(self.dir),
        )
        values.update(overrides)
        return TrainConfig(**values)

    def test_train_only_corpus_produces_both_files(self) -> None:
        with stub_hub(["train"], docs=60):
            prep.prepare(self._config())

        for name in ("train.bin", "val.bin"):
            self.assertTrue((self.dir / name).exists(), name)
            self.assertTrue((self.dir / f"{name}{prep.META_SUFFIX}").exists())

        meta = json.loads(
            (self.dir / f"val.bin{prep.META_SUFFIX}").read_text()
        )
        self.assertTrue(meta["derived_from_train"])

    def test_second_run_reuses_both_splits(self) -> None:
        config = self._config()
        with stub_hub(["train"], docs=60):
            prep.prepare(config)
        first = (self.dir / "train.bin").stat().st_mtime_ns

        # A corpus that would raise if opened proves nothing was re-read.
        with mock.patch(
            "datasets.load_dataset",
            side_effect=AssertionError("should not re-tokenize"),
        ):
            prep.prepare(config)

        self.assertEqual((self.dir / "train.bin").stat().st_mtime_ns, first)

    def test_overwrite_forces_a_rebuild(self) -> None:
        config = self._config()
        with stub_hub(["train"], docs=60):
            prep.prepare(config)
            prep.prepare(self._config(overwrite_data=True))
        self.assertTrue((self.dir / "train.bin").exists())

    def test_exhausted_corpus_names_the_cap(self) -> None:
        """A cap past the end of the corpus leaves nothing to hold out."""
        with stub_hub(["train"], docs=20):
            with self.assertRaisesRegex(ValueError, "max_train_docs"):
                prep.prepare(self._config(max_train_docs=40))


class NoLeak(unittest.TestCase):
    """Training and validation must never share a document.

    This is the regression test for a leak introduced while fixing the missing
    split: the carve decision was made per split, so with no document cap the
    training stream kept everything while validation took the first two
    thousand documents. Both files were written, nothing raised, and the eval
    loss would have been measured on data the model had trained on.
    """

    def _consumed(self, config: TrainConfig) -> dict[str, set[str]]:
        """Runs prepare and records which documents each split actually read."""
        seen: dict[str, set[str]] = {}
        real_load_split = prep._load_split

        def recording(cfg, split, carve=None):
            dataset, derived = real_load_split(cfg, split, carve)
            captured = seen.setdefault(split, set())

            class Recording(FakeStream):
                def __iter__(inner):
                    for doc in inner.docs:
                        captured.add(doc["text"])
                        yield doc

            return Recording(dataset.docs), derived

        with mock.patch.multiple(
            "datasets",
            load_dataset=lambda *a, **k: corpus(prep.HELD_OUT_DOCS * 2),
            get_dataset_split_names=lambda *a, **k: ["train"],
        ), mock.patch.object(prep, "_load_split", recording):
            prep.prepare(config)

        return seen

    def _config(self, tmp: str, cap: int | None) -> TrainConfig:
        return TrainConfig(
            dataset_name="org/corpus", tokenizer_name="gpt2",
            streaming=True, max_train_docs=cap, data_dir=tmp,
        )

    def test_uncapped_run_keeps_the_splits_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            consumed = self._consumed(self._config(tmp, None))

        self.assertTrue(consumed["train"])
        self.assertTrue(consumed["validation"])
        self.assertFalse(
            consumed["train"] & consumed["validation"],
            "validation documents appeared in the training stream",
        )

    def test_capped_run_keeps_the_splits_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            consumed = self._consumed(self._config(tmp, 100))

        self.assertTrue(consumed["train"])
        self.assertTrue(consumed["validation"])
        self.assertFalse(consumed["train"] & consumed["validation"])


if __name__ == "__main__":
    unittest.main()
