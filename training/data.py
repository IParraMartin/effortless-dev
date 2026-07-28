"""Corpus preparation and packed batching.

Text is tokenized once, ahead of training, into a flat array of token ids
written to disk as a memory map. Training then reads fixed-length blocks
straight out of that file, so no tokenization happens in the training loop and
the operating system handles caching.

Documents are concatenated end to end with an end-of-sequence token between
them, a layout usually called packing. Nothing is padded, which means no
compute is spent on padding and the model's strictly causal attention needs no
mask. The cost is that a block may straddle a document boundary; the
end-of-sequence token is what teaches the model to treat that as a hard reset.

Run this module from the repository root to build the files::

    python -m training.data --dataset_name Salesforce/wikitext \
        --dataset_config wikitext-103-raw-v1
"""

from __future__ import annotations

import gc
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.config import TrainConfig, parse_into
from src.tokenizer import load_tokenizer

#: Largest vocabulary that fits in ``uint16``. GPT-2's 50257 fits; anything
#: larger is stored as ``uint32`` instead, at twice the disk.
MAX_UINT16_VOCAB = 2**16

#: Sidecar recording how a ``.bin`` was written. Without it the element size
#: has to be guessed, and guessing wrong silently reinterprets every token.
#: It is written only after a split finishes, which also makes it the marker
#: for "this file is complete" rather than "this job died halfway".
META_SUFFIX = ".meta.json"

#: Documents held out for validation when a corpus ships no held-out split of
#: its own. Small on purpose: validation here measures loss during training,
#: and every document spent on it is one not trained on.
HELD_OUT_DOCS = 2_000

#: Sidecar fields that describe *what was tokenized*. A split may be reused
#: only when every one of these still matches, so changing the corpus, the
#: tokenizer or the document cap re-tokenizes instead of silently mixing.
_IDENTITY_FIELDS = (
    "dataset_name", "dataset_config", "text_column",
    "tokenizer_name", "max_train_docs",
)

#: Version of the sidecar schema, so a reader can reject files it predates.
META_SCHEMA_VERSION = 1


def token_dtype(vocab_size: int) -> np.dtype:
    """Picks the narrowest integer type that can hold every token id.

    Args:
        vocab_size: Number of distinct ids the tokenizer can emit.

    Returns:
        ``uint16`` when the vocabulary fits in sixteen bits, otherwise
        ``uint32``. There is no failure case: a large tokenizer costs disk, not
        an error.
    """
    return np.dtype(np.uint16 if vocab_size <= MAX_UINT16_VOCAB else np.uint32)


def read_meta(path: str | Path) -> dict:
    """Reads the sidecar describing a tokenized ``.bin``.

    Args:
        path: Path to the ``.bin`` file itself, not the sidecar.

    Returns:
        The recorded metadata. Files written before sidecars existed have none,
        and are reported as ``uint16`` with a ``legacy`` provenance marker,
        which is what they were.

    Raises:
        ValueError: If the sidecar was written by a newer schema than this
            code understands.
    """
    meta_path = Path(str(path) + META_SUFFIX)
    if not meta_path.exists():
        return {"dtype": "uint16", "schema_version": 0, "provenance": "legacy"}

    meta = json.loads(meta_path.read_text())
    version = meta.get("schema_version", 0)
    if version > META_SCHEMA_VERSION:
        raise ValueError(
            f"{meta_path} uses schema version {version}, but this code "
            f"understands at most {META_SCHEMA_VERSION}. Re-run preparation or "
            f"update the reader."
        )
    return meta


class PackedDataset(Dataset):
    """Fixed-length blocks read from a tokenized memory map.

    Args:
        path: Path to a ``.bin`` file written by :func:`prepare`.
        seq_len: Tokens per example. Each item reads ``seq_len + 1`` ids so the
            inputs and targets can be offset by one.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file holds too few tokens for even one block.
    """

    def __init__(self, path: str | Path, seq_len: int) -> None:
        self.path = Path(path)
        self.seq_len = seq_len

        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Run 'python -m training.data' to build it."
            )

        self.meta = read_meta(self.path)
        self.dtype = np.dtype(self.meta["dtype"])

        n_tokens = self.path.stat().st_size // self.dtype.itemsize
        self.n_blocks = (n_tokens - 1) // seq_len
        if self.n_blocks < 1:
            raise ValueError(
                f"{self.path} holds {n_tokens} tokens, too few for a block of "
                f"{seq_len}."
            )

        # Opened lazily: a memmap cannot be inherited safely across the fork
        # that dataloader workers use, so each worker maps the file itself.
        self._tokens: np.memmap | None = None

    def __len__(self) -> int:
        """Number of complete blocks in the file."""
        return self.n_blocks

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reads one block and splits it into inputs and targets.

        Args:
            index: Block index in ``[0, len(self))``.

        Returns:
            A tuple ``(input_ids, targets)``, both ``int64`` of length
            ``seq_len``, where ``targets`` is ``input_ids`` shifted by one.
        """
        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype=self.dtype, mode="r")

        start = index * self.seq_len
        block = self._tokens[start : start + self.seq_len + 1].astype(np.int64)
        chunk = torch.from_numpy(block)
        return chunk[:-1], chunk[1:]


def build_dataloader(
    path: str | Path,
    config: TrainConfig,
    world_size: int = 1,
    rank: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Wraps a :class:`PackedDataset` in a loader, sharded across ranks.

    Args:
        path: Path to the ``.bin`` file.
        config: Run settings supplying batch size and worker count.
        world_size: Number of processes sharing the dataset.
        rank: This process's index.
        shuffle: Whether to shuffle block order.

    Returns:
        A loader yielding ``(input_ids, targets)`` batches. Under distribution
        each rank sees a disjoint shard, so an epoch covers the corpus once in
        total rather than once per rank.
    """
    dataset = PackedDataset(path, config.seq_len)

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )


#: Extensions understood as local data files rather than hub repository ids.
LOCAL_SUFFIXES = {".json": "json", ".jsonl": "json", ".parquet": "parquet",
                  ".csv": "csv", ".txt": "text", ".arrow": "arrow"}


def _completed(path: Path, config: TrainConfig, tokenizer_size: int) -> int | None:
    """Reports whether a split on disk is finished and still matches the config.

    Args:
        path: The ``.bin`` file.
        config: Run settings the file would have to agree with.
        tokenizer_size: Vocabulary the tokenizer currently has.

    Returns:
        The token count of a reusable file, or ``None`` when it is absent,
        unfinished, or was built from different settings. A file whose sidecar
        disagrees is *not* reused: tokens from two different tokenizers in one
        stream would train without error and mean nothing.
    """
    meta_path = Path(str(path) + META_SUFFIX)
    if not path.exists() or not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None

    if meta.get("schema_version") != META_SCHEMA_VERSION:
        return None
    if meta.get("tokenizer_size") != tokenizer_size:
        return None
    for field in _IDENTITY_FIELDS:
        if meta.get(field) != getattr(config, field):
            return None

    # Cross-check the sidecar against the file it describes. A job killed
    # between the last write and the sidecar cannot produce this state, but a
    # truncated copy or a full disk can.
    expected = meta.get("n_tokens", 0) * np.dtype(meta["dtype"]).itemsize
    if path.stat().st_size != expected:
        return None

    return int(meta["n_tokens"])


def _available_splits(config: TrainConfig) -> list[str] | None:
    """Best-effort probe of a hub dataset's split names.

    Args:
        config: Run settings naming the dataset.

    Returns:
        The split names, or ``None`` when they could not be determined — an
        offline node, a private repository, a dataset whose metadata needs a
        script to evaluate. ``None`` means "unknown", never "none exist", and
        the caller falls back to trusting the requested name.
    """
    try:
        from datasets import get_dataset_split_names

        return list(
            get_dataset_split_names(config.dataset_name, config.dataset_config)
        )
    except Exception:
        return None


def needs_carving(config: TrainConfig) -> bool:
    """Whether the held-out set must be cut out of the training corpus.

    Decided **once per run**, not once per split, and that is the whole point.
    Whether validation is carved changes what *training* is allowed to contain:
    if the holdout comes off the front of the corpus, training has to skip past
    it. Deciding separately for each split produced exactly that leak — training
    kept every document while validation took the first two thousand, so the
    held-out set sat inside the training data and eval loss measured nothing.

    Args:
        config: Run settings naming the dataset.

    Returns:
        ``True`` when the corpus has no held-out split of its own.
    """
    if Path(config.dataset_name).suffix.lower() in LOCAL_SUFFIXES:
        return True

    available = _available_splits(config)
    if available is None:
        return False
    if "validation" in available or "test" in available:
        return False
    return "train" in available


def _load_split(
    config: TrainConfig,
    split: str,
    carve: bool | None = None,
) -> tuple[object, bool]:
    """Opens one split of the configured corpus.

    Accepts three things without the caller having to say which: a hub
    repository id, a path to a local data file, and a glob matching several.

    Split resolution is deliberately lenient, because corpora disagree about
    what the held-out split is called and plenty of pretraining corpora do not
    ship one at all. FineWeb-Edu is the case that motivated this: it has only
    ``train``, and an earlier version trusted the requested name under
    streaming and died with ``Bad split: validation`` *after* spending two
    hours writing the training file.

    Args:
        config: Run settings naming the dataset and split behaviour.
        split: Either ``"train"`` or ``"validation"``.
        carve: The run-wide decision from :func:`needs_carving`. Passing it
            keeps both splits consistent; ``None`` re-derives it, which is
            convenient for callers handling one split in isolation.

    Returns:
        A tuple ``(dataset, derived)``. ``derived`` is ``True`` when the corpus
        had no split of its own to serve this request and the caller must carve
        one out of the training stream.

    Raises:
        ValueError: If the dataset exists but has no usable split at all.
    """
    from datasets import load_dataset

    suffix = Path(config.dataset_name).suffix.lower()
    if suffix in LOCAL_SUFFIXES:
        # Local files carry no split metadata, so both splits read the same
        # files and are separated by carving.
        dataset = load_dataset(
            LOCAL_SUFFIXES[suffix],
            data_files=config.dataset_name,
            split="train",
            streaming=config.streaming,
        )
        return dataset, True

    if carve:
        # The run-wide decision already says there is no held-out split, so
        # both halves read the training corpus and are separated by carving.
        return load_dataset(
            config.dataset_name, config.dataset_config,
            split="train", streaming=config.streaming,
        ), True

    available = _available_splits(config)

    if available is None:
        # Unknown: trust the caller and let load_dataset report a bad name.
        resolved, derived = split, False
    elif split in available:
        resolved, derived = split, False
    elif split == "validation" and "test" in available:
        resolved, derived = "test", False
    elif "train" in available:
        # The common pretraining-corpus shape: one enormous train split and
        # nothing else. Carve the held-out set out of it rather than failing.
        resolved, derived = "train", True
    else:
        raise ValueError(
            f"{config.dataset_name}/{config.dataset_config} has splits "
            f"{available}, none usable as {split!r} and no 'train' to derive "
            f"one from."
        )

    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=resolved,
        streaming=config.streaming,
    )
    return dataset, derived


def _carve(dataset, split: str, config: TrainConfig):
    """Splits one corpus into disjoint training and validation streams.

    Where the held-out documents come from depends on whether training is
    capped, and the distinction matters more than it looks:

    * **Capped** (``max_train_docs`` set). Validation is taken from *beyond*
      where training stops. The two are disjoint, and the training stream is
      byte-for-byte what it would have been with no holdout at all — so a
      ``train.bin`` written before the holdout existed stays valid.
    * **Uncapped**. There is no "beyond", so validation comes off the front and
      training skips past it.

    Args:
        dataset: The training corpus, streaming or not.
        split: ``"train"`` or ``"validation"``.
        config: Run settings supplying the cap.

    Returns:
        The dataset restricted to that split's documents.
    """
    cap = config.max_train_docs

    if config.streaming:
        if split == "train":
            return dataset if cap is not None else dataset.skip(HELD_OUT_DOCS)
        if cap is not None:
            return dataset.skip(cap).take(HELD_OUT_DOCS)
        return dataset.take(HELD_OUT_DOCS)

    total = len(dataset)
    if cap is not None and cap < total:
        start, end = cap, min(cap + HELD_OUT_DOCS, total)
    else:
        # Either uncapped, or the cap covers the whole corpus so nothing lies
        # beyond it. Fall back to the front and make training skip.
        start, end = 0, max(min(HELD_OUT_DOCS, total // 10), 1)

    if split == "train":
        if start == 0:
            return dataset.select(range(end, total))
        return dataset
    return dataset.select(range(start, end))


def _iter_texts(dataset, column: str, limit: int | None):
    """Yields raw strings from a dataset of either style.

    Args:
        dataset: A ``Dataset`` or ``IterableDataset``.
        column: Column holding the text.
        limit: Maximum documents to yield, or ``None`` for all.

    Yields:
        Non-empty strings, in dataset order.

    Raises:
        ValueError: If ``column`` is absent, naming what is available instead.
    """
    # islice rather than enumerate-and-return: the latter pulls the document at
    # index `limit` out of the stream before deciding to stop, so the training
    # stream consumed one document past its cap. It was discarded rather than
    # tokenized, so nothing was wrong with the output — but it made the boundary
    # between train and a holdout carved from beyond the cap inexact, which is
    # not a property to leave approximate.
    source = dataset if limit is None else itertools.islice(dataset, limit)

    for example in source:
        if column not in example:
            raise ValueError(
                f"Column {column!r} not found. This dataset has "
                f"{sorted(example)}. Set --text_column to one of them."
            )
        text = example[column]
        if text and text.strip():
            yield text


def prepare(config: TrainConfig) -> None:
    """Tokenizes the configured corpus into ``train.bin`` and ``val.bin``.

    Tokens are appended to disk as they are produced rather than assembled in
    memory, so a corpus far larger than RAM can be prepared by combining this
    with ``streaming``.

    The element type is chosen from the tokenizer's size rather than fixed, and
    recorded in a ``.meta.json`` sidecar next to each ``.bin``. A vocabulary
    beyond 65536 tokens therefore costs twice the disk instead of refusing to
    prepare, and a reader never has to infer the element size from the file
    length.

    Args:
        config: Run settings naming the dataset, tokenizer, and output
            directory.

    Raises:
        ValueError: If a split yields no usable text.
    """
    tokenizer = load_tokenizer(config.tokenizer_name)
    dtype = token_dtype(len(tokenizer))

    out_dir = Path(config.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eos_id = tokenizer.eos_token_id

    # One decision for the whole run: see needs_carving on why per-split is a
    # leak rather than a style choice.
    carve = needs_carving(config)

    for split, filename in (("train", "train.bin"), ("validation", "val.bin")):
        path = out_dir / filename

        # Tokenizing a pretraining corpus takes hours, and a job that dies on
        # the second split should not throw away the first. A split with a
        # sidecar finished; one without it did not.
        if not config.overwrite_data:
            done = _completed(path, config, len(tokenizer))
            if done is not None:
                print(
                    f"{split:>10} -> {path}  {done:,} tokens  already complete, "
                    f"skipping (pass --overwrite_data=true to rebuild)"
                )
                continue

        dataset, _ = _load_split(config, split, carve)
        limit = config.max_train_docs if split == "train" else None
        if carve:
            dataset = _carve(dataset, split, config)

        total = 0
        batch: list[str] = []

        with open(path, "wb") as handle:
            def flush(texts: list[str]) -> int:
                """Tokenizes a batch and appends it, returning the token count."""
                if not texts:
                    return 0
                written = 0
                encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
                for ids in encoded:
                    ids.append(eos_id)
                    np.asarray(ids, dtype=dtype).tofile(handle)
                    written += len(ids)
                return written

            for text in _iter_texts(dataset, config.text_column, limit):
                batch.append(text)
                if len(batch) >= 1_000:
                    total += flush(batch)
                    batch = []
            total += flush(batch)

        if total == 0:
            if carve and split == "validation":
                raise ValueError(
                    f"No documents left to be held out for validation. It is "
                    f"taken from beyond max_train_docs "
                    f"({config.max_train_docs}), and the corpus appears to end "
                    f"at or before that. Lower --max_train_docs, or point "
                    f"--dataset_name at a larger config."
                )
            if carve and split == "train":
                raise ValueError(
                    f"No documents left for training. With max_train_docs "
                    f"unset, the first {HELD_OUT_DOCS} documents are held out "
                    f"for validation, and this corpus has no more than that. "
                    f"Set --max_train_docs so the holdout is taken from beyond "
                    f"the training data instead."
                )
            raise ValueError(
                f"Split {split!r} produced no tokens. Check --text_column "
                f"(currently {config.text_column!r})."
            )

        Path(str(path) + META_SUFFIX).write_text(
            json.dumps(
                {
                    "schema_version": META_SCHEMA_VERSION,
                    "dtype": dtype.name,
                    "n_tokens": total,
                    "tokenizer_name": config.tokenizer_name,
                    "tokenizer_size": len(tokenizer),
                    "dataset_name": config.dataset_name,
                    "dataset_config": config.dataset_config,
                    "text_column": config.text_column,
                    "split": split,
                    "max_train_docs": config.max_train_docs,
                    "derived_from_train": carve,
                    "held_out_docs": HELD_OUT_DOCS if carve else None,
                },
                indent=2,
            )
        )

        print(
            f"{split:>10} -> {path}  {total:,} tokens  {dtype.name}  "
            f"{os.path.getsize(path) / 1e6:.1f} MB",
            flush=True,
        )

        # Drop the reader now. A streaming dataset holds an HTTP session and
        # background prefetch threads, and leaving them for the interpreter to
        # collect is what lets one still be fetching a shard at shutdown.
        del dataset
        gc.collect()


if __name__ == "__main__":
    prepare(parse_into(TrainConfig))

    # Exit without waiting for the interpreter to finalize.
    #
    # A streaming reader's background threads can outlive the main thread and
    # touch the GIL after finalization has started, which aborts the process
    # with "PyGILState_Release: auto-releasing thread-state". Every file is
    # written and closed by the time prepare returns, so the work is safe — but
    # the nonzero exit makes Slurm report a completed job as FAILED, and a job
    # that lies about its own outcome is worse than one that is slow.
    #
    # This is the last statement in the program and the streams are flushed
    # first, so nothing that had not already run is skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
