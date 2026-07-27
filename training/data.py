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

import json
import os
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
META_SUFFIX = ".meta.json"

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


def _load_split(config: TrainConfig, split: str):
    """Opens one split of the configured corpus.

    Accepts three things without the caller having to say which: a hub
    repository id, a path to a local data file, and a glob matching several.
    Held-out splits are also resolved leniently, since corpora disagree about
    whether the second split is called ``validation`` or ``test``.

    Args:
        config: Run settings naming the dataset and split behaviour.
        split: Either ``"train"`` or ``"validation"``.

    Returns:
        A ``Dataset`` or, under streaming, an ``IterableDataset``.

    Raises:
        ValueError: If no usable split exists.
    """
    from datasets import load_dataset

    suffix = Path(config.dataset_name).suffix.lower()
    if suffix in LOCAL_SUFFIXES:
        # Local files carry no split metadata, so both splits read the same
        # files and are separated by the caller.
        return load_dataset(
            LOCAL_SUFFIXES[suffix],
            data_files=config.dataset_name,
            split="train",
            streaming=config.streaming,
        )

    if config.streaming:
        # Split names cannot be probed without downloading metadata for the
        # whole repository, so a stream trusts the name and reports failures.
        resolved = split
    else:
        from datasets import get_dataset_split_names

        available = get_dataset_split_names(
            config.dataset_name, config.dataset_config
        )
        if split in available:
            resolved = split
        elif split == "validation" and "test" in available:
            resolved = "test"
        else:
            raise ValueError(
                f"{config.dataset_name}/{config.dataset_config} has splits "
                f"{available}, none usable as {split!r}."
            )

    return load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=resolved,
        streaming=config.streaming,
    )


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
    for index, example in enumerate(dataset):
        if limit is not None and index >= limit:
            return
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
    is_local = Path(config.dataset_name).suffix.lower() in LOCAL_SUFFIXES

    for split, filename in (("train", "train.bin"), ("validation", "val.bin")):
        dataset = _load_split(config, split)
        limit = config.max_train_docs if split == "train" else None

        if is_local:
            # One file, two splits: hold back a slice for validation.
            if config.streaming:
                dataset = (
                    dataset.skip(1_000) if split == "train" else dataset.take(1_000)
                )
            else:
                held_out = min(1_000, len(dataset) // 10)
                dataset = (
                    dataset.select(range(held_out, len(dataset)))
                    if split == "train"
                    else dataset.select(range(held_out))
                )

        path = out_dir / filename
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
                },
                indent=2,
            )
        )

        print(
            f"{split:>10} -> {path}  {total:,} tokens  {dtype.name}  "
            f"{os.path.getsize(path) / 1e6:.1f} MB"
        )


if __name__ == "__main__":
    prepare(parse_into(TrainConfig))
