"""Configuration for the decoder-only Transformer and its training runs.

Two dataclasses live here:

    * :class:`TransformerConfig` describes the architecture — width, depth,
      attention shape, and the pieces of a modern LLM stack that have knobs.
    * :class:`TrainConfig` describes a training run: data, optimization
      schedule, precision, distribution, and checkpointing.

:func:`parse_into` turns either dataclass into a command line interface, and
:func:`parse_configs` parses both from one command line so a run is fully
specified in a single invocation.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import MISSING, dataclass, fields
from typing import TypeVar

#: Precisions accepted by ``TrainConfig.dtype``.
DTYPES = ("bf16", "fp16", "fp32")


@dataclass
class TransformerConfig:
    """Hyperparameters describing a :class:`~src.model.Transformer`.

    Attributes:
        vocab_size: Number of distinct tokens in the vocabulary.
        d_model: Width of the residual stream.
        n_layers: Number of stacked decoder blocks.
        n_heads: Number of query heads in each attention layer. Must divide
            ``d_model`` evenly.
        n_kv_heads: Number of key/value heads. Must divide ``n_heads`` evenly.
            Equal to ``n_heads`` yields standard multi-head attention, ``1``
            yields multi-query attention, and any divisor in between yields
            grouped-query attention.
        ff_dim: Inner dimension of the feed-forward network. When ``None`` it
            defaults to the LLaMA heuristic of ``8/3 * d_model`` rounded up to
            the nearest multiple of ``ff_multiple_of``.
        ff_multiple_of: Alignment used when ``ff_dim`` is inferred. Keeping the
            hidden size a multiple of a power of two is friendlier to tensor
            cores.
        max_seq_len: Longest sequence the rotary tables are built for, and
            therefore the longest context the model can attend over.
        dropout: Dropout probability applied to attention weights and to the
            output of each sublayer. Modern LLM pretraining usually leaves this
            at zero.
        rope_theta: Base period of the rotary embedding. Larger values slow the
            frequency decay and help at long context lengths.
        norm_eps: Epsilon added to the RMSNorm denominator for stability.
        tie_embeddings: Whether the output projection reuses the input
            embedding matrix.
        init_std: Standard deviation of the normal weight initialization.
    """

    vocab_size: int = 52_000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int | None = None
    ff_dim: int | None = None
    ff_multiple_of: int = 256
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02

    def __post_init__(self) -> None:
        """Fills in derived fields and validates the configuration.

        Raises:
            ValueError: If ``d_model`` is not divisible by ``n_heads``, or if
                ``n_heads`` is not divisible by ``n_kv_heads``.
        """
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})."
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})."
            )

        if self.ff_dim is None:
            hidden = int(8 * self.d_model / 3)
            self.ff_dim = self.ff_multiple_of * math.ceil(
                hidden / self.ff_multiple_of
            )

    @property
    def head_dim(self) -> int:
        """Dimensionality of a single attention head."""
        return self.d_model // self.n_heads


@dataclass
class Seeds:
    """The random streams a training run consumes.

    Kept separate so that changing one does not move the others. Streams left
    unset are derived from a single base seed, so the common case stays one
    number while any stream remains independently settable.

    Attributes:
        model_init: Parameter initialization. Deliberately *not* offset by rank,
            so a single-process run and rank zero of a distributed one start
            from identical weights.
        data_order: Which blocks each rank reads, in which order. The per-rank
            offset lives in the sampler's position formula, not in this seed.
        dropout: Stochastic regularization. Offset by rank at the call site, so
            ranks do not apply identical masks.
    """

    model_init: int
    data_order: int
    dropout: int


@dataclass
class TrainConfig:
    """Settings for a training run.

    Attributes:
        dataset_name: Hugging Face dataset repository id, or a path to a local
            data file (``.jsonl``/``.parquet``/``.csv``/``.txt``).
        dataset_config: Configuration name within that dataset.
        text_column: Column holding the raw text. Conventions vary: ``"text"``
            for most, ``"content"`` for code, ``"story"`` for TinyStories.
        streaming: Read the corpus as a stream instead of downloading it whole.
            Required for corpora that will not fit on disk.
        tokenizer_name: Tokenizer to encode the corpus with. The model's
            vocabulary size is derived from it.
        data_dir: Directory holding the tokenized ``train.bin``/``val.bin``
            memory maps produced by ``training/data.py``.
        seq_len: Tokens per training example.
        max_train_docs: Cap on documents tokenized during preparation. ``None``
            uses the whole split. Also decides where a held-out set is carved
            from when the corpus ships only a training split.
        overwrite_data: Re-tokenize splits already complete on disk.
        batch_size: Examples per micro-batch, per device.
        grad_accum_steps: Micro-batches accumulated before each optimizer step.
            The global batch is ``batch_size * grad_accum_steps * world_size``.
        max_steps: Total optimizer steps to run.
        learning_rate: Peak learning rate reached at the end of warmup.
        min_lr: Floor the cosine schedule decays to.
        warmup_steps: Steps spent linearly ramping up to ``learning_rate``.
        weight_decay: Decoupled weight decay, applied only to matrices.
        grad_clip: Global gradient-norm clip. ``0`` disables clipping.
        beta1: AdamW first-moment decay.
        beta2: AdamW second-moment decay. ``0.95`` is the usual LM choice, as
            ``0.999`` adapts too slowly for the loss spikes seen early on.
        dtype: Autocast precision, one of :data:`DTYPES`.
        compile_model: Whether to wrap the model in :func:`torch.compile`.
        seed: Base seed. Every stream in :class:`Seeds` is derived from it when
            left unset, so a run is still specified by one number.
        model_init_seed: Override for the initialization stream.
        data_order_seed: Override for the data-order stream.
        dropout_seed: Override for the dropout stream.
        num_workers: Dataloader worker processes per rank.
        ddp_backend: Collective backend. ``"nccl"`` on CUDA, ``"gloo"``
            otherwise; ``"auto"`` picks based on device availability.
        out_dir: Directory checkpoints are written to.
        save_every: Steps between checkpoints.
        resume_from: Checkpoint path to restore model, optimizer, and step from.
        eval_every: Steps between validation passes.
        eval_steps: Validation batches per pass.
        log_every: Steps between training log lines.
        wandb_project: Weights & Biases project to log to. Leaving this unset
            disables tracking entirely, and the ``wandb`` package is then not
            needed at all.
        wandb_run_name: Display name for the run. Defaults to a generated one.
        wandb_mode: ``"online"``, ``"offline"`` to buffer to disk for later
            syncing, or ``"disabled"`` to keep the calls but drop the data.
    """

    # Namespaced repository ids are required: the bare "wikitext" alias no
    # longer resolves in current huggingface_hub versions.
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    text_column: str = "text"
    streaming: bool = False
    tokenizer_name: str = "gpt2"
    data_dir: str = "data"
    seq_len: int = 512
    max_train_docs: int | None = None
    overwrite_data: bool = False

    batch_size: int = 8
    grad_accum_steps: int = 4
    max_steps: int = 10_000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    dtype: str = "bf16"
    compile_model: bool = True
    seed: int = 1337
    model_init_seed: int | None = None
    data_order_seed: int | None = None
    dropout_seed: int | None = None
    num_workers: int = 2

    ddp_backend: str = "auto"

    out_dir: str = "checkpoints"
    save_every: int = 1000
    resume_from: str | None = None
    eval_every: int = 500
    eval_steps: int = 50
    log_every: int = 10

    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"

    def __post_init__(self) -> None:
        """Validates the run settings.

        Raises:
            ValueError: If a positive-only field is zero or negative, if
                ``dtype`` is unrecognized, or if ``min_lr`` exceeds
                ``learning_rate``.
        """
        positive = {
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "eval_steps": self.eval_steps,
            "log_every": self.log_every,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {self.dtype!r}.")
        if self.min_lr > self.learning_rate:
            raise ValueError(
                f"min_lr ({self.min_lr}) cannot exceed learning_rate "
                f"({self.learning_rate})."
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}."
            )
        if self.grad_clip < 0:
            raise ValueError(f"grad_clip must be non-negative, got {self.grad_clip}.")

    def seeds(self) -> Seeds:
        """Resolves the named random streams this run consumes.

        Streams left unset on the command line are derived from :attr:`seed`
        with a small fixed offset, so the streams stay distinct while the run is
        still specified by one number.

        Returns:
            The resolved :class:`Seeds`.
        """
        return Seeds(
            model_init=self.seed if self.model_init_seed is None
            else self.model_init_seed,
            data_order=self.seed + 1 if self.data_order_seed is None
            else self.data_order_seed,
            dropout=self.seed + 2 if self.dropout_seed is None
            else self.dropout_seed,
        )


_T = TypeVar("_T")


def _add_field_argument(parser: argparse.ArgumentParser, name: str, kind: object,
                        default: object) -> None:
    """Registers one dataclass field as a command line flag.

    Args:
        parser: Parser to extend.
        name: Field name, used verbatim as ``--name``.
        kind: The field's declared type annotation.
        default: Value to use when the flag is absent.
    """
    flag = f"--{name}"
    annotation = str(kind)

    if kind is bool or annotation == "bool":
        # Accepting an explicit value keeps overrides symmetric, so a config
        # default of True can be turned off with --compile_model=false.
        parser.add_argument(
            flag,
            type=lambda value: value.lower() in ("1", "true", "yes"),
            default=default,
            metavar="BOOL",
        )
    elif "int" in annotation:
        parser.add_argument(
            flag,
            type=lambda value: None if value.lower() == "none" else int(value),
            default=default,
        )
    elif "float" in annotation:
        parser.add_argument(flag, type=float, default=default)
    else:
        parser.add_argument(
            flag,
            type=lambda value: None if value.lower() == "none" else value,
            default=default,
        )


def parse_into(cls: type[_T], argv: list[str] | None = None) -> _T:
    """Builds a dataclass instance from command line arguments.

    Every field becomes a ``--field_name`` flag whose default is the field's
    own default, so an unmodified invocation reproduces the dataclass exactly.
    Booleans take an explicit value (``--compile_model=false``) and the literal
    string ``none`` maps to ``None`` for optional fields.

    Args:
        cls: A dataclass type, typically :class:`TrainConfig`.
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        An instance of ``cls`` with parsed overrides applied. The dataclass's
        own ``__post_init__`` validation still runs.

    Example:
        >>> config = parse_into(TrainConfig, ["--max_steps=100", "--dtype=fp32"])
        >>> config.max_steps
        100
    """
    parser = argparse.ArgumentParser(description=cls.__doc__)
    for field in fields(cls):
        default = field.default if field.default is not MISSING else None
        _add_field_argument(parser, field.name, field.type, default)

    namespace = parser.parse_args(argv)
    return cls(**vars(namespace))


#: Architecture fields that training derives rather than accepts: the
#: vocabulary comes from the tokenizer and the context from ``seq_len``.
_DERIVED_MODEL_FIELDS = frozenset({"vocab_size", "max_seq_len"})


def parse_configs(
    argv: list[str] | None = None,
) -> tuple[TrainConfig, dict[str, object]]:
    """Parses run settings and architecture overrides from one command line.

    Both dataclasses contribute flags to a single parser, so a run can be sized
    and scheduled in one invocation::

        python -m training.train --n_layers=6 --d_model=384 --max_steps=5000

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        A tuple ``(train_config, model_overrides)``. The second element is
        ready to splat into :func:`src.tokenizer.config_from_tokenizer`, which
        supplies the vocabulary size itself.
    """
    parser = argparse.ArgumentParser(description="Train a decoder-only Transformer.")

    train_names, model_names = [], []
    for cls, names in ((TrainConfig, train_names), (TransformerConfig, model_names)):
        for field in fields(cls):
            if cls is TransformerConfig and field.name in _DERIVED_MODEL_FIELDS:
                continue
            default = field.default if field.default is not MISSING else None
            _add_field_argument(parser, field.name, field.type, default)
            names.append(field.name)

    parsed = vars(parser.parse_args(argv))
    train_config = TrainConfig(**{name: parsed[name] for name in train_names})
    overrides = {name: parsed[name] for name in model_names}
    return train_config, overrides
