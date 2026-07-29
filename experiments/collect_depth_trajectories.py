"""Recording what every candidate depth would have done, for every request.

The controller is fitted after the fact on a frozen backbone, so the data it
learns from has to be collected first: for each request, run *every* candidate
depth and record what it produced, what that cost, and what the probe saw. The
controller then learns a map from the probe features to the depth worth paying
for. Nothing here trains anything.

Two things this file is careful about, because both are easy to get wrong in a
way that silently inflates the result.

**Label provenance.** Three different questions get called "quality at depth d",
and they are not interchangeable:

* ``teacher_forced`` scores the reference continuation in one parallel pass.
  Cheap, low variance, and the standard thing to fit on — but it never lets the
  endpoint's own mistakes into its context.
* ``free_running`` generates from the endpoint and scores what came out. This
  is the one that supports a serving claim, and it is the one that degrades.
* ``agreement`` compares the endpoint's free-running output against full depth,
  which is a *relative* measure and says nothing about whether either is right.

All three are stored, under names that say which is which, and the trainer is
told explicitly which to fit. Presenting the first as though it were the third
is the single most effective way to overstate early exiting, so the schema does
not permit it silently.

**Cache exactness.** For request-level routing there is no cache approximation
at all: every layer that runs sees exactly the keys and values it would have
seen at full depth, because the layers below it all ran. This is a real
advantage over token-level early exit, where an exited token's upper-layer
entries have to be synthesized. It also means the "exact cache" qualifier that
matters so much for the token-level oracle is vacuous here — recorded in the
schema as ``cache_semantics: "exact"`` so a later reader does not have to
reconstruct the argument.

Run it::

    python -m experiments.collect_depth_trajectories --out results/trajectories
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import RoutingConfig, TransformerConfig
from src.model import Transformer
from src.routing import pool_prompt_features
from experiments.workloads import (
    RequestShape,
    real_text_corpus,
    Workload,
    mixed_difficulty_corpus,
    split_by_source,
    train_demo_backbone,
)
from utils.costs import AnalyticalCostModel
from utils.provenance import RunRecord, file_digest

#: Version of the trajectory record layout. Bump when a field changes meaning.
#:
#: Version 2 adds document identity, per-request NLL *sums* alongside the means,
#: and the corpus the requests came from. The sums are what make a corpus-level
#: number correct: requests differ in length, so a mean of per-request means is
#: not the corpus NLL, and the difference grows with the spread of lengths.
SCHEMA_VERSION = 2

#: Corpora the collector can draw from.
#:
#: ``synthetic`` is ``mixed_difficulty_corpus``: a mechanism test whose depth
#: structure is a rule the experimenter installed. It answers "does the machinery
#: work", never "do real prompts vary in the depth they need".
#:
#: ``real_text`` draws from a tokenized corpus written by ``training.data``. This
#: is the only setting whose output belongs in a paper.
CORPORA = ("synthetic", "real_text")


def parse_shapes(shapes: tuple[str, ...]) -> tuple[RequestShape, ...]:
    """Parses ``"prompt:continuation"`` shape specifications.

    Args:
        shapes: Strings such as ``("64:32", "256:128")``.

    Returns:
        The parsed shapes, in the order given.

    Raises:
        ValueError: If a specification is malformed or non-positive. A shape that
            silently parsed to zero would produce requests with no continuation
            and an NLL over nothing.
    """
    parsed = []
    for shape in shapes:
        try:
            prompt, continuation = (int(part) for part in str(shape).split(":"))
        except ValueError as error:
            raise ValueError(
                f"shape {shape!r} must look like 'prompt:continuation', "
                f"for example '128:64'."
            ) from error
        if prompt < 1 or continuation < 1:
            raise ValueError(
                f"shape {shape!r} must have positive lengths, got "
                f"prompt={prompt}, continuation={continuation}."
            )
        parsed.append(RequestShape(prompt, continuation))
    if not parsed:
        raise ValueError("at least one shape is required.")
    return tuple(parsed)


def check_corpus_compatible(
    model_config: TransformerConfig,
    splits: list[tuple[str, Workload]],
    corpus_metadata: dict,
) -> None:
    """Rejects a corpus the model cannot score, before any scoring happens.

    Both checks below were found by running the collector rather than by reading
    it, and both used to surface as a stack trace from inside the embedding or
    the length guard — after minutes of work on the shapes that happened to fit.
    A collection that dies partway through leaves shards a resume will treat as
    complete, so these are startup errors by design.

    Args:
        model_config: Architecture that will do the scoring.
        splits: The workloads about to be scored.
        corpus_metadata: Corpus description, which carries the tokenizer size.

    Raises:
        ValueError: If a request is longer than the model's context, or if the
            corpus was tokenized with a vocabulary the model does not cover.
    """
    too_long = sorted(
        {
            (subset.prompt_len, subset.continuation_len)
            for _, subset in splits
            if len(subset)
            and subset.prompt_len + subset.continuation_len > model_config.max_seq_len
        }
    )
    if too_long:
        raise ValueError(
            f"request shape(s) {too_long} exceed the model's context of "
            f"{model_config.max_seq_len} tokens. Teacher forcing scores the "
            f"prompt and the continuation in one sequence, so the shape's *total* "
            f"must fit. Drop the long shapes from --shapes."
        )

    tokenizer_size = corpus_metadata.get("tokenizer_size")
    if tokenizer_size is not None and tokenizer_size > model_config.vocab_size:
        raise ValueError(
            f"the corpus was tokenized with a vocabulary of {tokenizer_size} but "
            f"the model covers {model_config.vocab_size}. Scoring it would index "
            f"past the embedding on any token above {model_config.vocab_size - 1}, "
            f"and a corpus that happened to contain none would score fine and be "
            f"the wrong corpus."
        )

    highest = max(
        (int(subset.sequences().max()) for _, subset in splits if len(subset)),
        default=-1,
    )
    if highest >= model_config.vocab_size:
        raise ValueError(
            f"the drawn requests contain token id {highest}, outside the model's "
            f"vocabulary of {model_config.vocab_size}."
        )


@dataclass
class TrajectoryConfig:
    """Settings for one collection run.

    Attributes:
        checkpoint: Model checkpoint to load. ``None`` trains the demonstration
            backbone in-process, which is what makes the pipeline runnable
            without any prior artefact.
        out_dir: Directory for the trajectory files.
        n_requests: Requests in the demonstration corpus.
        validation_fraction: Share of *sources* held out.
        probe_depth: Blocks run before the routing decision.
        depth_tiers: Candidate depths. ``None`` uses the model's exits.
        max_new_tokens: Tokens generated when collecting free-running labels.
            ``0`` skips them, which is much faster and much less informative.
        cache_dtype: Precision assumed when reporting cache bytes.
        seed: Base seed for corpus, training, and splitting.
        backbone_steps: Optimizer steps when training the demonstration model.
        resample: Whether the demonstration backbone draws a fresh batch each
            step instead of reusing a fixed corpus. Leaving it on is what stops
            the model memorizing the training requests; turning it off
            reproduces the memorization result recorded in the findings, where
            a convincing depth gradient existed only on the training split.
    """

    checkpoint: str | None = None
    out_dir: str = "results/trajectories"
    n_requests: int = 1024
    validation_fraction: float = 0.25
    probe_depth: int = 1
    depth_tiers: tuple[int, ...] | None = None
    max_new_tokens: int = 6
    cache_dtype: str = "bf16"
    seed: int = 0
    backbone_steps: int = 900
    resample: bool = True
    corpus: str = "synthetic"
    data: str = "data/val.bin"
    eos_id: int | None = None
    shapes: tuple[str, ...] = ("64:32", "128:64", "256:128")


@dataclass
class RequestRecord:
    """Everything collected about one request.

    Attributes:
        request_id: Index within the collection run.
        source_id: Group used for leakage-safe splitting.
        split: ``"train"`` or ``"validation"``.
        difficulty: Ground-truth label from the workload, where one exists.
        prompt_len: Prompt length in tokens.
        continuation_len: Reference continuation length.
        tiers: Candidate depths, ascending.
        teacher_forced_nll: Mean NLL of the reference continuation at each
            tier, scored in one parallel pass.
        teacher_forced_accuracy: Next-token accuracy over the same positions.
        teacher_forced_top1_agreement: Share of continuation positions whose
            greedy token matches full depth, teacher forced.
        free_running_reward: Exact-match rate of the generated continuation
            against the reference, per tier. Empty when generation was skipped.
        free_running_agreement: Share of generated tokens matching full
            depth's own generation, per tier.
        final_nll: Teacher-forced NLL at full depth, the usual reference point.
        cost_macs: Estimated multiply-accumulates per tier, prefill plus
            decode plus one head call per generated token.
        cost_depth_fraction: Tier divided by full depth, a cost measure with no
            hardware assumptions in it at all.
        kv_bytes: Cache bytes per tier.
        teacher_forced_nll_sum: Total NLL over scored positions, per tier. Sum
            these and ``valid_tokens`` separately to get a corpus NLL; averaging
            ``teacher_forced_nll`` across requests of different lengths does not.
        teacher_forced_top1_count: Correct top-1 predictions per tier.
        valid_tokens: Positions that contributed, the denominator for both.
        document_id: Document the request was cut from. Requests sharing one are
            not independent observations, so this is the clustered bootstrap's
            resampling unit.
        domain: Source label, for distribution-shift analysis.
        token_hash: Digest of the request's tokens, so the record identifies
            content rather than an offset that a re-tokenization would move.
        corpus_offset: Where the request began in the token file.
        shape: Request shape label, e.g. ``"p128c64"``.
    """

    request_id: int
    source_id: int
    split: str
    difficulty: str
    prompt_len: int
    continuation_len: int
    tiers: list[int]
    teacher_forced_nll: list[float]
    teacher_forced_accuracy: list[float]
    teacher_forced_top1_agreement: list[float]
    # Schema 2. Sums and counts, so a corpus number can be formed correctly by
    # summing numerators and denominators separately.
    teacher_forced_nll_sum: list[float] = field(default_factory=list)
    teacher_forced_top1_count: list[int] = field(default_factory=list)
    valid_tokens: int = 0
    # Schema 2. Identity, so a record names the bytes it scored and the document
    # a clustered bootstrap should resample.
    document_id: int = -1
    domain: str = "unknown"
    token_hash: str = ""
    corpus_offset: int = -1
    shape: str = ""
    free_running_reward: list[float] = field(default_factory=list)
    free_running_agreement: list[float] = field(default_factory=list)
    final_nll: float = 0.0
    cost_macs: list[float] = field(default_factory=list)
    cost_depth_fraction: list[float] = field(default_factory=list)
    kv_bytes: list[int] = field(default_factory=list)


@torch.no_grad()
def teacher_forced_labels(
    model: Transformer,
    workload: Workload,
    tiers: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Scores the reference continuation at every candidate depth.

    One parallel pass per tier over the whole workload. The readout covers all
    positions here, which the serving path never does — this is offline data
    collection, and paying a full-sequence vocabulary projection to get dense
    labels is the right trade.

    Args:
        model: Frozen backbone.
        workload: Requests with their reference continuations.
        tiers: Candidate depths.

    Returns:
        Arrays shaped ``(n_requests, n_tiers)`` for ``nll``, ``accuracy``,
        ``top1_agreement``, ``nll_sum`` and ``top1_count``, plus a scalar
        ``valid_tokens``. The sums are returned alongside the means because a
        corpus NLL is a ratio of totals: averaging per-request means weights a
        32-token request the same as a 128-token one.
    """
    sequences = workload.sequences()
    inputs, targets = sequences[:, :-1], sequences[:, 1:]
    start = workload.prompt_len - 1

    nll, accuracy, predictions = [], [], []
    nll_sum, top1_count = [], []
    for depth in tiers:
        state = model.forward_to_depth(inputs, depth)
        logits = model.endpoint_logits(state.hidden, depth)[:, start:]
        gold = targets[:, start:]

        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            gold.reshape(-1),
            reduction="none",
        ).view_as(gold)

        nll.append(per_token.mean(dim=1))
        nll_sum.append(per_token.sum(dim=1))
        greedy = logits.argmax(dim=-1)
        correct = (greedy == gold).float()
        accuracy.append(correct.mean(dim=1))
        top1_count.append(correct.sum(dim=1))
        predictions.append(greedy)

    deepest = predictions[-1]
    agreement = [
        (greedy == deepest).float().mean(dim=1) for greedy in predictions
    ]

    return {
        "nll": torch.stack(nll, dim=1).numpy(),
        "accuracy": torch.stack(accuracy, dim=1).numpy(),
        "top1_agreement": torch.stack(agreement, dim=1).numpy(),
        "nll_sum": torch.stack(nll_sum, dim=1).numpy(),
        "top1_count": torch.stack(top1_count, dim=1).numpy(),
        "valid_tokens": int(gold.size(1)),
    }


@torch.no_grad()
def free_running_labels(
    model: Transformer,
    workload: Workload,
    tiers: tuple[int, ...],
    max_new_tokens: int,
    probe_depth: int,
) -> dict[str, np.ndarray]:
    """Generates from every endpoint and scores what actually came out.

    This is the label that supports a serving claim. It is more expensive and
    noisier than the teacher-forced one, and it is the one that falls when an
    endpoint's own errors feed back into its context, so collecting both is how
    that gap becomes visible rather than assumed.

    Args:
        model: Frozen backbone with a router attached.
        workload: Requests to generate from.
        tiers: Candidate depths.
        max_new_tokens: Tokens to generate.
        probe_depth: Probe depth, only used to keep the fixed-depth path
            identical to the routed one.

    Returns:
        Arrays shaped ``(n_requests, n_tiers)`` for ``reward`` — exact-match
        rate against the reference — and ``agreement`` against full depth's own
        generation.
    """
    reward, generations = [], []
    for depth in tiers:
        model.attach_router(
            RoutingConfig(
                routing_mode="fixed",
                fixed_depth=depth,
                probe_depth=probe_depth,
                depth_tiers=tiers,
            )
        )
        out = model.generate_routed(
            workload.prompts, max_new_tokens=max_new_tokens, temperature=0.0
        )
        produced = torch.stack(out.completions())
        target = workload.references[:, :max_new_tokens]

        generations.append(produced)
        reward.append((produced == target).float().mean(dim=1))

    deepest = generations[-1]
    agreement = [
        (produced == deepest).float().mean(dim=1) for produced in generations
    ]
    return {
        "reward": torch.stack(reward, dim=1).numpy(),
        "agreement": torch.stack(agreement, dim=1).numpy(),
    }


@torch.no_grad()
def probe_features(
    model: Transformer,
    workload: Workload,
    probe_depth: int,
    pooling: str = "last_mean",
    include_length: bool = True,
) -> np.ndarray:
    """Computes the only thing the controller is ever allowed to see.

    Args:
        model: Frozen backbone.
        workload: Requests to probe.
        probe_depth: Blocks to run.
        pooling: Pooling scheme.
        include_length: Whether to append normalized prompt length.

    Returns:
        Features shaped ``(n_requests, feature_dim)``.
    """
    state = model.forward_to_depth(workload.prompts, probe_depth)
    return pool_prompt_features(
        state.hidden,
        pooling=pooling,
        include_length=include_length,
        max_seq_len=model.config.max_seq_len,
    ).numpy()


def tier_costs(
    config: TransformerConfig,
    tiers: tuple[int, ...],
    prompt_len: int,
    generated: int,
    cache_dtype: str = "bf16",
    lora_rank: int = 0,
    lora_targets_per_block: int = 0,
) -> dict[str, np.ndarray]:
    """Computes what each tier costs for one request shape.

    Args:
        config: Model architecture. ``exit_adapter_rank`` is read from it and
            charged once per generated token, alongside the vocabulary head.
        tiers: Candidate depths.
        prompt_len: Prompt length.
        generated: Tokens generated.
        cache_dtype: Precision assumed for cache bytes.
        lora_rank: Rank of any low-rank updates wrapped onto the backbone.
            Charged per executed block, so it grows with the routed depth.
        lora_targets_per_block: How many projections per block carry an update.

    Returns:
        Arrays over tiers for ``macs``, ``depth_fraction``, and ``kv_bytes``.

    Raises:
        ValueError: If ``generated`` is not positive. The generation path reads
            the first token from the prompt prefill and then performs exactly
            ``generated - 1`` incremental decode forwards.
    """
    if generated < 1:
        raise ValueError(f"generated must be positive, got {generated}.")

    cost = AnalyticalCostModel.from_config(config)

    macs, kv = [], []
    for depth in tiers:
        total = cost.prefill_macs(depth, prompt_len)
        # The prefill logits emit token one. Only tokens 1 .. generated-1 are
        # fed back through the backbone, so charging ``generated`` decode calls
        # overstates every tier by one full incremental forward.
        for step in range(generated - 1):
            total += cost.decode_macs(depth, prompt_len + step + 1)
        # One vocabulary projection per generated token, which is the whole
        # point of reading out only at the endpoint.
        total += generated * cost.head_macs
        # Retrofit modules are part of the endpoint, so they are part of its
        # cost. Omitting them reports a frontier the endpoints do not sit on.
        if config.exit_adapter_rank:
            total += generated * cost.exit_adapter_macs(config.exit_adapter_rank)
        if lora_rank and lora_targets_per_block:
            per_block = cost.lora_macs(lora_rank, lora_targets_per_block)
            total += depth * (prompt_len + generated) * per_block
        macs.append(total)
        # One position short of prompt + generated: the last token emitted is
        # never fed back, so nothing ever attends to it and its keys and values
        # are not written. Confirmed against a real cache across four request
        # shapes, where measured bytes are exactly
        # ``(prompt_len + generated - 1) * depth * kv_width * 2 * b``.
        cached = max(prompt_len + generated - 1, prompt_len)
        kv.append(cost.kv_bytes(depth, cached, cache_dtype))

    return {
        "macs": np.array(macs, dtype=float),
        "depth_fraction": np.array(tiers, dtype=float) / config.n_layers,
        "kv_bytes": np.array(kv, dtype=int),
    }


def collect(config: TrajectoryConfig) -> tuple[list[RequestRecord], np.ndarray, dict]:
    """Runs a whole collection.

    Args:
        config: Collection settings.

    Returns:
        A tuple ``(records, features, metadata)``.

    Raises:
        ValueError: If a supplied checkpoint has no model configuration.
    """
    torch.manual_seed(config.seed)

    if config.corpus not in CORPORA:
        raise ValueError(
            f"corpus must be one of {CORPORA}, got {config.corpus!r}."
        )

    corpus_metadata: dict
    if config.corpus == "real_text":
        # One workload per request shape, so no batch ever contains padding and
        # each shape's cost is exact rather than averaged.
        buckets, corpus_metadata = real_text_corpus(
            config.data,
            shapes=parse_shapes(config.shapes),
            n_requests=config.n_requests,
            eos_id=config.eos_id,
            seed=config.seed,
        )
        splits = []
        for bucket in buckets:
            train_part, validation_part = split_by_source(
                bucket, config.validation_fraction, seed=config.seed
            )
            splits.append(("train", train_part))
            splits.append(("validation", validation_part))
    else:
        workload = mixed_difficulty_corpus(
            n_requests=config.n_requests, seed=config.seed
        )
        train, validation = split_by_source(
            workload, config.validation_fraction, seed=config.seed
        )
        splits = [("train", train), ("validation", validation)]
        corpus_metadata = {
            "corpus": "synthetic",
            "warning": (
                "mixed_difficulty_corpus is a mechanism test. Its depth "
                "structure is a rule installed by the experimenter, so a result "
                "on it says whether the machinery works, not whether real "
                "prompts vary in the depth they need. Do not report it as "
                "evidence about language."
            ),
        }

    if config.checkpoint is None:
        print("training the demonstration backbone ...")
        model = train_demo_backbone(
            train,
            steps=config.backbone_steps,
            seed=config.seed,
            resample=config.resample,
        )
        checkpoint_digest = None
    else:
        blob = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
        if "model_config" not in blob:
            raise ValueError(
                f"{config.checkpoint} has no 'model_config'; it was not "
                f"written by training.train.save_checkpoint."
            )
        model = Transformer(blob["model_config"])
        # Unexpected keys are fine — a checkpoint saved with a router attached
        # carries controller weights this backbone does not want. *Missing*
        # keys are not: they leave part of the backbone at its random
        # initialization, and every number downstream would then describe an
        # untrained model without anything raising. So the two directions are
        # treated differently rather than both waved through by strict=False.
        incompatible = model.load_state_dict(blob["model"], strict=False)
        if incompatible.missing_keys:
            raise ValueError(
                f"{config.checkpoint} is missing {len(incompatible.missing_keys)} "
                f"parameter(s) this architecture needs, starting with "
                f"{incompatible.missing_keys[:3]}. Loading anyway would leave "
                f"them randomly initialized and silently invalidate every "
                f"measurement taken from this model."
            )
        model.eval()
        checkpoint_digest = file_digest(config.checkpoint)

    tiers = tuple(config.depth_tiers or model.config.exit_depths)
    routing = RoutingConfig(
        routing_mode="request",
        probe_depth=config.probe_depth,
        depth_tiers=tiers,
    ).resolve(model.config)
    model.attach_router(routing)

    records: list[RequestRecord] = []
    feature_blocks: list[np.ndarray] = []

    check_corpus_compatible(model.config, splits, corpus_metadata)

    for split_name, subset in splits:
        if not len(subset):
            continue
        print(
            f"scoring {len(subset)} {split_name} requests "
            f"({subset.prompt_len}+{subset.continuation_len}) at depths {tiers} ..."
        )

        teacher = teacher_forced_labels(model, subset, tiers)
        features = probe_features(
            model,
            subset,
            config.probe_depth,
            routing.controller_pooling,
            routing.controller_use_length,
        )
        free = (
            free_running_labels(
                model, subset, tiers, config.max_new_tokens, config.probe_depth
            )
            if config.max_new_tokens
            else {}
        )
        costs = tier_costs(
            model.config,
            tiers,
            subset.prompt_len,
            config.max_new_tokens or subset.continuation_len,
            config.cache_dtype,
        )

        feature_blocks.append(features)
        for row in range(len(subset)):
            records.append(
                RequestRecord(
                    request_id=len(records),
                    source_id=subset.source_ids[row],
                    split=split_name,
                    difficulty=subset.difficulty[row],
                    prompt_len=subset.prompt_len,
                    continuation_len=subset.continuation_len,
                    tiers=list(tiers),
                    teacher_forced_nll=teacher["nll"][row].tolist(),
                    teacher_forced_accuracy=teacher["accuracy"][row].tolist(),
                    teacher_forced_top1_agreement=(
                        teacher["top1_agreement"][row].tolist()
                    ),
                    free_running_reward=(
                        free["reward"][row].tolist() if free else []
                    ),
                    free_running_agreement=(
                        free["agreement"][row].tolist() if free else []
                    ),
                    final_nll=float(teacher["nll"][row][-1]),
                    cost_macs=costs["macs"].tolist(),
                    cost_depth_fraction=costs["depth_fraction"].tolist(),
                    kv_bytes=costs["kv_bytes"].tolist(),
                    teacher_forced_nll_sum=teacher["nll_sum"][row].tolist(),
                    teacher_forced_top1_count=[
                        int(value) for value in teacher["top1_count"][row]
                    ],
                    valid_tokens=teacher["valid_tokens"],
                    document_id=(
                        subset.document_ids[row]
                        if subset.document_ids is not None
                        else subset.source_ids[row]
                    ),
                    domain=(
                        subset.domains[row]
                        if subset.domains is not None
                        else "synthetic"
                    ),
                    token_hash=(
                        subset.token_hashes[row]
                        if subset.token_hashes is not None
                        else ""
                    ),
                    corpus_offset=(
                        subset.offsets[row] if subset.offsets is not None else -1
                    ),
                    shape=f"p{subset.prompt_len}c{subset.continuation_len}",
                )
            )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "tiers": list(tiers),
        "probe_depth": config.probe_depth,
        "n_layers": model.config.n_layers,
        "d_model": model.config.d_model,
        "feature_dim": int(feature_blocks[0].shape[1]),
        "controller_pooling": routing.controller_pooling,
        "controller_use_length": routing.controller_use_length,
        "cache_dtype": config.cache_dtype,
        "max_new_tokens": config.max_new_tokens,
        "checkpoint": config.checkpoint,
        "checkpoint_sha256": checkpoint_digest,
        "cache_semantics": "exact",
        "label_types": {
            "teacher_forced_*": "reference continuation scored in one pass",
            "free_running_*": "generated at the endpoint, then scored",
            "*_agreement": "relative to full depth, not to the truth",
        },
        "model_config": asdict(model.config),
        "corpus_metadata": corpus_metadata,
        "aggregation": (
            "corpus NLL = sum(teacher_forced_nll_sum) / sum(valid_tokens). "
            "Averaging teacher_forced_nll across requests weights a short "
            "request the same as a long one."
        ),
        "clustering_unit": "document_id",
    }
    return records, np.concatenate(feature_blocks), metadata


def write(
    records: list[RequestRecord],
    features: np.ndarray,
    metadata: dict,
    out_dir: str | Path,
    record: RunRecord,
) -> Path:
    """Writes trajectories, features, and provenance to a directory.

    Features go to a separate ``.npz`` because they are dense floats and the
    labels are not; keeping them apart means the label file stays readable by
    eye, which matters more than a single-file layout.

    Args:
        records: Per-request records.
        features: Probe features shaped ``(n_requests, feature_dim)``.
        metadata: Schema and model description.
        out_dir: Destination directory.
        record: Provenance for the run.

    Returns:
        The directory written to.
    """
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)

    with (path / "trajectories.jsonl").open("w") as handle:
        for item in records:
            handle.write(json.dumps(asdict(item)) + "\n")

    np.savez_compressed(path / "features.npz", features=features.astype(np.float32))
    record.write(path / "run.json", payload={"metadata": metadata})
    return path


def load(out_dir: str | Path) -> tuple[list[dict], np.ndarray, dict]:
    """Reads back what :func:`write` produced.

    Args:
        out_dir: Directory written by :func:`write`.

    Returns:
        A tuple ``(records, features, metadata)``.

    Raises:
        ValueError: If the schema version is newer than this code understands.
        FileNotFoundError: If the directory is missing a required file.
    """
    path = Path(out_dir)
    run = json.loads((path / "run.json").read_text())
    metadata = run["results"]["metadata"]

    version = metadata.get("schema_version", 0)
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"{path} uses trajectory schema {version}, but this code "
            f"understands at most {SCHEMA_VERSION}."
        )

    records = [
        json.loads(line)
        for line in (path / "trajectories.jsonl").read_text().splitlines()
        if line.strip()
    ]
    features = np.load(path / "features.npz")["features"]
    return records, features, metadata


def parse_args(argv: list[str] | None = None) -> TrajectoryConfig:
    """Builds a :class:`TrajectoryConfig` from the command line.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed configuration.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", dest="out_dir", default="results/trajectories")
    parser.add_argument(
        "--corpus",
        default="synthetic",
        choices=CORPORA,
        help=(
            "'real_text' draws from a tokenized corpus and is the only setting "
            "whose output belongs in a paper; 'synthetic' is a mechanism test."
        ),
    )
    parser.add_argument(
        "--data",
        default="data/val.bin",
        help="Token file for --corpus=real_text.",
    )
    parser.add_argument(
        "--eos_id",
        type=lambda value: None if value.lower() == "none" else int(value),
        default=None,
        help=(
            "End-of-text token id, used to find document boundaries. Omitting it "
            "makes the clustered bootstrap degenerate to an unclustered one."
        ),
    )
    parser.add_argument(
        "--shapes",
        default="64:32,128:64,256:128",
        help="Comma-separated prompt:continuation lengths for --corpus=real_text.",
    )
    parser.add_argument("--n_requests", type=int, default=1024)
    parser.add_argument("--validation_fraction", type=float, default=0.25)
    parser.add_argument("--probe_depth", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=6)
    parser.add_argument("--cache_dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone_steps", type=int, default=900)
    parser.add_argument(
        "--resample",
        type=lambda value: value.lower() in ("1", "true", "yes"),
        default=True,
        metavar="BOOL",
    )
    parsed = vars(parser.parse_args(argv))
    parsed["shapes"] = tuple(
        part.strip() for part in parsed["shapes"].split(",") if part.strip()
    )
    return TrajectoryConfig(**parsed)


def main(argv: list[str] | None = None) -> None:
    """Collects trajectories and writes them to disk.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.
    """
    config = parse_args(argv)
    provenance = RunRecord.create(
        "experiments.collect_depth_trajectories",
        config=asdict(config),
        seeds={"corpus": config.seed, "backbone": config.seed,
               "split": config.seed},
        inputs={
            "checkpoint": config.checkpoint or "trained in-process",
            **({"corpus": config.data} if config.corpus == "real_text" else {}),
        },
        notes=[
            "Request-level depth capping involves no cache approximation: "
            "every executed layer sees exact keys and values.",
            "teacher_forced_* and free_running_* answer different questions "
            "and must not be substituted for one another.",
        ],
    )
    print(provenance.summary())
    print()

    records, features, metadata = collect(config)
    path = write(records, features, metadata, config.out_dir, provenance)

    train = sum(1 for r in records if r.split == "train")
    print(f"\nwrote {len(records)} records ({train} train, "
          f"{len(records) - train} validation) to {path}")
    print(f"features {features.shape}, tiers {metadata['tiers']}")


if __name__ == "__main__":
    main()
