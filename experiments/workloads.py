"""Requests to route, and a backbone small enough to route them on a laptop.

Every result in this repository is at toy scale, and the honest way to handle
that is to make the toy workload *diagnostic* rather than realistic. The corpus
here is built so that the answer routing should give is known in advance:

* **easy** requests continue a constant run: the answer is the previous token.
  One block can apply that rule.
* **hard** requests continue an induction pattern: find where the current token
  appeared before and copy what followed it. That is a composition of two
  attention steps and needs at least two blocks.

This is the third design, and the two that failed are worth recording because
each failed for a different reason.

The first made hard requests *induction over the whole stack* and expected the
gradient to run across all six depths. Measured, the shallowest endpoint
reached **1.000** on both halves: two layers suffice for induction, which is a
known result and should have been anticipated.

The second made hard requests **memorization** — an arbitrary continuation
attached to a random prompt — which does produce a depth gradient, since
capacity is what depth supplies. It produced 0.47 / 0.82 / 0.87 across depths
and looked ideal. It is also *useless for routing*, and the reason is the whole
point of holding data out: memorization does not generalize, so on held-out
requests every depth scored at chance and the depth gradient vanished
completely. The measured adaptivity gain fell to **+0.005**, and a controller
fitted on it learned nothing because there was nothing to learn. A workload
whose depth structure exists only on the training split cannot demonstrate
routing.

What survives both objections is a *rule* that generalizes and whose depth
requirement is a composition count. The gradient here sits between depths one
and two, which is narrower than the earlier designs appeared to offer, and
narrow is the honest answer rather than the disappointing one.

Both kinds carry a tag token in position zero, so the information a controller
needs is present in the prompt and reachable from a one-block probe. That makes
this a test of the *machinery* — can the controller read a signal that is
demonstrably there, and does routing on it save what the accounting says — and
not evidence that real prompts separate this cleanly. They may not, and they
certainly do not separate along a tag. Establishing what real prompts do is the
point of running this at scale on real text, which has not been done.

A corpus of independent random tokens with no structure at all would be useless
here for the same reason it was useless for the cache-error study: with nothing
to learn, depth buys nothing, every endpoint is equally bad, and the experiment
returns a null by construction rather than by fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.config import TransformerConfig
from src.model import Transformer

#: Token marking a request whose continuation is a constant run.
TAG_EASY = 1

#: Token marking a request whose continuation requires induction.
TAG_HARD = 2

#: Lowest ordinary content token, leaving the tags unambiguous.
FIRST_CONTENT_TOKEN = 3

#: Backbone used by the demonstration pipeline, small enough to train on CPU in
#: a couple of minutes. It carries an exit on every layer because the depth
#: gradient this workload produces sits between depths one and two; exits every
#: second layer would straddle it and hide the only signal there is to route on.
DEMO_MODEL = dict(
    vocab_size=64,
    d_model=64,
    n_layers=6,
    n_heads=4,
    n_kv_heads=2,
    ff_dim=128,
    max_seq_len=48,
    exit_every=1,
    min_exit_layer=0,
    self_distill_weight=0.5,
)

#: Requests in the demonstration corpus. Both halves are rule-based and
#: generalize, so this is sized for a stable held-out estimate rather than to
#: exceed the model's capacity.
DEMO_REQUESTS = 4096


@dataclass
class Workload:
    """A batch of requests with their reference continuations.

    Attributes:
        prompts: Prompt ids shaped ``(n_requests, prompt_len)``. Every prompt
            has the same length here, which keeps the toy pipeline free of
            padding questions that the routing code handles but that would
            obscure what is being measured.
        references: Reference continuations shaped
            ``(n_requests, continuation_len)``.
        difficulty: ``"easy"`` or ``"hard"`` per request — the ground truth a
            controller is trying to recover, recorded so its errors can be
            attributed.
        source_ids: Group each request came from. Splits are made on this, not
            on the request index, because two requests built from the same
            underlying block share structure and would leak across a split.
        spec: The shape arguments this corpus was built from, so more requests
            from the same distribution can be drawn later. Training resamples
            rather than reusing a fixed set, and needs this to do it.
    """

    prompts: torch.Tensor
    references: torch.Tensor
    difficulty: list[str]
    source_ids: list[int]
    spec: dict = field(default_factory=dict)
    #: Identity of the document each request was cut from. ``None`` for the
    #: synthetic corpus, where requests are generated rather than sampled.
    document_ids: list[int] | None = None
    #: Content digest of each request's tokens, so a record identifies the bytes
    #: it was scored on rather than the offsets they happened to live at.
    token_hashes: list[str] | None = None
    #: Where each request begins in the underlying token file.
    offsets: list[int] | None = None
    #: Free-form label carried through to the record, for source-level shift
    #: analysis. Uniform when the corpus has no domain metadata.
    domains: list[str] | None = None
    #: UTF-8 byte length of each request's *continuation* text. This is what
    #: makes quality comparable across tokenizers: per-token NLL is not, since a
    #: tokenizer that splits text more finely earns a lower per-token loss on the
    #: same string. ``None`` when no tokenizer was supplied to decode with.
    continuation_bytes: list[int] | None = None
    #: The decoded prompt and continuation text of each request, so a model with
    #: a different tokenizer can re-encode the identical string.
    texts: list[tuple[str, str]] | None = None
    #: Digest of the continuation text, so two models can prove they scored the
    #: same content rather than the same offsets.
    text_hashes: list[str] | None = None

    def __len__(self) -> int:
        """Number of requests."""
        return self.prompts.size(0)

    @property
    def prompt_len(self) -> int:
        """Tokens per prompt."""
        return self.prompts.size(1)

    @property
    def continuation_len(self) -> int:
        """Tokens per reference continuation."""
        return self.references.size(1)

    def sequences(self) -> torch.Tensor:
        """Prompts and references concatenated, for teacher-forced scoring."""
        return torch.cat((self.prompts, self.references), dim=1)

    def select(self, rows: list[int]) -> Workload:
        """Extracts a subset of requests.

        Args:
            rows: Request indices to keep.

        Returns:
            A new workload holding those requests.
        """
        index = torch.tensor(rows, dtype=torch.long)

        def subset(values: list | None) -> list | None:
            return None if values is None else [values[row] for row in rows]

        return Workload(
            prompts=self.prompts.index_select(0, index),
            references=self.references.index_select(0, index),
            difficulty=[self.difficulty[row] for row in rows],
            source_ids=[self.source_ids[row] for row in rows],
            spec=dict(self.spec),
            document_ids=subset(self.document_ids),
            token_hashes=subset(self.token_hashes),
            offsets=subset(self.offsets),
            domains=subset(self.domains),
            continuation_bytes=subset(self.continuation_bytes),
            texts=subset(self.texts),
            text_hashes=subset(self.text_hashes),
        )


def mixed_difficulty_corpus(
    n_requests: int = DEMO_REQUESTS,
    block_len: int = 12,
    repeat_len: int = 6,
    continuation_len: int = 4,
    vocab_size: int = 64,
    hard_fraction: float = 0.5,
    seed: int = 0,
) -> Workload:
    """Builds requests whose required depth is known in advance.

    Every prompt has the same length, which is not cosmetic. An earlier version
    padded short prompts by repeating their last token, and that silently
    destroyed the induction task: the padded copies became the most recent
    occurrence of the query, so looking it up returned the padding rather than
    the answer. Held-out accuracy sat at chance across every depth and looked
    like a modelling failure. Building both halves at one length removes the
    possibility.

    Args:
        n_requests: Number of requests.
        block_len: Length of the block a hard request repeats.
        repeat_len: How much of that block is repeated before the continuation.
        continuation_len: Tokens to be predicted.
        vocab_size: Vocabulary the content tokens are drawn from.
        hard_fraction: Share of requests needing induction.
        seed: Random seed.

    Returns:
        The workload. Prompts are ``1 + block_len + repeat_len`` tokens for
        both halves.

    Raises:
        ValueError: If the vocabulary leaves no room for content tokens, or if
            the block cannot supply the repeat and the continuation.
    """
    if vocab_size <= FIRST_CONTENT_TOKEN:
        raise ValueError(
            f"vocab_size must exceed {FIRST_CONTENT_TOKEN} to leave room for "
            f"content tokens, got {vocab_size}."
        )
    if repeat_len + continuation_len > block_len:
        raise ValueError(
            f"block_len ({block_len}) must cover repeat_len ({repeat_len}) "
            f"plus continuation_len ({continuation_len}); the continuation is "
            f"read out of the block itself."
        )
    if continuation_len + 4 > block_len:
        raise ValueError(
            f"block_len ({block_len}) leaves no room to vary the repeat "
            f"distance around continuation_len ({continuation_len}), and a "
            f"fixed distance makes the task solvable by position alone."
        )
    if repeat_len < 2:
        raise ValueError(
            f"repeat_len ({repeat_len}) must be at least 2: the body has to "
            f"hold the longest block plus a repeat of at least two tokens, or "
            f"the repeat gets truncated and the answer becomes unrecoverable."
        )

    generator = torch.Generator().manual_seed(seed)
    n_hard = int(round(n_requests * hard_fraction))

    prompts, references, difficulty, source_ids = [], [], [], []

    def content(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randint(
            FIRST_CONTENT_TOKEN, vocab_size, shape, generator=generator
        )

    body_len = block_len + repeat_len

    for request in range(n_requests):
        is_hard = request < n_hard
        if is_hard:
            # A block followed by the start of its own repeat, at a distance
            # that changes from request to request. The varying distance is
            # load bearing: with the block always the same length, "copy from
            # exactly k positions back" solves the task with one attention
            # layer and no lookup at all, and the measured depth gradient
            # disappeared entirely. Varying it forces a content-based match,
            # which is the two-step composition depth is supposed to buy.
            span = int(
                torch.randint(
                    continuation_len + 4, block_len + 1, (1,), generator=generator
                )
            )
            # The repeat has to fit inside the body alongside the block. An
            # earlier version clamped a negative filler to zero and then
            # truncated the prompt, which silently cut the tail off the repeat:
            # the last prompt token was no longer the one the answer follows,
            # so a fraction of the "hard" requests had no recoverable answer at
            # all. Bounding the prefix instead makes that unrepresentable.
            largest_prefix = min(span - continuation_len, body_len - span)
            prefix = int(
                torch.randint(2, largest_prefix + 1, (1,), generator=generator)
            )
            filler = body_len - span - prefix

            block = content((span,))
            prompt = torch.cat(
                (
                    torch.tensor([TAG_HARD]),
                    content((filler,)),
                    block,
                    block[:prefix],
                )
            )
            reference = block[prefix : prefix + continuation_len]
        else:
            # Random content whose continuation is simply the previous token.
            # One block suffices, so every endpoint solves these and the
            # cheapest one is correct.
            body = content((body_len,))
            prompt = torch.cat((torch.tensor([TAG_EASY]), body))
            reference = body[-1].repeat(continuation_len)

        prompts.append(prompt)
        references.append(reference)
        difficulty.append("hard" if is_hard else "easy")
        # Requests built from one block share structure, so they must not be
        # split across train and validation. Here each request has its own
        # block, but the field exists so real corpora can group by document.
        source_ids.append(request)

    return Workload(
        prompts=torch.stack(prompts),
        references=torch.stack(references),
        difficulty=difficulty,
        source_ids=source_ids,
        spec={
            "block_len": block_len,
            "repeat_len": repeat_len,
            "continuation_len": continuation_len,
            "vocab_size": vocab_size,
            "hard_fraction": hard_fraction,
        },
    )


@dataclass(frozen=True)
class RequestShape:
    """One prompt/continuation length pair.

    Cost is a function of request shape, not a scalar per tier: prefill scales
    with the prompt and decode scales with the context as it grows. A collection
    that used one shape everywhere would produce a cost model that cannot be
    queried for anything else, so requests are drawn across several shapes and
    each record carries its own.

    Attributes:
        prompt_len: Prompt tokens.
        continuation_len: Reference continuation tokens.
    """

    prompt_len: int
    continuation_len: int

    @property
    def total(self) -> int:
        """Tokens the request occupies end to end."""
        return self.prompt_len + self.continuation_len

    def label(self) -> str:
        """A short identifier used in shard filenames and records."""
        return f"p{self.prompt_len}c{self.continuation_len}"


#: Shapes the real-text collector draws by default. Chosen so prefill and decode
#: costs differ materially across them; a shape sweep that varies only slightly
#: cannot separate the two components.
DEFAULT_SHAPES = (
    RequestShape(64, 32),
    RequestShape(128, 64),
    RequestShape(256, 128),
)


def real_text_corpus(
    path: str | Path,
    shapes: tuple[RequestShape, ...] = DEFAULT_SHAPES,
    n_requests: int = 1024,
    eos_id: int | None = None,
    seed: int = 0,
    split: str = "validation",
    tokenizer_name: str | None = None,
) -> tuple[list[Workload], dict]:
    """Draws requests from a tokenized corpus written by ``training.data``.

    This is what replaces :func:`mixed_difficulty_corpus` for anything that will
    be reported. The synthetic corpus remains a mechanism test: its depth
    structure is a rule the experimenter installed, so measuring a controller on
    it says whether the machinery works, not whether real prompts vary in the
    depth they need.

    Three design choices are worth stating, because each one removes a way the
    collection could be wrong rather than merely inconvenient:

    **No padding, ever.** Requests are grouped into one workload per shape, so
    every tensor is rectangular without a pad token in it. The model's attention
    is causal-only and has no padding mask, so a padded batch would let real
    positions attend to pad positions and silently corrupt every score. Bucketing
    by shape also makes per-shape cost exact instead of averaged.

    **Requests do not straddle documents.** With ``eos_id`` given, document
    boundaries are found exactly and every request is a contiguous slice inside
    one document. A slice spanning an end-of-text boundary is a request whose
    continuation is unrelated to its prompt, which would depress every endpoint
    equally and add noise the controller cannot predict.

    **The document is the cluster.** ``source_ids`` is the document id, so a
    clustered bootstrap resamples documents rather than requests. Two requests
    from one document are not independent observations, and treating them as such
    narrows every interval the evaluation reports.

    Args:
        path: Path to a ``.bin`` written by ``python -m training.data``.
        shapes: Request shapes to draw. Requests are spread evenly across them.
        n_requests: Total requests to draw.
        eos_id: End-of-text token id. When ``None``, document boundaries are
            unknown and the whole file is treated as one document — which makes
            the clustered bootstrap degenerate to the unclustered one, so the
            returned metadata records that this happened.
        seed: Seed for offset selection.
        split: Split label recorded on every request.
        tokenizer_name: Tokenizer to decode requests back to text with. Supplying
            it records each continuation's UTF-8 byte length and text digest,
            which is what a cross-tokenizer comparison needs: per-token NLL is
            not comparable between tokenizers, and bits per byte is. Defaults to
            the tokenizer recorded in the corpus sidecar. Pass ``""`` to skip
            decoding, at the cost of being unable to compare against a model from
            another tokenizer family.

    Returns:
        A tuple ``(workloads, metadata)`` with one workload per shape that got at
        least one request, and metadata describing the corpus, the tokenizer, the
        document count and the shapes drawn.

    Raises:
        FileNotFoundError: If the token file does not exist.
        ValueError: If ``n_requests`` is not positive, or if no document is long
            enough for the shortest requested shape — a silent empty collection
            would look like a successful run that measured nothing.
    """
    import hashlib

    import numpy as np

    from training.data import read_meta

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with 'python -m training.data'."
        )
    if n_requests < 1:
        raise ValueError(f"n_requests must be positive, got {n_requests}.")

    meta = read_meta(path)
    tokens = np.memmap(path, dtype=np.dtype(meta["dtype"]), mode="r")

    segments = _document_segments(tokens, eos_id)
    shortest = min(shape.total for shape in shapes)
    usable = [(start, end) for start, end in segments if end - start >= shortest]
    if not usable:
        raise ValueError(
            f"no document in {path} holds {shortest} tokens, the shortest "
            f"requested shape. Either the corpus is tiny or eos_id={eos_id} is "
            f"wrong for this tokenizer, which would split on the wrong token."
        )

    resolved_tokenizer = (
        meta.get("tokenizer_name") if tokenizer_name is None else tokenizer_name
    )
    decoder = None
    decode_error = None
    if resolved_tokenizer:
        try:
            from src.tokenizer import load_tokenizer

            decoder = load_tokenizer(resolved_tokenizer)
        except Exception as error:  # noqa: BLE001 - reported, never fatal
            decode_error = f"{type(error).__name__}: {error}"

    rng = np.random.default_rng(seed)
    per_shape: dict[RequestShape, list[dict]] = {shape: [] for shape in shapes}

    # Round-robin over shapes so a truncated collection is still balanced across
    # them, rather than holding every short request and no long ones.
    for request in range(n_requests):
        shape = shapes[request % len(shapes)]
        candidates = [
            index
            for index, (start, end) in enumerate(usable)
            if end - start >= shape.total
        ]
        if not candidates:
            continue
        document = int(rng.choice(candidates))
        start, end = usable[document]
        offset = int(rng.integers(start, end - shape.total + 1))
        block = np.asarray(tokens[offset : offset + shape.total], dtype=np.int64)
        per_shape[shape].append(
            {
                "document": document,
                "offset": offset,
                "tokens": block,
                "hash": hashlib.sha256(block.tobytes()).hexdigest()[:32],
            }
        )

    workloads = []
    for shape, drawn in per_shape.items():
        if not drawn:
            continue
        stacked = torch.from_numpy(np.stack([item["tokens"] for item in drawn]))
        texts = byte_lengths = text_hashes = None
        if decoder is not None:
            texts, byte_lengths, text_hashes = _decode_requests(
                decoder, stacked, shape.prompt_len
            )
        workloads.append(
            Workload(
                prompts=stacked[:, : shape.prompt_len],
                references=stacked[:, shape.prompt_len :],
                difficulty=["unknown"] * len(drawn),
                source_ids=[item["document"] for item in drawn],
                document_ids=[item["document"] for item in drawn],
                token_hashes=[item["hash"] for item in drawn],
                offsets=[item["offset"] for item in drawn],
                domains=[meta.get("dataset_name", "unknown")] * len(drawn),
                continuation_bytes=byte_lengths,
                texts=texts,
                text_hashes=text_hashes,
                spec={
                    "corpus": "real_text",
                    "shape": shape.label(),
                    "prompt_len": shape.prompt_len,
                    "continuation_len": shape.continuation_len,
                    "split": split,
                },
            )
        )

    drawn_total = sum(len(workload) for workload in workloads)
    metadata = {
        "corpus": "real_text",
        "path": str(path),
        "split": split,
        "dataset_name": meta.get("dataset_name"),
        "dataset_config": meta.get("dataset_config"),
        "tokenizer_name": meta.get("tokenizer_name"),
        "tokenizer_size": meta.get("tokenizer_size"),
        "corpus_tokens": int(meta.get("n_tokens", tokens.size)),
        "eos_id": eos_id,
        "documents_found": len(segments),
        "documents_long_enough": len(usable),
        "requests_requested": n_requests,
        "requests_drawn": drawn_total,
        "shapes": [shape.label() for shape in shapes],
        "seed": seed,
        "decoded_with": resolved_tokenizer if decoder is not None else None,
        "quality_unit": "bits_per_byte" if decoder is not None else "per_token",
    }
    if decode_error is not None:
        metadata["decode_note"] = (
            f"could not load tokenizer {resolved_tokenizer!r} ({decode_error}), so "
            f"continuation byte lengths are unavailable. Quality can only be "
            f"reported per token, which is not comparable against a model that "
            f"uses a different tokenizer."
        )
    elif decoder is None:
        metadata["decode_note"] = (
            "decoding was skipped, so quality can only be reported per token, "
            "which is not comparable across tokenizers."
        )
    if eos_id is None:
        metadata["clustering_note"] = (
            "eos_id was not supplied, so document boundaries are unknown and the "
            "whole file is treated as one document. source_id is therefore "
            "constant and a clustered bootstrap degenerates to an unclustered "
            "one, which reports intervals that are too narrow."
        )
    if drawn_total < n_requests:
        metadata["truncation_note"] = (
            f"{n_requests - drawn_total} request(s) were not drawn because no "
            f"document was long enough for their shape"
        )
    return workloads, metadata


def _document_segments(
    tokens, eos_id: int | None
) -> list[tuple[int, int]]:
    """Finds document boundaries in a packed token file.

    Args:
        tokens: The token array.
        eos_id: End-of-text token id, or ``None`` to treat the file as one
            document.

    Returns:
        Half-open ``(start, end)`` ranges, one per document, excluding the
        separator itself. Empty documents are dropped.
    """
    import numpy as np

    if eos_id is None:
        return [(0, int(tokens.size))]

    breaks = np.flatnonzero(np.asarray(tokens) == eos_id)
    segments, start = [], 0
    for position in breaks:
        if position > start:
            segments.append((start, int(position)))
        start = int(position) + 1
    if start < tokens.size:
        segments.append((start, int(tokens.size)))
    return segments


def split_by_source(
    workload: Workload,
    validation_fraction: float = 0.25,
    seed: int = 0,
) -> tuple[Workload, Workload]:
    """Splits a workload by source, never by request.

    Splitting on the request index would place two requests derived from the
    same block on opposite sides, and a controller could then score well by
    recognizing the block rather than by judging difficulty. Grouping by source
    removes that route.

    Args:
        workload: Requests to split.
        validation_fraction: Share of *sources* held out.
        seed: Random seed for the shuffle.

    Returns:
        A tuple ``(train, validation)``.
    """
    sources = sorted(set(workload.source_ids))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(sources), generator=generator).tolist()

    n_held = max(1, int(round(len(sources) * validation_fraction)))
    held_out = {sources[index] for index in order[:n_held]}

    train_rows = [
        row for row, source in enumerate(workload.source_ids)
        if source not in held_out
    ]
    val_rows = [
        row for row, source in enumerate(workload.source_ids)
        if source in held_out
    ]
    return workload.select(train_rows), workload.select(val_rows)


def train_demo_backbone(
    workload: Workload,
    steps: int = 3000,
    batch_size: int = 64,
    learning_rate: float = 3e-3,
    seed: int = 0,
    verbose: bool = True,
    resample: bool = True,
    resample_seed: int = 1_000_000,
) -> Transformer:
    """Trains a multi-exit backbone on a workload.

    The exits are trained together, which is the ordinary multi-exit objective:
    depth-weighted cross-entropy plus self-distillation from the final layer.
    Nothing about routing enters training, because the controller is fitted
    afterwards on a frozen backbone — the two must be separable before joint
    training can be interpreted.

    **Batches are resampled from the distribution rather than drawn from a fixed
    set**, and that is not an optimization. Trained on a fixed 3072-request
    corpus, this model memorized it: hard-request accuracy climbed 0.50 → 0.94
    across depths on the training split and *fell* 0.29 → 0.24 on held-out
    requests. Depth was buying memorization, not the rule, so the deepest
    endpoint was the worst one on data it had not seen — which would have made
    the routing demonstration measure overfitting. Since both halves of this
    corpus are rules with an unlimited supply of instances, drawing fresh ones
    removes the possibility entirely.

    Args:
        workload: Requests to train on, and the shape to resample from.
        steps: Optimizer steps.
        batch_size: Sequences per step.
        learning_rate: AdamW learning rate.
        seed: Random seed for initialization and batching.
        verbose: Whether to print progress.
        resample: Whether to draw a fresh batch each step.
        resample_seed: Base of the resampling seed stream, kept far from the
            evaluation corpus's seeds so training never draws a held-out
            request by coincidence.

    Returns:
        The trained model, in eval mode.
    """
    torch.manual_seed(seed)
    model = Transformer(TransformerConfig(**DEMO_MODEL))
    sequences = workload.sequences()
    spec = dict(workload.spec)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for step in range(steps):
        if resample and spec:
            batch = mixed_difficulty_corpus(
                n_requests=batch_size, seed=resample_seed + step, **spec
            ).sequences()
        else:
            rows = torch.randint(0, sequences.size(0), (batch_size,))
            batch = sequences[rows]
        out = model(batch[:, :-1], targets=batch[:, 1:])
        optimizer.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if verbose and (step + 1) % max(steps // 4, 1) == 0:
            per_exit = " ".join(
                f"L{layer}:{value:.3f}"
                for layer, value in sorted(out.exit_losses.items())
            )
            loss = float(out.loss.detach())
            print(f"  step {step + 1:>4}  loss {loss:.4f}  {per_exit}")

    model.eval()
    return model


@torch.no_grad()
def exit_accuracy_by_difficulty(
    model: Transformer,
    workload: Workload,
) -> dict[str, dict[int, float]]:
    """Measures each endpoint's accuracy on each kind of request.

    This is the diagnostic that says whether the workload did what it was built
    to do. If easy and hard requests saturate at the same depth there is
    nothing for a router to exploit, and any routing result afterwards would be
    measuring noise.

    Args:
        model: Trained backbone.
        workload: Requests to score.

    Returns:
        Next-token accuracy over the continuation, keyed by difficulty and then
        by depth.
    """
    sequences = workload.sequences()
    inputs, targets = sequences[:, :-1], sequences[:, 1:]
    start = workload.prompt_len - 1

    results: dict[str, dict[int, float]] = {}
    for depth in model.config.exit_depths:
        state = model.forward_to_depth(inputs, depth)
        predictions = model.endpoint_logits(state.hidden, depth).argmax(dim=-1)
        correct = predictions[:, start:] == targets[:, start:]

        for kind in ("easy", "hard"):
            rows = [i for i, d in enumerate(workload.difficulty) if d == kind]
            if not rows:
                continue
            value = float(correct[torch.tensor(rows)].float().mean())
            results.setdefault(kind, {})[depth] = value
    return results


def format_accuracy_table(results: dict[str, dict[int, float]]) -> str:
    """Renders :func:`exit_accuracy_by_difficulty` as a table.

    Args:
        results: Accuracy keyed by difficulty and depth.

    Returns:
        A multi-line string.
    """
    depths = sorted({depth for row in results.values() for depth in row})
    header = f"{'difficulty':>10}  " + "  ".join(f"d{d:<5}" for d in depths)
    lines = [header, "-" * len(header)]
    for kind, row in sorted(results.items()):
        cells = "  ".join(f"{row.get(depth, float('nan')):.3f}" for depth in depths)
        lines.append(f"{kind:>10}  {cells}")
    return "\n".join(lines)


if __name__ == "__main__":
    workload = mixed_difficulty_corpus()
    print(f"{len(workload)} requests, prompt {workload.prompt_len}, "
          f"continuation {workload.continuation_len}")

    print("\ntraining demo backbone ...")
    model = train_demo_backbone(workload)

    print("\nendpoint accuracy on the continuation:")
    print(format_accuracy_table(exit_accuracy_by_difficulty(model, workload)))


def _decode_requests(
    tokenizer,
    sequences: torch.Tensor,
    prompt_len: int,
) -> tuple[list[tuple[str, str]], list[int], list[str]]:
    """Decodes requests back to text and measures the continuations in bytes.

    Per-token loss is not comparable across tokenizers. A tokenizer that splits
    the same string into more pieces earns a lower average loss per piece without
    predicting anything better, so comparing a GPT-2-tokenized model against a
    GPT-NeoX-tokenized one on per-token NLL rewards whichever tokenizer is
    finer-grained. Bits per byte removes the tokenizer from the denominator:

    .. code-block:: text

        bits_per_byte = nll_sum_nats / (ln(2) * utf8_bytes(continuation))

    Args:
        tokenizer: Tokenizer that produced the ids.
        sequences: Token ids shaped ``(n_requests, prompt_len + continuation)``.
        prompt_len: Where the continuation starts.

    Returns:
        A tuple ``(texts, byte_lengths, hashes)``. ``texts`` holds
        ``(prompt, continuation)`` string pairs so a model from another family
        can re-encode the identical string; ``byte_lengths`` is the UTF-8 length
        of each continuation; ``hashes`` digests each continuation so two models
        can prove they scored the same content.
    """
    import hashlib

    texts: list[tuple[str, str]] = []
    byte_lengths: list[int] = []
    hashes: list[str] = []

    for row in sequences.tolist():
        prompt = tokenizer.decode(row[:prompt_len])
        continuation = tokenizer.decode(row[prompt_len:])
        encoded = continuation.encode("utf-8")
        texts.append((prompt, continuation))
        byte_lengths.append(len(encoded))
        hashes.append(hashlib.sha256(encoded).hexdigest()[:32])

    return texts, byte_lengths, hashes
