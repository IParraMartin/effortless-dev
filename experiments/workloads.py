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
        return Workload(
            prompts=self.prompts.index_select(0, index),
            references=self.references.index_select(0, index),
            difficulty=[self.difficulty[row] for row in rows],
            source_ids=[self.source_ids[row] for row in rows],
            spec=dict(self.spec),
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
