"""Choosing the exit threshold.

``exit_threshold`` decides how much accuracy is traded for how much saved
compute, and there is no principled default: it depends on the model, the
tokenizer, and the data. This module measures the tradeoff so the number can be
picked from a curve rather than guessed.

The measurement reuses :class:`~model.ExitStatistics`, which records every
exit's uncertainty and correctness in one full-depth pass. Because those
statistics are already reduced over the vocabulary, an entire sweep costs one
forward pass regardless of how many thresholds are tried.

Caveat, and it matters: the sweep replays a full-depth pass, so every exit saw
*exact* keys and values from the layers below. Real early-exit generation feeds
propagated states into those layers instead. The numbers here are therefore an
upper bound on quality. Use the sweep to narrow the range, then confirm the
chosen threshold with actual generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.model import ExitStatistics


@dataclass
class SweepPoint:
    """What one threshold would have produced.

    Two different numbers are easy to confuse here, so both are carried
    explicitly. A token that stops at layer *index* ``L`` has executed ``L + 1``
    blocks, and it is the executed count — the *depth* — that compute is
    proportional to. Everything user-facing reports depth; the layer index is
    retained only because it is what indexes :attr:`ExitStatistics.exit_layers`.

    Attributes:
        threshold: The uncertainty cutoff evaluated.
        mean_exit_depth: Average number of blocks executed per position, in
            ``[1, n_layers]``.
        accuracy: Fraction of positions whose greedy prediction was correct.
        nll: Mean negative log-likelihood of the true next token.
        compute_saved: Fraction of block evaluations avoided relative to always
            running the full stack.
        exit_rate: Fraction of positions stopping at each exit, ordered as
            :attr:`ExitStatistics.exit_layers`.
    """

    threshold: float
    mean_exit_depth: float
    accuracy: float
    nll: float
    compute_saved: float
    exit_rate: tuple[float, ...]

    @property
    def mean_exit_layer(self) -> float:
        """Average stopping layer *index*, one less than the executed depth."""
        return self.mean_exit_depth - 1.0


def resolve_exits(
    stats: ExitStatistics,
    threshold: float,
    min_exit_layer: int = 0,
) -> torch.Tensor:
    """Finds where each position would stop at a given threshold.

    Args:
        stats: Statistics from :meth:`model.Transformer.exit_statistics`.
        threshold: Positions stop at the first exit whose uncertainty falls
            below this.
        min_exit_layer: Exits shallower than this are not allowed to fire.

    Returns:
        Index into ``stats.exit_layers`` for each position, shaped
        ``(batch, seq_len)``. Positions that never grew confident resolve to
        the final exit.
    """
    eligible = torch.tensor(
        [layer >= min_exit_layer for layer in stats.exit_layers],
        device=stats.uncertainty.device,
    ).view(-1, 1, 1)
    fired = (stats.uncertainty < threshold) & eligible

    # The last exit is the fallback, so forcing it to fire makes argmax always
    # find something and removes the "nothing fired" special case.
    fired[-1] = True
    return fired.float().argmax(dim=0)


def sweep_thresholds(
    stats: ExitStatistics,
    thresholds: list[float] | tuple[float, ...],
    min_exit_layer: int = 0,
) -> list[SweepPoint]:
    """Evaluates a range of exit thresholds against recorded statistics.

    Args:
        stats: Statistics from :meth:`model.Transformer.exit_statistics`.
        thresholds: Uncertainty cutoffs to evaluate, typically ascending.
        min_exit_layer: Earliest layer permitted to terminate a position.

    Returns:
        One :class:`SweepPoint` per threshold, in the order given.

    Example:
        >>> stats = model.exit_statistics(inputs, targets)
        >>> points = sweep_thresholds(stats, [0.1, 0.3, 0.5])
        >>> print(format_sweep(points, stats))
    """
    layer_of_exit = torch.tensor(
        stats.exit_layers, dtype=torch.float, device=stats.uncertainty.device
    )
    valid = stats.valid
    n_valid = valid.sum().clamp(min=1)
    n_exits = len(stats.exit_layers)

    points = []
    for threshold in thresholds:
        chosen = resolve_exits(stats, threshold, min_exit_layer)

        # gather along the exit dimension to read each position's own exit
        index = chosen.unsqueeze(0)
        correct = stats.correct.gather(0, index).squeeze(0)
        nll = stats.nll.gather(0, index).squeeze(0)

        depth = layer_of_exit[chosen]
        mean_depth = float((depth * valid).sum() / n_valid)

        rates = tuple(
            float(((chosen == i) & valid).sum() / n_valid) for i in range(n_exits)
        )

        points.append(
            SweepPoint(
                threshold=float(threshold),
                # A token stopping at layer L executed L+1 of n_layers blocks.
                mean_exit_depth=mean_depth + 1.0,
                accuracy=float((correct & valid).sum() / n_valid),
                nll=float((nll * valid).sum() / n_valid),
                compute_saved=1.0 - (mean_depth + 1.0) / stats.n_layers,
                exit_rate=rates,
            )
        )
    return points


def format_sweep(points: list[SweepPoint], stats: ExitStatistics) -> str:
    """Renders a sweep as a table.

    Args:
        points: Output of :func:`sweep_thresholds`.
        stats: The statistics the sweep was computed from, used for the depth
            of the model.

    Returns:
        A multi-line string with one row per threshold.
    """
    header = (
        f"{'threshold':>10}  {'mean depth':>11}  {'accuracy':>9}  "
        f"{'nll':>7}  {'saved':>7}"
    )
    lines = [header, "-" * len(header)]
    for point in points:
        lines.append(
            f"{point.threshold:>10.3f}  "
            f"{point.mean_exit_depth:>7.2f}/{stats.n_layers:<3}  "
            f"{point.accuracy:>9.4f}  {point.nll:>7.4f}  "
            f"{point.compute_saved:>6.1%}"
        )
    return "\n".join(lines)


def teacher_forced_top1_agreement_oracle_exact_cache(
    stats: ExitStatistics,
    min_exit_layer: int = 0,
) -> tuple[float, float, float]:
    """Measures how shallow a *perfect* readout-agreement policy could go.

    For each position, finds the shallowest exit already producing the deepest
    exit's greedy token, and averages the depth that implies. Nothing about the
    model changes; only the decision of when to stop is made by an oracle
    instead of by a confidence heuristic.

    The name is deliberately unwieldy, because every qualifier in it is load
    bearing and dropping any one of them overstates the result:

    * **teacher forced** — the token sequence is given, so this never measures
      what free-running generation would do.
    * **top-1 agreement** — sufficiency is defined as reproducing the final
      exit's *argmax for the current token*. Matching the current token does
      not imply the two paths stay together over a continuation, because the
      shallow state also goes into the cache that later tokens read.
    * **exact cache** — it replays a full-depth pass, so every exit saw true
      keys and values. Real early exiting feeds propagated states into those
      layers instead.

    What it does bound, and what makes it worth measuring, is *readout
    redundancy*: how much of the stack is already carrying the final answer,
    token by token. The gap between it and what a threshold achieves is a
    property of the exit policy, and is the part better confidence estimation
    could recover.

    Args:
        stats: Statistics from :meth:`model.Transformer.exit_statistics`.
        min_exit_layer: Earliest layer permitted to terminate a position.

    Returns:
        A tuple ``(mean_exit_depth, compute_saved, early_fraction)``, where the
        first counts executed blocks and the last is the share of positions
        stopped before the final layer.
    """
    eligible = torch.tensor(
        [layer >= min_exit_layer for layer in stats.exit_layers],
        device=stats.uncertainty.device,
    ).view(-1, 1, 1)

    fired = stats.agrees_with_final & eligible
    fired[-1] = True  # the last exit is always available as a fallback

    chosen = fired.float().argmax(dim=0)
    layer_of_exit = torch.tensor(
        stats.exit_layers, dtype=torch.float, device=stats.uncertainty.device
    )
    depth = layer_of_exit[chosen]

    valid = stats.valid
    n_valid = valid.sum().clamp(min=1)
    mean_depth = float((depth * valid).sum() / n_valid) + 1.0
    early = float(((chosen < len(stats.exit_layers) - 1) & valid).sum() / n_valid)

    return mean_depth, 1.0 - mean_depth / stats.n_layers, early


#: Short alias kept so existing call sites and notes still resolve. Prefer the
#: explicit name in anything reported, since the qualifiers are what keep the
#: number from being read as an end-to-end serving result.
oracle_frontier = teacher_forced_top1_agreement_oracle_exact_cache


def recommend_threshold(
    points: list[SweepPoint],
    max_accuracy_drop: float = 0.01,
) -> SweepPoint:
    """Picks the most aggressive threshold that stays accurate enough.

    The baseline is the point with the highest accuracy, which is normally the
    strictest threshold, since running deeper cannot hurt a well-trained stack.

    Args:
        points: Output of :func:`sweep_thresholds`.
        max_accuracy_drop: Absolute accuracy the caller is willing to give up.

    Returns:
        The :class:`SweepPoint` saving the most compute within that budget.

    Raises:
        ValueError: If ``points`` is empty.
    """
    if not points:
        raise ValueError("points must not be empty.")

    best_accuracy = max(point.accuracy for point in points)
    affordable = [
        point
        for point in points
        if point.accuracy >= best_accuracy - max_accuracy_drop
    ]
    return max(affordable, key=lambda point: point.compute_saved)


if __name__ == "__main__":
    from src.config import TransformerConfig
    from src.model import Transformer

    torch.manual_seed(0)

    config = TransformerConfig(
        vocab_size=256, d_model=64, n_layers=6, n_heads=4, n_kv_heads=2,
        max_seq_len=128, min_exit_layer=1,
    )
    model = Transformer(config)

    # The curve only means something if depth actually buys accuracy. A task
    # small enough to memorize is solved perfectly by every exit and produces a
    # flat table, so the corpus here is deliberately larger than the model can
    # fit: the shallow exits run out of capacity and the deep ones pull ahead.
    corpus = torch.randint(0, 256, (256, 65))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(300):
        batch = corpus[torch.randint(0, corpus.size(0), (16,))]
        out = model(batch[:, :-1], targets=batch[:, 1:])
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()
    model.eval()

    corpus = corpus[:32]
    stats = model.exit_statistics(corpus[:, :-1], corpus[:, 1:])
    print("per-exit accuracy:", [
        f"L{layer}={float(stats.correct[i].float().mean()):.2f}"
        for i, layer in enumerate(stats.exit_layers)
    ])
    print()
    points = sweep_thresholds(
        stats, [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0], config.min_exit_layer
    )
    print(format_sweep(points, stats))

    pick = recommend_threshold(points, max_accuracy_drop=0.0)
    print(
        f"\nrecommended threshold {pick.threshold:.3f}: "
        f"{pick.compute_saved:.1%} of block evaluations saved "
        f"at accuracy {pick.accuracy:.4f}"
    )

    depth, saved, early = teacher_forced_top1_agreement_oracle_exact_cache(
        stats, config.min_exit_layer
    )
    print(
        f"teacher_forced_top1_agreement_oracle_exact_cache: {saved:.1%} of "
        f"block evaluations saved at mean depth {depth:.2f}/{config.n_layers}, "
        f"stopping {early:.1%} of tokens early with the final readout unchanged"
    )
    print(
        f"  -> the confidence heuristic captures "
        f"{pick.compute_saved / max(saved, 1e-9):.0%} of that headroom "
        f"(diagnostic only: teacher forced, exact cache, current token)"
    )
