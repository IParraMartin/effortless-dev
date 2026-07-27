"""Measuring how early-exit error compounds over a long generation.

Each token that exits early leaves approximate keys and values behind it. Every
later token attends to that approximation, exits on the strength of it, and
leaves its own approximation in turn. The error is therefore not a fixed
per-token cost but something that accumulates along the sequence, and a
threshold that looks harmless on a short continuation may not stay harmless on
a long one.

Two questions are separated here, because they are usually confounded:

    * **Drift.** Holding the token sequence fixed, how far do the model's
      predictions move as the cache fills with propagated entries? This is
      measured by :func:`measure_drift`, which feeds identical context to both
      the exact and the early-exit path, so the only difference is the cache.
    * **Divergence.** Left to generate freely, how many tokens pass before the
      early-exit model says something different? That is
      :func:`divergence_point`, and it is the number a user would notice.

Drift is the cleaner signal and divergence is the one that matters; a mitigation
worth having should improve both.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.model import KVCache, Transformer


@dataclass
class DriftProfile:
    """Position-by-position record of how far early exiting has strayed.

    Attributes:
        kl: Divergence from the exact next-token distribution at each position,
            shaped ``(batch, seq_len)``, in nats.
        exit_layers: Depth each position stopped at, same shape.
        agreement: Whether the early-exit argmax matched the exact argmax, same
            shape.
        cache_error: Relative error of the cached keys at each layer, shaped
            ``(n_layers,)``. Layer zero is always exact; the error grows with
            depth because deeper layers are further from any exit point.
        refresh_every: Refresh period this profile was measured under.
        threshold: Exit threshold this profile was measured under.
    """

    kl: torch.Tensor
    exit_layers: torch.Tensor
    agreement: torch.Tensor
    cache_error: torch.Tensor
    refresh_every: int
    threshold: float

    @property
    def mean_kl(self) -> float:
        """Average divergence across every position."""
        return float(self.kl.mean())

    @property
    def mean_exit_layer(self) -> float:
        """Average stopping layer *index* across every position."""
        return float(self.exit_layers.float().mean())

    @property
    def mean_exit_depth(self) -> float:
        """Average number of blocks executed, which is ``mean_exit_layer + 1``."""
        return self.mean_exit_layer + 1.0

    @property
    def agreement_rate(self) -> float:
        """Fraction of positions whose greedy prediction was unchanged."""
        return float(self.agreement.float().mean())


@torch.no_grad()
def measure_drift(
    model: Transformer,
    input_ids: torch.Tensor,
    threshold: float | None = None,
    refresh_every: int = 0,
) -> DriftProfile:
    """Compares early-exit decoding against the exact path on fixed context.

    Both paths consume exactly the same tokens, so neither can wander off onto
    a different continuation. Whatever separates them is attributable to the
    cache holding propagated keys and values rather than real ones.

    Args:
        model: The model to measure. Not modified.
        input_ids: Token ids shaped ``(batch, seq_len)``.
        threshold: Exit threshold, defaulting to the model's configured one.
        refresh_every: Force a full-depth token this often. ``0`` disables it.

    Returns:
        A :class:`DriftProfile` for this setting.
    """
    was_training = model.training
    model.eval()

    threshold = model.config.exit_threshold if threshold is None else threshold

    # Exact reference: one full-depth pass over the whole sequence.
    reference = model(input_ids).logits
    exact_cache = KVCache(model.config.n_layers)
    model(input_ids, cache=exact_cache)

    approx_logits, exit_layers = model.trace_decode(
        input_ids, threshold=threshold, refresh_every=refresh_every
    )

    reference_logprobs = F.log_softmax(reference.float(), dim=-1)
    approx_logprobs = F.log_softmax(approx_logits.float(), dim=-1)
    kl = (
        reference_logprobs.exp() * (reference_logprobs - approx_logprobs)
    ).sum(dim=-1)

    # Rebuild the cache the early-exit path actually produced, so its contents
    # can be compared entry by entry against the exact one.
    approx_cache = KVCache(model.config.n_layers)
    _replay_cache(model, input_ids, approx_cache, threshold, refresh_every)

    errors = []
    for layer in range(model.config.n_layers):
        exact_keys = exact_cache.keys[layer]
        approx_keys = approx_cache.keys[layer]
        scale = exact_keys.pow(2).sum().clamp(min=1e-9)
        errors.append(((approx_keys - exact_keys).pow(2).sum() / scale).sqrt())

    if was_training:
        model.train()

    return DriftProfile(
        kl=kl,
        exit_layers=exit_layers,
        agreement=reference.argmax(-1) == approx_logits.argmax(-1),
        cache_error=torch.stack(errors),
        refresh_every=refresh_every,
        threshold=threshold,
    )


@torch.no_grad()
def _replay_cache(
    model: Transformer,
    input_ids: torch.Tensor,
    cache: KVCache,
    threshold: float,
    refresh_every: int,
) -> None:
    """Fills a cache exactly as :meth:`Transformer.trace_decode` would.

    Args:
        model: The model to run.
        input_ids: Token ids shaped ``(batch, seq_len)``.
        cache: Cache to populate in place.
        threshold: Exit threshold.
        refresh_every: Refresh period, or ``0``.
    """
    model(input_ids[:, :1], cache=cache)
    for position in range(1, input_ids.size(1)):
        model._decode_step(
            input_ids[:, position : position + 1],
            cache,
            model._threshold_at(position, threshold, refresh_every),
        )


def bucket_by_position(
    profile: DriftProfile,
    n_buckets: int = 8,
) -> list[tuple[int, int, float, float, float]]:
    """Summarizes a profile in equal slices along the sequence.

    Compounding shows up as a trend across these rows. A flat profile means the
    approximation is costing a constant amount per token; a rising one means
    the error is feeding itself.

    Args:
        profile: The profile to summarize.
        n_buckets: Number of slices.

    Returns:
        Tuples of ``(start, end, mean_kl, agreement, mean_exit_layer)``, one per
        slice, with ``end`` exclusive.
    """
    length = profile.kl.size(1)
    width = max(length // n_buckets, 1)

    rows = []
    for start in range(0, length, width):
        end = min(start + width, length)
        rows.append(
            (
                start,
                end,
                float(profile.kl[:, start:end].mean()),
                float(profile.agreement[:, start:end].float().mean()),
                float(profile.exit_layers[:, start:end].float().mean()),
            )
        )
    return rows


def format_drift(profile: DriftProfile, n_buckets: int = 8) -> str:
    """Renders a drift profile as a table.

    Args:
        profile: The profile to render.
        n_buckets: Number of position slices.

    Returns:
        A multi-line string.
    """
    header = (
        f"{'positions':>14}  {'mean KL':>9}  {'agreement':>10}  {'mean depth':>11}"
    )
    lines = [header, "-" * len(header)]
    for start, end, kl, agreement, layer in bucket_by_position(profile, n_buckets):
        lines.append(
            f"{f'{start}-{end}':>14}  {kl:>9.4f}  {agreement:>10.3f}  "
            f"{layer + 1.0:>11.2f}"
        )

    layers = " ".join(f"L{i}:{float(e):.3f}" for i, e in enumerate(profile.cache_error))
    lines.append(f"\ncache key error by layer: {layers}")
    return "\n".join(lines)


@torch.no_grad()
def divergence_point(
    model: Transformer,
    prompts: torch.Tensor,
    max_new_tokens: int,
    threshold: float | None = None,
    refresh_every: int = 0,
) -> torch.Tensor:
    """Finds where free generation first departs from the exact path.

    Both runs decode greedily so the comparison is deterministic and any
    difference is caused by early exiting rather than by sampling noise.

    Args:
        model: The model to measure.
        prompts: Prompt token ids shaped ``(batch, prompt_len)``.
        max_new_tokens: Tokens to generate.
        threshold: Exit threshold, defaulting to the configured one.
        refresh_every: Refresh period, or ``0``.

    Returns:
        Index of the first differing generated token per row, shaped
        ``(batch,)``. Rows that never diverged hold ``max_new_tokens``.
    """
    exact = model.generate(
        prompts, max_new_tokens, temperature=0.0, threshold=0.0, refresh_every=0
    ).sequences[:, prompts.size(1):]
    approx = model.generate(
        prompts,
        max_new_tokens,
        temperature=0.0,
        threshold=threshold,
        refresh_every=refresh_every,
    ).sequences[:, prompts.size(1):]

    differs = exact != approx
    # argmax finds the first True; rows that never differ get the full length.
    first = differs.float().argmax(dim=1)
    return torch.where(differs.any(dim=1), first, torch.full_like(first, approx.size(1)))


def compare_refresh(
    model: Transformer,
    input_ids: torch.Tensor,
    periods: list[int],
    threshold: float | None = None,
) -> str:
    """Measures what periodic full-depth tokens buy, at what cost.

    Args:
        model: The model to measure.
        input_ids: Token ids shaped ``(batch, seq_len)``.
        periods: Refresh periods to try. ``0`` means no refreshing.
        threshold: Exit threshold held fixed across the comparison.

    Returns:
        A multi-line table.

    Warning:
        Read this together with :func:`refresh_vs_threshold`. Refreshing raises
        the average depth, so quality improves here partly because more compute
        is being spent. This table alone cannot say whether refreshing beats
        simply lowering the threshold.
    """
    header = (
        f"{'refresh':>9}  {'mean KL':>9}  {'agreement':>10}  "
        f"{'mean depth':>11}  {'deepest cache err':>18}"
    )
    lines = [header, "-" * len(header)]

    for period in periods:
        profile = measure_drift(model, input_ids, threshold, refresh_every=period)
        label = "off" if period == 0 else f"every {period}"
        lines.append(
            f"{label:>9}  {profile.mean_kl:>9.4f}  "
            f"{profile.agreement_rate:>10.3f}  {profile.mean_exit_depth:>11.2f}  "
            f"{float(profile.cache_error[-1]):>18.4f}"
        )
    return "\n".join(lines)


def refresh_vs_threshold(
    model: Transformer,
    input_ids: torch.Tensor,
    periods: list[int],
    threshold: float,
    candidate_thresholds: list[float],
) -> str:
    """Tests whether refreshing beats just exiting later, at equal compute.

    Refreshing is only interesting if it is a *better use of depth* than
    lowering the threshold. Both spend compute and both improve quality, so
    comparing them at their natural operating points proves nothing.

    Each refresh setting is therefore matched against the plain setting that
    reaches the same average depth, found by sweeping thresholds, and the two
    are compared at that matched cost. A positive margin means the compute is
    better spent on periodic exact anchors than on uniformly deeper exits.

    Args:
        model: The model to measure.
        input_ids: Token ids shaped ``(batch, seq_len)``.
        periods: Refresh periods to evaluate. ``0`` is skipped as trivial.
        threshold: Exit threshold used alongside refreshing.
        candidate_thresholds: Thresholds forming the no-refresh reference
            curve. Should span from strict to permissive.

    Returns:
        A multi-line table, one row per refresh period.
    """
    reference = []
    for candidate in candidate_thresholds:
        profile = measure_drift(model, input_ids, candidate, refresh_every=0)
        reference.append(
            (profile.mean_exit_depth, profile.agreement_rate, profile.mean_kl,
             candidate)
        )

    header = (
        f"{'refresh':>9}  {'mean depth':>11}  {'agreement':>10}  "
        f"{'matched thr':>12}  {'its agreement':>14}  {'margin':>8}"
    )
    lines = [header, "-" * len(header)]

    for period in periods:
        if period == 0:
            continue
        profile = measure_drift(model, input_ids, threshold, refresh_every=period)
        depth = profile.mean_exit_depth

        # The no-refresh setting that spends the same average depth.
        match = min(reference, key=lambda row: abs(row[0] - depth))
        margin = profile.agreement_rate - match[1]
        lines.append(
            f"{f'every {period}':>9}  {depth:>11.2f}  "
            f"{profile.agreement_rate:>10.3f}  {match[3]:>12.2f}  "
            f"{match[1]:>14.3f}  {margin:>+8.3f}"
        )
    return "\n".join(lines)
