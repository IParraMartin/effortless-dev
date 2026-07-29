"""Deciding whether a difference is real.

The central claim this repository is being built toward — that vertical routing
can stand in for a family of independently trained models — is a *non-inferiority*
claim, and non-inferiority is not established by two point estimates looking
similar. It needs a margin declared in advance and a confidence bound that
clears it. This module supplies the machinery for that, and
:func:`substitution_report` refuses to print the word "replaces" unless the
predeclared test actually passes.

Three choices here are deliberate:

* **Everything is paired by request.** Systems are run on the same inputs, and
  the statistic is the per-request difference. Unpaired comparison throws away
  the request-to-request variation that usually dominates, and on a few hundred
  prompts that is the difference between a detectable effect and noise.
* **Bootstrap rather than a t-test.** Quality metrics here are bounded, skewed,
  or binary, and the estimands include a *ratio of frontier differences* whose
  sampling distribution has no closed form. Resampling handles all of them the
  same way, and it lets the frontier be rebuilt inside every replicate, which
  is the only correct way to put an interval on a frontier-derived quantity.
* **Seeds are a level of the hierarchy, not a footnote.** When a system is
  trained several times, resampling requests alone treats one training run as
  the population. :func:`hierarchical_bootstrap` resamples seeds first, then
  requests within them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

#: Default resamples. Enough for a stable 95% interval; raise it for tail
#: quantities, where the interval is set by far fewer effective observations.
DEFAULT_RESAMPLES = 10_000


@dataclass
class Interval:
    """A point estimate with a bootstrap confidence interval.

    Attributes:
        estimate: The statistic on the observed data.
        low: Lower confidence bound.
        high: Upper confidence bound.
        level: Coverage, for example ``0.95``.
        resamples: Bootstrap replicates behind the bounds.
        clustered: Whether whole clusters were resampled rather than individual
            observations. Recorded because the same interval means different
            things in the two cases, and an unclustered interval over correlated
            observations is too narrow.
        n_clusters: Number of resampling units, when clustered. This is the
            effective sample size the interval rests on, and it can be far below
            the number of observations.
    """

    estimate: float
    low: float
    high: float
    level: float = 0.95
    resamples: int = DEFAULT_RESAMPLES
    clustered: bool = False
    n_clusters: int | None = None

    def __str__(self) -> str:
        unit = f" ({self.n_clusters} clusters)" if self.clustered else ""
        return (
            f"{self.estimate:+.4f} [{self.low:+.4f}, {self.high:+.4f}]{unit}"
        )

    def excludes(self, value: float) -> bool:
        """Whether the interval lies entirely on one side of a value.

        Args:
            value: Reference, usually zero or one.

        Returns:
            ``True`` if ``value`` falls outside the interval.
        """
        return value < self.low or value > self.high


def paired_bootstrap(
    values: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    clusters: Sequence[int] | np.ndarray | None = None,
) -> Interval:
    """Bootstraps a statistic over per-request observations.

    Args:
        values: One observation per request — normally the *difference*
            between two systems on that request, which is what makes the
            comparison paired.
        statistic: Reduction applied to each resample.
        level: Coverage of the interval.
        resamples: Number of replicates.
        seed: Random seed, recorded so an interval is reproducible.
        clusters: Group label per observation, normally the document a request
            was cut from. When given, whole clusters are resampled with
            replacement instead of individual observations.

            This is not a refinement. Requests from one document are correlated,
            and resampling them independently treats each as fresh evidence,
            which makes the interval too narrow — so a predeclared
            non-inferiority test can *pass* on correlation rather than on
            effect. Leaving this ``None`` is only correct when each observation
            is its own cluster.

    Returns:
        The estimate and its percentile interval. ``clustered`` records which
        resampling unit was used, so a reader can tell.

    Raises:
        ValueError: If ``values`` is empty, or if ``clusters`` has a different
            length.
    """
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("values must not be empty.")

    rng = np.random.default_rng(seed)

    if clusters is not None:
        labels = np.asarray(clusters)
        if labels.shape[0] != data.size:
            raise ValueError(
                f"clusters has {labels.shape[0]} labels for {data.size} "
                f"observations; every observation needs exactly one."
            )
        groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
        replicates = np.empty(resamples)
        for replicate in range(resamples):
            chosen = rng.integers(0, len(groups), size=len(groups))
            drawn = np.concatenate([groups[index] for index in chosen])
            replicates[replicate] = statistic(data[drawn])
        tail = (1.0 - level) / 2.0
        return Interval(
            estimate=float(statistic(data)),
            low=float(np.quantile(replicates, tail)),
            high=float(np.quantile(replicates, 1.0 - tail)),
            level=level,
            resamples=resamples,
            clustered=True,
            n_clusters=len(groups),
        )

    indices = rng.integers(0, data.size, size=(resamples, data.size))
    replicates = np.array([statistic(data[row]) for row in indices])

    tail = (1.0 - level) / 2.0
    return Interval(
        estimate=float(statistic(data)),
        low=float(np.quantile(replicates, tail)),
        high=float(np.quantile(replicates, 1.0 - tail)),
        level=level,
        resamples=resamples,
    )


def hierarchical_bootstrap(
    groups: Sequence[Sequence[float]],
    statistic: Callable[[np.ndarray], float] = np.mean,
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """Bootstraps over clusters and then within them.

    Use this when observations come from several training seeds, several
    documents, or a conversation with several turns. Resampling the individual
    observations alone would treat one seed as if it were the population of
    seeds, and would report an interval far narrower than the evidence
    supports.

    Args:
        groups: One sequence of observations per cluster.
        statistic: Reduction applied to the pooled resample.
        level: Coverage of the interval.
        resamples: Number of replicates.
        seed: Random seed.

    Returns:
        The estimate and its percentile interval.

    Raises:
        ValueError: If there are no non-empty clusters.
    """
    clusters = [np.asarray(group, dtype=float) for group in groups]
    clusters = [cluster for cluster in clusters if cluster.size]
    if not clusters:
        raise ValueError("groups must contain at least one non-empty cluster.")

    rng = np.random.default_rng(seed)
    pooled = np.concatenate(clusters)

    replicates = np.empty(resamples)
    for replicate in range(resamples):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        drawn = [
            clusters[index][
                rng.integers(0, clusters[index].size, size=clusters[index].size)
            ]
            for index in chosen
        ]
        replicates[replicate] = statistic(np.concatenate(drawn))

    tail = (1.0 - level) / 2.0
    return Interval(
        estimate=float(statistic(pooled)),
        low=float(np.quantile(replicates, tail)),
        high=float(np.quantile(replicates, 1.0 - tail)),
        level=level,
        resamples=resamples,
    )


def pareto_frontier(
    points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Keeps only the points no other point beats on both axes.

    Args:
        points: ``(cost, quality)`` pairs. Lower cost and higher quality are
            better.

    Returns:
        The non-dominated points, ascending in cost and strictly increasing in
        quality. A point that costs more without being better is dropped, as is
        one that ties on quality at higher cost.
    """
    ordered = sorted(points, key=lambda point: (point[0], -point[1]))

    frontier: list[tuple[float, float]] = []
    best = -np.inf
    for cost, quality in ordered:
        if quality > best:
            frontier.append((cost, quality))
            best = quality
    return frontier


def interpolate_frontier(
    frontier: Sequence[tuple[float, float]],
    cost: float,
) -> float | None:
    """Reads a frontier's quality at a cost, without extrapolating.

    Args:
        frontier: Output of :func:`pareto_frontier`.
        cost: Cost to evaluate at.

    Returns:
        Piecewise-linearly interpolated quality, or ``None`` when ``cost``
        falls outside the measured range. Returning ``None`` rather than the
        nearest endpoint is the point: a frontier says nothing about operating
        points that were never run, and quietly extending it is how a
        comparison at "matched cost" ends up matching nothing.
    """
    if not frontier:
        return None

    costs = np.array([point[0] for point in frontier])
    qualities = np.array([point[1] for point in frontier])

    if cost < costs[0] or cost > costs[-1]:
        return None
    return float(np.interp(cost, costs, qualities))


def common_cost_range(
    *frontiers: Sequence[tuple[float, float]],
) -> tuple[float, float] | None:
    """Finds the cost interval every frontier covers.

    Args:
        *frontiers: Frontiers to intersect.

    Returns:
        ``(low, high)``, or ``None`` if they do not overlap. Comparisons
        outside this range are not comparisons.
    """
    if not frontiers or any(not frontier for frontier in frontiers):
        return None

    low = max(frontier[0][0] for frontier in frontiers)
    high = min(frontier[-1][0] for frontier in frontiers)
    return (low, high) if low <= high else None


#: Smallest horizontal gain for which a substitution ratio is reported. This
#: is a declared threshold, not a numerical guard: with a denominator of a
#: thousandth of a quality point, "recovers 100% of the gain" describes a
#: horizontal system that gained nothing, and the ratio flatters whichever
#: system happens to sit above the other by rounding. Set it in the units of
#: the quality metric being used.
MIN_HORIZONTAL_GAIN = 1e-3


def vertical_substitution_ratio(
    vertical: Sequence[tuple[float, float]],
    horizontal: Sequence[tuple[float, float]],
    baseline_quality: float,
    cost: float,
    min_gain: float = MIN_HORIZONTAL_GAIN,
) -> float | None:
    """How much of horizontal routing's gain vertical routing recovers.

    Defined at a matched cost as the ratio of each system's improvement over a
    declared low-cost baseline. One means vertical routing recovers all of it.

    Args:
        vertical: Vertical frontier.
        horizontal: Horizontal frontier.
        baseline_quality: Quality of the declared cheap baseline.
        cost: Matched cost to evaluate at.
        min_gain: Horizontal gain below which the ratio is withheld, in the
            units of the quality metric.

    Returns:
        The ratio, or ``None`` when either frontier does not reach this cost or
        the horizontal gain is smaller than ``min_gain``. Withholding is the
        honest outcome, not a failure: if the horizontal system barely improves
        on the baseline there is nothing to substitute for, and a ratio
        computed anyway would be dominated by noise in its denominator.
    """
    q_vertical = interpolate_frontier(vertical, cost)
    q_horizontal = interpolate_frontier(horizontal, cost)
    if q_vertical is None or q_horizontal is None:
        return None

    denominator = q_horizontal - baseline_quality
    if abs(denominator) < min_gain:
        return None
    return (q_vertical - baseline_quality) / denominator


def integrated_substitution_ratio(
    vertical: Sequence[tuple[float, float]],
    horizontal: Sequence[tuple[float, float]],
    baseline_quality: float,
    samples: int = 128,
    min_gain: float = MIN_HORIZONTAL_GAIN,
) -> tuple[float, tuple[float, float]] | None:
    """Averages substitution over the whole cost range both systems support.

    A ratio at one cost invites cherry-picking. This integrates the two gains
    separately over the common range and divides once, which is stable where a
    mean of pointwise ratios is not.

    Args:
        vertical: Vertical frontier.
        horizontal: Horizontal frontier.
        baseline_quality: Quality of the declared cheap baseline.
        samples: Trapezoid samples across the range.
        min_gain: Average horizontal gain below which the ratio is withheld.

    Returns:
        ``(ratio, (low_cost, high_cost))``, or ``None`` if the frontiers do not
        overlap or the horizontal gain integrates to nothing.
    """
    span = common_cost_range(vertical, horizontal)
    if span is None or span[0] >= span[1]:
        return None

    grid = np.linspace(span[0], span[1], samples)
    v_gain = np.array(
        [interpolate_frontier(vertical, c) - baseline_quality for c in grid]
    )
    h_gain = np.array(
        [interpolate_frontier(horizontal, c) - baseline_quality for c in grid]
    )

    v_area = float(np.trapezoid(v_gain, grid))
    h_area = float(np.trapezoid(h_gain, grid))
    # Compared as an average gain so the threshold means the same thing here as
    # it does pointwise, rather than scaling with how wide the range happens
    # to be.
    if abs(h_area) / max(span[1] - span[0], 1e-12) < min_gain:
        return None
    return v_area / h_area, span


def bootstrap_substitution_ratio(
    vertical_quality: np.ndarray,
    vertical_cost: np.ndarray,
    horizontal_quality: np.ndarray,
    horizontal_cost: np.ndarray,
    cost: float,
    baseline_quality: float,
    level: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
    min_gain: float = MIN_HORIZONTAL_GAIN,
) -> Interval | None:
    """Puts a confidence interval on the substitution ratio.

    The ratio is a function of two *frontiers*, not of a per-request
    difference, so an interval on it cannot be formed by resampling the ratio's
    inputs independently. Each replicate resamples **requests** and then
    rebuilds both frontiers from scratch inside the replicate, which is what
    propagates the uncertainty in every candidate's mean cost and quality
    through to the ratio. Bootstrapping the endpoints separately and dividing
    afterwards would understate it badly.

    Args:
        vertical_quality: Per-request quality of each vertical candidate,
            shaped ``(n_requests, n_tiers)``.
        vertical_cost: Matching per-request costs.
        horizontal_quality: Per-request quality of each independent model,
            shaped ``(n_requests, n_models)``, on the same requests.
        horizontal_cost: Matching per-request costs.
        cost: Matched cost to evaluate the ratio at.
        baseline_quality: Quality of the declared cheap baseline.
        level: Coverage of the interval.
        resamples: Bootstrap replicates.
        seed: Random seed.
        min_gain: Horizontal gain below which a replicate is discarded.

    Returns:
        The interval, or ``None`` when the ratio is undefined on the observed
        data or on too many replicates to be reported honestly. A ratio that
        exists at the point estimate but collapses under resampling is not a
        result, and returning ``None`` says so rather than quoting a bound
        built from whichever replicates happened to survive.
    """
    def ratio_from(rows: np.ndarray) -> float | None:
        vertical = pareto_frontier(
            list(
                zip(
                    vertical_cost[rows].mean(axis=0),
                    vertical_quality[rows].mean(axis=0),
                )
            )
        )
        horizontal = pareto_frontier(
            list(
                zip(
                    horizontal_cost[rows].mean(axis=0),
                    horizontal_quality[rows].mean(axis=0),
                )
            )
        )
        return vertical_substitution_ratio(
            vertical, horizontal, baseline_quality, cost, min_gain
        )

    n = len(vertical_quality)
    estimate = ratio_from(np.arange(n))
    if estimate is None:
        return None

    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(resamples):
        value = ratio_from(rng.integers(0, n, size=n))
        if value is not None:
            replicates.append(value)

    # If the ratio is undefined on a large minority of replicates, the
    # surviving ones are a biased subsample and any interval from them
    # overstates how well determined the quantity is.
    if len(replicates) < 0.9 * resamples:
        return None

    tail = (1.0 - level) / 2.0
    return Interval(
        estimate=float(estimate),
        low=float(np.quantile(replicates, tail)),
        high=float(np.quantile(replicates, 1.0 - tail)),
        level=level,
        resamples=len(replicates),
    )


def systems_realization_gap(
    theoretical_saving: float,
    measured_saving: float,
) -> dict[str, float]:
    """Compares a compute saving against the latency it actually delivered.

    Args:
        theoretical_saving: ``1 - routed_macs / baseline_macs``.
        measured_saving: ``1 - routed_latency / baseline_latency``.

    Returns:
        The two savings, their difference, and their ratio. A gap near the
        theoretical saving means none of the arithmetic reached the clock — the
        usual outcome when a batch runs every layer for whoever needs it.
    """
    gap = theoretical_saving - measured_saving
    ratio = (
        measured_saving / theoretical_saving
        if abs(theoretical_saving) > 1e-9
        else float("nan")
    )
    return {
        "theoretical_saving": theoretical_saving,
        "measured_saving": measured_saving,
        "realization_gap": gap,
        "realization_ratio": ratio,
    }


@dataclass
class NonInferiorityResult:
    """Outcome of a predeclared substitution test.

    Attributes:
        quality_difference: Paired ``vertical - horizontal`` quality interval.
        cost_difference: Paired ``vertical - horizontal`` cost interval.
        quality_margin: How much quality may be given up.
        cost_tolerance: How much extra cost is acceptable.
        quality_passes: Whether the quality bound cleared the margin.
        cost_passes: Whether the cost bound stayed within tolerance.
        vsr: Substitution ratio interval, when one could be formed.
        vsr_target: Predeclared substitution target.
        vsr_passes: Whether the ratio's lower bound cleared the target.
    """

    quality_difference: Interval
    cost_difference: Interval
    quality_margin: float
    cost_tolerance: float
    quality_passes: bool
    cost_passes: bool
    vsr: Interval | None = None
    vsr_target: float | None = None
    vsr_passes: bool | None = None

    @property
    def substitution_supported(self) -> bool:
        """Whether every declared criterion passed.

        A missing substitution-ratio test does not count as a pass; if a target
        was declared and no ratio could be formed, the claim is unsupported.
        """
        if not (self.quality_passes and self.cost_passes):
            return False
        if self.vsr_target is None:
            return True
        return bool(self.vsr_passes)


def non_inferiority_test(
    quality_vertical: Sequence[float],
    quality_horizontal: Sequence[float],
    cost_vertical: Sequence[float],
    cost_horizontal: Sequence[float],
    quality_margin: float,
    cost_tolerance: float,
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    clusters: Sequence[int] | np.ndarray | None = None,
) -> NonInferiorityResult:
    """Tests whether one system can stand in for another at matched cost.

    Two one-sided conditions, both declared before looking at the data:

    * quality is non-inferior when the lower bound on ``Q_V - Q_H`` exceeds
      ``-quality_margin``;
    * cost is acceptable when the upper bound on ``C_V - C_H`` is at most
      ``cost_tolerance``.

    Note what the quality condition is not. It is not "the interval contains
    zero", which merely means the study was too small to tell the systems
    apart, and would let an underpowered experiment *support* substitution.

    Args:
        quality_vertical: Per-request quality of the vertical system.
        quality_horizontal: Per-request quality of the horizontal system, on
            the same requests in the same order.
        cost_vertical: Per-request cost of the vertical system.
        cost_horizontal: Per-request cost of the horizontal system.
        quality_margin: Largest acceptable quality loss, positive.
        cost_tolerance: Largest acceptable cost increase.
        level: Coverage of the intervals.
        resamples: Bootstrap replicates.
        seed: Random seed.
        clusters: Resampling unit, normally the document id. Omitting it treats
            every request as independent evidence and narrows both intervals,
            which for a one-sided non-inferiority test biases toward passing.

    Returns:
        The populated :class:`NonInferiorityResult`.

    Raises:
        ValueError: If the sequences are not all the same length, which would
            mean the comparison is not paired.
    """
    lengths = {
        len(quality_vertical),
        len(quality_horizontal),
        len(cost_vertical),
        len(cost_horizontal),
    }
    if len(lengths) != 1:
        raise ValueError(
            f"Paired comparison needs equal-length inputs, got lengths "
            f"{sorted(lengths)}. Systems must be evaluated on the same "
            f"requests in the same order."
        )

    quality_delta = np.asarray(quality_vertical, float) - np.asarray(
        quality_horizontal, float
    )
    cost_delta = np.asarray(cost_vertical, float) - np.asarray(
        cost_horizontal, float
    )

    quality_interval = paired_bootstrap(
        quality_delta, level=level, resamples=resamples, seed=seed,
        clusters=clusters,
    )
    cost_interval = paired_bootstrap(
        cost_delta, level=level, resamples=resamples, seed=seed + 1,
        clusters=clusters,
    )

    return NonInferiorityResult(
        quality_difference=quality_interval,
        cost_difference=cost_interval,
        quality_margin=quality_margin,
        cost_tolerance=cost_tolerance,
        quality_passes=quality_interval.low > -quality_margin,
        cost_passes=cost_interval.high <= cost_tolerance,
    )


def substitution_report(result: NonInferiorityResult) -> str:
    """Renders a substitution test, and states the verdict only if earned.

    Args:
        result: Output of :func:`non_inferiority_test`.

    Returns:
        A multi-line summary ending in an explicit verdict.
    """
    lines = [
        f"quality  Q_V - Q_H = {result.quality_difference}  "
        f"(margin -{result.quality_margin:.4f}) "
        f"{'PASS' if result.quality_passes else 'FAIL'}",
        f"cost     C_V - C_H = {result.cost_difference}  "
        f"(tolerance {result.cost_tolerance:.4f}) "
        f"{'PASS' if result.cost_passes else 'FAIL'}",
    ]
    if result.vsr_target is not None:
        state = "PASS" if result.vsr_passes else "FAIL"
        shown = result.vsr if result.vsr is not None else "unavailable"
        lines.append(
            f"VSR      {shown}  (target {result.vsr_target:.2f}) {state}"
        )

    if result.substitution_supported:
        lines.append(
            "verdict  substitution supported at the predeclared margins"
        )
    else:
        lines.append(
            "verdict  substitution NOT supported; report the numbers, not a "
            "claim of equivalence"
        )
    return "\n".join(lines)
