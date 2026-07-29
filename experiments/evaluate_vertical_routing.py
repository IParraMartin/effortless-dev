"""The central comparison: one elastic backbone against a family of models.

Every system below is evaluated on the *same* requests under the *same*
decoding settings, and every difference is computed per request before being
averaged. That is what makes the confidence intervals meaningful; an unpaired
comparison throws away the request-to-request variation that dominates at these
sample sizes.

Systems evaluated
-----------------

Vertical, all from one backbone:

* full depth — the reference;
* every fixed endpoint — isolates *endpoint quality* from *routing quality*,
  and is the baseline a router has to beat;
* the entropy threshold — the token-level heuristic, retained because it is
  what the prior literature uses;
* the learned request-level router;
* the request-level outcome oracle over the same tiers — a diagnostic, not a
  target: it chooses by looking at realized outcomes, so no policy can reach it;
* a cross-fitted probe policy — one specified learner over the probe features,
  which is what a controller should actually be measured against;
* the best static randomized mixture at matched average cost. This one is easy
  to leave out and is the one that hurts: flipping a weighted coin between two
  fixed depths already produces a curve, so a router that merely lands on that
  curve has demonstrated nothing beyond knowing its own average.

Horizontal, from independently trained checkpoints supplied by a manifest:

* each independent model;
* a horizontal router;
* the horizontal oracle;
* a small-to-large cascade, charged for the small model's run on requests that
  escalate.

The two-tier ``vertical_no_reuse_oracle_cascade`` is **not** in the horizontal
family despite its shape. Both cascade arms take their qualities from one
backbone's depth endpoints and escalate on an outcome-based rule, so the pair
measures what prefix reuse is worth and nothing about independent models.

The manifest keeps this repository from having to train a model family to be
useful. Without one, the horizontal side is reported as absent rather than
filled in with the vertical numbers under another name.

Run it::

    python -m experiments.evaluate_vertical_routing \\
        --trajectories results/trajectories --controller results/controller
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.routing import DepthController
from experiments.collect_depth_trajectories import load
from experiments.train_depth_controller import (
    COST_METRICS,
    QUALITY_METRICS,
    choose,
    normalized_cost,
    per_tier_utility,
    quality_matrix,
    split_rows,
)
from utils.provenance import RunRecord
from utils.statistics import (
    bootstrap_substitution_ratio,
    common_cost_range,
    integrated_substitution_ratio,
    non_inferiority_test,
    paired_bootstrap,
    pareto_frontier,
    substitution_report,
    vertical_substitution_ratio,
)

#: Version of the evaluation record layout.
SCHEMA_VERSION = 2


@dataclass
class EvaluationConfig:
    """Settings for one evaluation.

    Attributes:
        trajectories: Directory of collected trajectories.
        controller: Directory of trained controllers, or ``None`` to evaluate
            only the systems that need no controller.
        controller_seed: Exact controller fit to evaluate. Selecting a seed by
            filename order after seeing results is not a predeclared policy.
        manifest: JSON manifest describing independently trained models. Absent
            means the horizontal side is not evaluated, and is reported as not
            evaluated rather than quietly skipped.
        out: Directory for results.
        quality_metric: Quality column to report.
        cost_metric: Cost column to report.
        lambdas: Prices of compute to sweep, tracing the frontier.
        quality_margin: Predeclared non-inferiority margin on quality.
        cost_tolerance: Predeclared tolerance on cost.
        vsr_target: Predeclared substitution target.
        resamples: Bootstrap replicates.
        seed: Bootstrap seed.
    """

    trajectories: str = "results/trajectories"
    controller: str | None = "results/controller"
    controller_seed: int = 0
    manifest: str | None = None
    out: str = "results/evaluation"
    quality_metric: str = "teacher_forced_accuracy"
    cost_metric: str = "cost_depth_fraction"
    lambdas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
    quality_margin: float = 0.02
    cost_tolerance: float = 0.0
    vsr_target: float = 0.9
    resamples: int = 2000
    seed: int = 0


@dataclass
class SystemResult:
    """One system at one operating point.

    Attributes:
        name: System name.
        family: ``"vertical"``, ``"horizontal"``, or ``"reference"``.
        operating_point: The price of compute, or a tier label.
        quality: Per-request quality.
        cost: Per-request normalized cost.
        choices: Per-request tier or model index, where one was chosen.
        notes: Anything a reader must know to interpret the row.
    """

    name: str
    family: str
    operating_point: float | str
    quality: np.ndarray
    cost: np.ndarray
    choices: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def mean_quality(self) -> float:
        """Average quality across requests."""
        return float(self.quality.mean())

    @property
    def mean_cost(self) -> float:
        """Average cost across requests."""
        return float(self.cost.mean())

    def point(self) -> tuple[float, float]:
        """The ``(cost, quality)`` pair this system contributes to a frontier."""
        return self.mean_cost, self.mean_quality


def request_clusters(records: list[dict]) -> np.ndarray | None:
    """Resolves the unit every interval should be resampled over.

    Args:
        records: Trajectory records for the rows being evaluated.

    Returns:
        One cluster label per record — the document each request was cut from —
        or ``None`` when the records predate schema 2 and carry no document
        identity.

        ``None`` is returned rather than silently substituting ``request_id``,
        because those two produce different intervals and the caller has to be
        able to say which it got. Falling back quietly would report an
        unclustered interval under a clustered label.
    """
    if not records:
        return None
    labels = [record.get("document_id", -1) for record in records]
    if any(label is None or label < 0 for label in labels):
        return None
    return np.asarray(labels)


def load_controller(path: str | Path) -> tuple[DepthController, dict]:
    """Restores a controller saved by the trainer.

    Args:
        path: Controller checkpoint file.

    Returns:
        A tuple ``(controller, blob)``.

    Raises:
        ValueError: If the checkpoint predates or postdates this reader.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    version = blob.get("schema_version", 0)
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"{path} uses controller schema {version}, newer than this reader."
        )

    controller = DepthController(
        d_model=blob["feature_dim"],
        n_tiers=blob["n_tiers"],
        hidden_dim=blob["hidden_dim"],
        output=blob["output"],
        input_dim=blob["feature_dim"],
    )
    controller.load_state_dict(blob["state_dict"])
    # Stamp the cost metric the controller was selected under, so a later
    # select() without a cost vector fails instead of silently substituting a
    # depth fraction for whatever this was validated against.
    controller.cost_metric = (
        blob.get("train_config", {}) or {}
    ).get("cost_metric")
    controller.eval()
    return controller, blob


def controller_report_rows(records: list[dict], blob: dict) -> list[int]:
    """Resolves the untouched reporting split stored with a controller.

    Args:
        records: Trajectory records containing unique ``request_id`` values.
        blob: Loaded controller checkpoint.

    Returns:
        Row indices in the controller's recorded reporting order.

    Raises:
        ValueError: If the checkpoint predates split identities, references
            absent requests, duplicates request ids, or includes non-validation
            examples.
    """
    split_ids = blob.get("split_request_ids", {})
    report_ids = split_ids.get("report")
    if report_ids is None:
        raise ValueError(
            "Controller checkpoint does not record its reporting request ids. "
            "Refit it with controller schema 2 before evaluating; using the "
            "whole validation split would mix calibration and reporting "
            "examples."
        )

    by_request_id: dict[int, int] = {}
    for row, record in enumerate(records):
        request_id = record["request_id"]
        if request_id in by_request_id:
            raise ValueError(f"Duplicate trajectory request_id {request_id}.")
        by_request_id[request_id] = row

    missing = [request_id for request_id in report_ids
               if request_id not in by_request_id]
    if missing:
        raise ValueError(
            f"Controller reporting split contains {len(missing)} request "
            f"id(s) absent from these trajectories, starting with "
            f"{missing[:3]}."
        )

    rows = [by_request_id[request_id] for request_id in report_ids]
    if any(records[row].get("split") != "validation" for row in rows):
        raise ValueError(
            "Controller reporting ids include non-validation requests."
        )
    return rows


def fixed_endpoints(
    quality: np.ndarray,
    cost: np.ndarray,
    tiers: list[int],
) -> list[SystemResult]:
    """Evaluates every fixed depth.

    Args:
        quality: Quality shaped ``(n_requests, n_tiers)``.
        cost: Normalized cost, same shape.
        tiers: Candidate depths.

    Returns:
        One result per tier.
    """
    return [
        SystemResult(
            name=f"fixed_depth_{tier}",
            family="vertical",
            operating_point=f"depth_{tier}",
            quality=quality[:, index],
            cost=cost[:, index],
            choices=np.full(len(quality), index),
        )
        for index, tier in enumerate(tiers)
    ]


def oracle_system(
    quality: np.ndarray,
    cost: np.ndarray,
    routing_lambda: float,
    family: str = "vertical",
) -> SystemResult:
    """Evaluates the per-request oracle at one price of compute.

    Args:
        quality: Quality shaped ``(n_requests, n_candidates)``.
        cost: Normalized cost, same shape.
        routing_lambda: Price of a unit of cost.
        family: Which side of the comparison this oracle belongs to.

    Returns:
        The oracle's result. Unattainable by construction — choosing the best
        endpoint requires having run them all — but it bounds every policy over
        the same candidates, which is what makes router regret interpretable.
    """
    index = per_tier_utility(quality, cost, routing_lambda).argmax(axis=1)
    rows = np.arange(len(index))
    return SystemResult(
        name=f"{family}_oracle",
        family=family,
        operating_point=routing_lambda,
        quality=quality[rows, index],
        cost=cost[rows, index],
        choices=index,
        notes=["unattainable: requires having evaluated every candidate"],
    )


def cross_fitted_probe_policy(
    features: np.ndarray,
    quality: np.ndarray,
    cost: np.ndarray,
    routing_lambda: float,
    folds: int = 5,
    hidden: int = 128,
    steps: int = 600,
    seed: int = 0,
) -> SystemResult:
    """One reference policy fitted from the probe features, out of fold.

    The outcome oracle takes ``max`` over candidates per request, which requires
    knowing how each one turned out. No deployable policy can have that. Quoting
    it as the target for a learned router charges the router for information it
    was never given, and the difference is not a small correction: on this
    repository's own workload the outcome oracle reported +0.051 of headroom
    while this policy attained **+0.008**, so 85% of that "gain" required the
    answer. A router measured against the oracle looks broken when it is
    performing near optimally.

    So this is a second, information-respecting comparator. A deliberately
    over-powered per-tier quality regressor — larger than the controller under
    test — is fitted on the probe features by K-fold cross-fitting, so its
    prediction for a request never saw that request. Choosing by predicted
    utility and scoring by true quality gives a policy that obeys the same
    information constraint as the controller.

    **This is not a ceiling and must not be called one.** It is the out-of-fold
    performance of one specified learner from one model class. Another model
    class can beat it, in which case the controller's "regret" against it goes
    negative — a number that is meaningless if the comparator was described as a
    bound. The theoretical best policy measurable from these features
    (``feature_bayes_value`` in the naming convention) is not computed here and
    is generally unknown.

    Read it alongside the outcome oracle rather than instead of it. The gap
    between them is the part of the headroom that depends on knowing the answer;
    the gap between this and the best fixed endpoint is what a learner of this
    class actually extracted.

    Args:
        features: Probe features shaped ``(n_requests, feature_dim)``.
        quality: Quality shaped ``(n_requests, n_tiers)``.
        cost: Normalized cost, same shape.
        routing_lambda: Price of a unit of cost.
        folds: Cross-fitting folds.
        hidden: Width of the regressor, intentionally generous.
        steps: Optimizer steps per fold.
        seed: Random seed for the fold split and initialization.

    Returns:
        The policy's result, with its choices.
    """
    import torch.nn.functional as F

    n, n_tiers = quality.shape
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    predicted = np.zeros_like(quality)

    for fold in range(folds):
        held = order[fold::folds]
        fitted = np.setdiff1d(order, held)
        if not len(fitted) or not len(held):
            continue

        torch.manual_seed(seed)
        train_x = torch.tensor(features[fitted], dtype=torch.float32)
        mean, deviation = train_x.mean(0), train_x.std(0) + 1e-6

        net = torch.nn.Sequential(
            torch.nn.Linear(features.shape[1], hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, n_tiers),
        )
        optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        target = torch.tensor(quality[fitted], dtype=torch.float32)

        for _ in range(steps):
            loss = F.mse_loss(net((train_x - mean) / deviation), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            held_x = torch.tensor(features[held], dtype=torch.float32)
            predicted[held] = net((held_x - mean) / deviation).numpy()

    index = (predicted - routing_lambda * cost).argmax(axis=1)
    rows = np.arange(n)
    return SystemResult(
        name="cross_fitted_probe_policy",
        family="vertical",
        operating_point=routing_lambda,
        quality=quality[rows, index],
        cost=cost[rows, index],
        choices=index,
        notes=[
            "out-of-fold performance of one specified learner on the probe "
            "features; an information-respecting comparator, not a ceiling -- "
            "another model class can beat it"
        ],
    )


def learned_router(
    controller: DepthController,
    features: np.ndarray,
    quality: np.ndarray,
    cost: np.ndarray,
    routing_lambda: float,
    threshold: float = 0.5,
) -> SystemResult:
    """Evaluates the learned request-level router at one price of compute.

    Args:
        controller: Trained controller.
        features: Probe features.
        quality: Quality shaped ``(n_requests, n_tiers)``.
        cost: Normalized cost, same shape.
        routing_lambda: Price of a unit of cost.
        threshold: Sufficiency cutoff for the ordinal head.

    Returns:
        The router's result.
    """
    index = choose(controller, features, cost, routing_lambda, threshold)
    rows = np.arange(len(index))
    return SystemResult(
        name="learned_request_router",
        family="vertical",
        operating_point=routing_lambda,
        quality=quality[rows, index],
        cost=cost[rows, index],
        choices=index,
    )


def best_static_mixture(
    quality: np.ndarray,
    cost: np.ndarray,
    target_cost: float,
    seed: int = 0,
) -> SystemResult | None:
    """The best coin-flip between two fixed depths at a matched average cost.

    This is the baseline that deflates a weak router. Randomizing between two
    endpoints traces the straight line joining them, so it already produces a
    quality-cost curve without looking at the request at all. A router only
    demonstrates *adaptivity* by beating this line, not by landing on it.

    Args:
        quality: Quality shaped ``(n_requests, n_tiers)``.
        cost: Normalized cost, same shape.
        target_cost: Average cost to match.
        seed: Retained for command-line compatibility. The returned arrays use
            the exact expectation of the randomized policy, so no assignment
            draw (and therefore no Monte Carlo seed noise) is needed.

    Returns:
        The best mixture reaching that cost, or ``None`` if no pair brackets it.
    """
    mean_quality = quality.mean(axis=0)
    mean_cost = cost.mean(axis=0)
    del seed

    best: tuple[float, int, int, float] | None = None
    for low in range(quality.shape[1]):
        for high in range(quality.shape[1]):
            span = mean_cost[high] - mean_cost[low]
            if abs(span) < 1e-12:
                continue
            weight = (target_cost - mean_cost[low]) / span
            if not 0.0 <= weight <= 1.0:
                continue

            value = (1 - weight) * mean_quality[low] + weight * mean_quality[high]
            if best is None or value > best[0]:
                best = (value, low, high, float(weight))

    if best is None:
        return None

    _, low, high, weight = best
    return SystemResult(
        name="best_static_mixture",
        family="vertical",
        operating_point=float(target_cost),
        quality=(1.0 - weight) * quality[:, low] + weight * quality[:, high],
        cost=(1.0 - weight) * cost[:, low] + weight * cost[:, high],
        choices=None,
        notes=[
            "request-independent randomized mixture, reported in expectation: "
            f"tier-index {low} with probability {1.0 - weight:.6f}, "
            f"tier-index {high} with probability {weight:.6f}"
        ],
    )


def load_manifest(path: str | Path) -> list[dict]:
    """Reads a manifest of independently trained models.

    The manifest exists so the horizontal side can be supplied by whoever
    trained the family, rather than requiring this repository to train one. It
    is validated on the way in, because a silently mismatched tokenizer or a
    missing cost would produce a comparison that looks fine and means nothing.

    Args:
        path: JSON file holding a list of model entries.

    Returns:
        The entries, each with ``model_id``, ``tokenizer_id``, ``tier``,
        ``cost``, and ``results`` naming a file of per-request quality.

    Raises:
        ValueError: If an entry is missing a required field, or if the entries
            disagree about the tokenizer.
    """
    entries = json.loads(Path(path).read_text())
    required = {"model_id", "tokenizer_id", "tier", "cost", "results"}

    for entry in entries:
        missing = required - set(entry)
        if missing:
            raise ValueError(
                f"Manifest entry {entry.get('model_id', '?')} is missing "
                f"{sorted(missing)}."
            )

    tokenizers = {entry["tokenizer_id"] for entry in entries}
    if len(tokenizers) > 1:
        raise ValueError(
            f"Manifest mixes tokenizers {sorted(tokenizers)}. Quality is not "
            f"comparable across tokenizations, and neither is per-token cost."
        )
    return sorted(entries, key=lambda entry: entry["tier"])


def horizontal_systems(
    manifest: list[dict],
    n_requests: int,
    lambdas: tuple[float, ...],
) -> tuple[list[SystemResult], np.ndarray, np.ndarray]:
    """Evaluates independently trained models and their oracle.

    Args:
        manifest: Entries from :func:`load_manifest`.
        n_requests: Requests the vertical side was evaluated on.
        lambdas: Prices of compute to evaluate the oracle at.

    Returns:
        A tuple ``(systems, quality, cost)`` where the matrices are shaped
        ``(n_requests, n_models)``.

    Raises:
        ValueError: If a model's results do not cover the same requests.
    """
    quality_columns, cost_columns = [], []
    for entry in manifest:
        values = np.asarray(json.loads(Path(entry["results"]).read_text()), float)
        if values.shape[0] != n_requests:
            raise ValueError(
                f"Model {entry['model_id']} reports {values.shape[0]} requests "
                f"but the vertical side has {n_requests}. Paired comparison "
                f"requires the same requests in the same order."
            )
        quality_columns.append(values)
        cost_columns.append(np.full(n_requests, float(entry["cost"])))

    quality = np.stack(quality_columns, axis=1)
    cost = np.stack(cost_columns, axis=1)
    cost = cost / max(cost[:, -1].max(), 1e-12)

    systems = [
        SystemResult(
            name=f"independent_{entry['model_id']}",
            family="horizontal",
            operating_point=f"tier_{entry['tier']}",
            quality=quality[:, index],
            cost=cost[:, index],
        )
        for index, entry in enumerate(manifest)
    ]
    systems += [
        oracle_system(quality, cost, lam, family="horizontal") for lam in lambdas
    ]
    return systems, quality, cost


def cascade_system(
    quality: np.ndarray,
    cost: np.ndarray,
    escalate: np.ndarray,
    reuse_prefix: bool,
    name: str,
) -> SystemResult:
    """Costs a two-tier cascade, charging the cheap pass on every request.

    The difference between the two families lives entirely in the cost. Without
    prefix reuse an escalated request is rerun from scratch, so it pays the cheap
    path's cost *plus* the expensive one's. With reuse it continues through the
    suffix and pays the shallow prefix once. Charging both the same way would
    hand the reusing system a saving it has not earned.

    Warning:
        The no-reuse arm is **not** a horizontal model cascade, and is named
        ``vertical_no_reuse_oracle_cascade`` to stop it being read as one. Both
        arms draw their qualities from the *same* backbone's depth endpoints, and
        both escalate using an outcome-based rule that a deployed system could not
        evaluate. It measures what prefix reuse is worth, holding everything else
        fixed. A real horizontal cascade needs independently trained models,
        their own per-request predictions, measured per-shape serving costs, and
        an escalation policy over available features only.

    Args:
        quality: Quality shaped ``(n_requests, 2)`` for the cheap and expensive
            paths.
        cost: Normalized cost, same shape.
        escalate: Boolean mask of requests that go to the expensive path.
        reuse_prefix: Whether the expensive path reuses the cheap path's work.
        name: System name.

    Returns:
        The cascade's result.
    """
    cheap, dear = cost[:, 0], cost[:, 1]
    total = np.where(
        escalate,
        dear if reuse_prefix else cheap + dear,
        cheap,
    )
    return SystemResult(
        name=name,
        family="vertical" if reuse_prefix else "horizontal",
        operating_point=f"escalation_rate_{escalate.mean():.2f}",
        quality=np.where(escalate, quality[:, 1], quality[:, 0]),
        cost=total,
        choices=escalate.astype(int),
        notes=[
            "reusable prefix: escalation pays only the suffix"
            if reuse_prefix
            else "no reuse: escalation reruns the cheap path's work. Same "
            "backbone, oracle escalation -- not a horizontal model cascade"
        ],
    )


def estimands(
    systems: list[SystemResult],
    vertical_quality: np.ndarray,
    vertical_cost: np.ndarray,
    horizontal_quality: np.ndarray | None,
    horizontal_cost: np.ndarray | None,
    tiers: list[int],
    config: EvaluationConfig,
    horizontal_tiers: list[int] | None = None,
    clusters: np.ndarray | None = None,
) -> dict:
    """Computes the estimands the paper's claims rest on.

    Args:
        systems: Every evaluated system.
        vertical_quality: Vertical quality shaped ``(n_requests, n_tiers)``.
        vertical_cost: Vertical cost, same shape.
        horizontal_quality: Horizontal quality, or ``None``.
        horizontal_cost: Horizontal cost, or ``None``.
        tiers: Vertical candidate depths.
        config: Evaluation settings.
        horizontal_tiers: Tier each independent model claims, used to align
            the two sides for the sharing tax.

    Returns:
        A dictionary of estimands, each with the caveats needed to read it.
    """
    by_name = {system.name: system for system in systems}
    results: dict = {}

    # Frontiers, built from the systems that trace one: the fixed endpoints
    # and the learned router swept over prices of compute. Oracles and
    # mixtures are excluded deliberately -- the first is unattainable and the
    # second is the baseline the frontier is being compared against, so
    # folding either in would make the frontier flatter to beat.
    vertical_points = [
        system.point()
        for system in systems
        if system.family == "vertical"
        and (
            system.name.startswith("fixed_depth_")
            or system.name.startswith("learned_request_router@")
        )
    ]
    vertical_frontier = pareto_frontier(vertical_points)
    results["vertical_frontier"] = vertical_frontier

    if horizontal_quality is not None:
        horizontal_points = [
            system.point()
            for system in systems
            if system.family == "horizontal"
            and system.name.startswith("independent_")
        ]
        horizontal_frontier = pareto_frontier(horizontal_points)
        results["horizontal_frontier"] = horizontal_frontier
    else:
        horizontal_frontier = None
        results["horizontal_frontier"] = None
        results["horizontal_note"] = (
            "No manifest supplied, so no independently trained models were "
            "evaluated. Every horizontal estimand is unavailable, not zero."
        )

    # Adaptivity gain and router regret, per price of compute.
    per_lambda = []
    for lam in config.lambdas:
        utility = per_tier_utility(vertical_quality, vertical_cost, lam)
        oracle = float(utility.max(axis=1).mean())
        best_fixed = float(utility.mean(axis=0).max())

        entry = {
            "lambda": lam,
            "oracle_utility": oracle,
            "best_fixed_utility": best_fixed,
            "adaptivity_gain": oracle - best_fixed,
        }

        # The plain oracle knows how each candidate turned out; no router can.
        # Splitting the gain into the part a policy over these features could
        # reach and the part it could not is what stops a near-optimal router
        # being reported as a failure.
        probe_policy = by_name.get(f"cross_fitted_probe_policy@{lam}")
        if probe_policy is not None:
            attained = float(
                (probe_policy.quality - lam * probe_policy.cost).mean()
            )
            entry["probe_policy_utility"] = attained
            entry["probe_policy_gain"] = attained - best_fixed
            entry["outcome_only_gain"] = oracle - attained

        router = by_name.get(f"learned_request_router@{lam}")
        if router is not None:
            achieved = float((router.quality - lam * router.cost).mean())
            entry["router_utility"] = achieved
            entry["router_regret"] = oracle - achieved
            if "probe_policy_utility" in entry:
                entry["router_regret_vs_probe_policy"] = (
                    entry["probe_policy_utility"] - achieved
                )
            entry["mean_depth"] = float(np.array(tiers)[router.choices].mean())

            # The comparison that separates adaptivity from cost control. A
            # coin flip between two endpoints reaches the same average cost
            # without looking at the request at all, so beating it is the
            # minimum evidence that the controller is reading anything.
            mixture = by_name.get(
                f"best_static_mixture@{round(router.mean_cost, 6):.3f}"
            )
            if mixture is not None:
                margin = paired_bootstrap(
                    router.quality - mixture.quality,
                    resamples=config.resamples,
                    seed=config.seed,
                    clusters=clusters,
                )
                entry["vs_static_mixture"] = asdict(margin)
                entry["beats_static_mixture"] = bool(margin.low > 0.0)
        per_lambda.append(entry)
    results["per_lambda"] = per_lambda

    # Sharing tax by tier: the cost of one backbone serving every size, with
    # the routing decision held fixed at the tier. Matched on the tier a model
    # *claims*, never on column order -- a manifest with three models and a
    # backbone with six exits is the normal case, and aligning them positionally
    # would compare an independent model against whichever endpoint happened to
    # share its index.
    taxes = []
    unmatched = []
    if horizontal_quality is not None:
        for column, tier in enumerate(horizontal_tiers or []):
            if tier not in tiers:
                unmatched.append(tier)
                continue
            index = tiers.index(tier)
            difference = (
                horizontal_quality[:, column] - vertical_quality[:, index]
            )
            interval = paired_bootstrap(
                difference, resamples=config.resamples, seed=config.seed,
                clusters=clusters,
            )
            taxes.append(
                {
                    "tier": tier,
                    "independent_quality": float(
                        horizontal_quality[:, column].mean()
                    ),
                    "endpoint_quality": float(vertical_quality[:, index].mean()),
                    "sharing_tax": asdict(interval),
                }
            )
    results["sharing_tax"] = taxes or None
    if unmatched:
        results["sharing_tax_unmatched_tiers"] = unmatched

    # Complementarity: what does a second candidate get right that the first
    # gets wrong? This is where "depths are not specialists" becomes testable.
    results["complementarity"] = complementarity_matrix(vertical_quality, tiers)
    if horizontal_quality is not None:
        results["horizontal_complementarity"] = complementarity_matrix(
            horizontal_quality, list(range(horizontal_quality.shape[1]))
        )

    # Substitution, only where both frontiers are supported.
    if horizontal_frontier:
        span = common_cost_range(vertical_frontier, horizontal_frontier)
        results["common_cost_range"] = span
        baseline = min(point[1] for point in horizontal_frontier)

        if span:
            midpoint = 0.5 * (span[0] + span[1])
            results["vsr_at_midpoint"] = vertical_substitution_ratio(
                vertical_frontier, horizontal_frontier, baseline, midpoint
            )
        integrated = integrated_substitution_ratio(
            vertical_frontier, horizontal_frontier, baseline
        )
        results["integrated_vsr"] = (
            {"ratio": integrated[0], "range": integrated[1]}
            if integrated
            else None
        )

    return results


def complementarity_matrix(
    quality: np.ndarray,
    labels: list[int],
    threshold: float = 0.5,
) -> dict:
    """Estimates, for each ordered pair, how often the second rescues the first.

    Args:
        quality: Quality shaped ``(n_requests, n_candidates)``.
        labels: Candidate names.
        threshold: Quality at or above which a request counts as solved.

    Returns:
        The matrix and its labels. A near-zero off-diagonal means the errors
        are nested — deeper endpoints solve a superset — and routing can then
        save cost but never exceed the best single candidate. Non-zero entries
        are what an oracle over candidates converts into a quality gain.
    """
    solved = quality >= threshold
    size = quality.shape[1]

    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            matrix[i][j] = float((solved[:, j] & ~solved[:, i]).mean())

    return {
        "labels": labels,
        "threshold": threshold,
        "rescue_rate": matrix,
        "note": (
            "entry [i][j] is P(candidate j solves it and candidate i does "
            "not); a matrix of zeros above the diagonal means nested errors"
        ),
    }


def evaluate(config: EvaluationConfig) -> dict:
    """Runs the whole comparison.

    Args:
        config: Evaluation settings.

    Returns:
        A results dictionary ready to serialize.
    """
    records, features, metadata = load(config.trajectories)
    tiers = metadata["tiers"]

    controller = None
    controller_blob = None
    if config.controller:
        checkpoint = (
            Path(config.controller)
            / f"controller-seed{config.controller_seed}.pt"
        )
        if not checkpoint.exists():
            raise ValueError(
                f"Requested controller seed {config.controller_seed}, but "
                f"{checkpoint} does not exist. Choose a seed before evaluation "
                f"or train that seed; do not select one from report results."
            )
        controller, controller_blob = load_controller(checkpoint)

    rows = split_rows(records, "validation")
    if controller_blob is not None:
        rows = controller_report_rows(records, controller_blob)
    if not rows:
        raise ValueError(
            "No validation requests in the trajectories; nothing to evaluate."
        )
    subset = [records[row] for row in rows]

    quality = quality_matrix(subset, config.quality_metric)
    cost = normalized_cost(subset, config.cost_metric)
    probe = features[rows]
    systems: list[SystemResult] = list(fixed_endpoints(quality, cost, tiers))
    systems += [oracle_system(quality, cost, lam) for lam in config.lambdas]
    for lam in config.lambdas:
        estimated = cross_fitted_probe_policy(probe, quality, cost, lam, seed=config.seed)
        estimated.name = f"cross_fitted_probe_policy@{lam}"
        systems.append(estimated)

    if controller is not None:
        threshold = controller_blob["metrics"].get(
            "sufficiency_threshold", 0.5
        )
        for lam in config.lambdas:
            system = learned_router(
                controller, probe, quality, cost, lam, threshold
            )
            system.name = f"learned_request_router@{lam}"
            systems.append(system)

    # The mixture baseline is only interesting at the costs the router chose:
    # matched-cost is the whole point, and generating one at every system's
    # cost buries the comparison in rows nothing is being compared against.
    matched_costs = sorted(
        {
            round(system.mean_cost, 6)
            for system in systems
            if system.name.startswith("learned_request_router@")
        }
    ) or sorted({round(system.mean_cost, 6) for system in systems})

    for target in matched_costs:
        mixture = best_static_mixture(quality, cost, target, seed=config.seed)
        if mixture is not None:
            mixture.name = f"best_static_mixture@{target:.3f}"
            systems.append(mixture)

    # Two-tier cascades, costed by their own rules. The escalation decision is
    # shared so the only difference between them is prefix reuse.
    if len(tiers) >= 2:
        pair_quality = quality[:, [0, -1]]
        pair_cost = cost[:, [0, -1]]
        escalate = quality[:, 0] < quality[:, -1]
        systems.append(
            cascade_system(
                pair_quality, pair_cost, escalate, True, "vertical_cascade"
            )
        )
        systems.append(
            cascade_system(
                pair_quality, pair_cost, escalate, False, "vertical_no_reuse_oracle_cascade"
            )
        )

    horizontal_quality = horizontal_cost = None
    horizontal_tiers: list[int] | None = None
    if config.manifest:
        manifest = load_manifest(config.manifest)
        horizontal_tiers = [int(entry["tier"]) for entry in manifest]
        extra, horizontal_quality, horizontal_cost = horizontal_systems(
            manifest, len(subset), config.lambdas
        )
        systems += extra

    clusters = request_clusters(subset)
    numbers = estimands(
        systems, quality, cost, horizontal_quality, horizontal_cost,
        tiers, config, horizontal_tiers, clusters=clusters,
    )
    numbers["resampling_unit"] = {
        "field": "document_id" if clusters is not None else "request",
        "n_clusters": None if clusters is None else int(len(set(clusters.tolist()))),
        "n_requests": len(subset),
        "note": (
            "intervals resample documents; requests from one document are not "
            "independent evidence"
            if clusters is not None
            else "records carry no document_id (schema 1), so every request is "
            "treated as its own cluster and every interval below is narrower "
            "than it should be"
        ),
    }

    substitution = None
    if horizontal_quality is not None and controller is not None:
        mid = config.lambdas[len(config.lambdas) // 2]
        router = next(
            s for s in systems if s.name == f"learned_request_router@{mid}"
        )
        best_horizontal = max(
            (s for s in systems if s.name.startswith("independent_")),
            key=lambda s: s.mean_quality,
        )
        result = non_inferiority_test(
            router.quality, best_horizontal.quality,
            router.cost, best_horizontal.cost,
            quality_margin=config.quality_margin,
            cost_tolerance=config.cost_tolerance,
            resamples=config.resamples,
            seed=config.seed,
            clusters=clusters,
        )
        result.vsr_target = config.vsr_target

        # The substitution ratio has to be tested, not just reported as an
        # estimand elsewhere. Its interval comes from resampling requests and
        # rebuilding both frontiers inside each replicate.
        span = numbers.get("common_cost_range")
        if span:
            baseline = min(
                point[1] for point in numbers["horizontal_frontier"]
            )
            # The vertical frontier must be built from the same candidates here
            # as in the reported estimand -- fixed endpoints *and* the router
            # swept over lambda. Bootstrapping a frontier of endpoints alone
            # while reporting one that includes the router would put an
            # interval on a different quantity than the point estimate, and the
            # two disagreed by 0.25 when this was first wired up.
            routed = [
                system for system in systems
                if system.name.startswith("learned_request_router@")
            ]
            candidate_quality = np.column_stack(
                [quality] + [system.quality for system in routed]
            )
            candidate_cost = np.column_stack(
                [cost] + [system.cost for system in routed]
            )
            result.vsr = bootstrap_substitution_ratio(
                candidate_quality, candidate_cost,
                horizontal_quality, horizontal_cost,
                cost=0.5 * (span[0] + span[1]),
                baseline_quality=baseline,
                resamples=config.resamples,
                seed=config.seed,
            )
        result.vsr_passes = (
            result.vsr is not None and result.vsr.low >= config.vsr_target
        )

        substitution = {
            "report": substitution_report(result),
            "supported": result.substitution_supported,
            "quality_difference": asdict(result.quality_difference),
            "cost_difference": asdict(result.cost_difference),
            "vsr": asdict(result.vsr) if result.vsr is not None else None,
            "vsr_target": config.vsr_target,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "n_requests": len(subset),
        "tiers": tiers,
        "quality_metric": config.quality_metric,
        "cost_metric": config.cost_metric,
        "systems": [
            {
                "name": system.name,
                "family": system.family,
                "operating_point": system.operating_point,
                "mean_quality": system.mean_quality,
                "mean_cost": system.mean_cost,
                "notes": system.notes,
            }
            for system in systems
        ],
        "estimands": numbers,
        "substitution": substitution,
        "trajectory_metadata": metadata,
    }


def format_markdown(results: dict) -> str:
    """Renders the evaluation as a readable summary.

    Args:
        results: Output of :func:`evaluate`.

    Returns:
        A Markdown document.
    """
    lines = [
        "# Vertical routing evaluation",
        "",
        f"- requests: {results['n_requests']}",
        f"- tiers (executed depths): {results['tiers']}",
        f"- quality: `{results['quality_metric']}`",
        f"- cost: `{results['cost_metric']}`, normalized to full depth",
        "",
        "## Systems",
        "",
        "| system | family | operating point | quality | cost |",
        "|---|---|---|---:|---:|",
    ]
    for system in results["systems"]:
        lines.append(
            f"| `{system['name']}` | {system['family']} | "
            f"{system['operating_point']} | {system['mean_quality']:.4f} | "
            f"{system['mean_cost']:.4f} |"
        )

    lines += ["", "## Adaptivity gain and router regret", "",
              "| lambda | outcome oracle | cross-fitted probe policy | best fixed | "
              "**probe-policy** gain | outcome-only gain | "
              "router regret vs probe policy | mean depth | vs static mixture |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for entry in results["estimands"]["per_lambda"]:
        regret = entry.get("router_regret_vs_probe_policy", entry.get("router_regret"))
        depth = entry.get("mean_depth")
        probe_policy = entry.get("probe_policy_utility")
        probe_gain = entry.get("probe_policy_gain")
        outcome_only = entry.get("outcome_only_gain")
        mixture = entry.get("vs_static_mixture")
        verdict = "n/a"
        if mixture is not None:
            verdict = (
                f"{mixture['estimate']:+.4f} "
                f"[{mixture['low']:+.4f}, {mixture['high']:+.4f}]"
                f"{' **beats**' if entry.get('beats_static_mixture') else ''}"
            )
        fmt = lambda v, w=4: "n/a" if v is None else f"{v:+.{w}f}"
        lines.append(
            f"| {entry['lambda']:.2f} | {entry['oracle_utility']:.4f} | "
            f"{'n/a' if probe_policy is None else f'{probe_policy:.4f}'} | "
            f"{entry['best_fixed_utility']:.4f} | "
            f"**{fmt(probe_gain)}** | {fmt(outcome_only)} | "
            f"{fmt(regret)} | "
            f"{'n/a' if depth is None else f'{depth:.2f}'} | {verdict} |"
        )

    lines += [
        "",
        "**Read the `probe-policy gain` column, not `outcome oracle - best "
        "fixed`.** The outcome oracle picks per request by looking at how each "
        "candidate turned out, which no deployable policy can do. The "
        "cross-fitted probe policy is one specified learner restricted to the "
        "probe features, fitted out of fold. The gap between them is headroom "
        "that required the answer. On this repository's own workload the outcome "
        "oracle showed +0.051 and the probe policy attained +0.008, so judging "
        "the router against the outcome oracle would have reported a "
        "near-optimal policy as a failure.",
        "",
        "The probe policy is **not a ceiling**. It is the out-of-fold "
        "performance of one model class, so a better learner can beat it and "
        "the regret column can legitimately go negative. The theoretical best "
        "policy measurable from these features is not computed here.",
        "",
        "So: a large probe-policy gain with large regret against the probe "
        "policy is a *controller* problem. A small probe-policy gain is an "
        "*endpoint or feature* problem, and no amount of controller tuning will "
        "fix it.",
        "",
        "The last column is the one that separates adaptivity from cost "
        "control. A weighted coin flip between two fixed depths reaches any "
        "average cost between them without reading the request at all, so a "
        "router whose interval spans zero has shown only that it can hit a "
        "budget.",
        "",
    ]

    complementarity = results["estimands"].get("complementarity")
    if complementarity:
        lines += ["## Depth complementarity", "",
                  f"P(column solves it and row does not), at quality "
                  f">= {complementarity['threshold']}:", "",
                  "| oracle row \\ candidate | " +
                  " | ".join(f"d{l}" for l in complementarity["labels"]) + " |",
                  "|---|" + "---:|" * len(complementarity["labels"])]
        for label, row in zip(complementarity["labels"],
                              complementarity["rescue_rate"]):
            lines.append(
                f"| d{label} | " + " | ".join(f"{v:.3f}" for v in row) + " |"
            )
        lines += [
            "",
            "Zeros above the diagonal mean the errors are nested: deeper "
            "endpoints solve a superset, so routing can save cost but cannot "
            "beat the best single endpoint on quality.",
            "",
        ]

    if results["estimands"].get("horizontal_frontier") is None:
        lines += [
            "## Horizontal comparison",
            "",
            "**Not evaluated.** No manifest of independently trained models "
            "was supplied, so every horizontal estimand — sharing tax, "
            "substitution ratio, horizontal regret — is unavailable. That is "
            "not the same as zero, and the central claim of the paper cannot "
            "be assessed from this run.",
            "",
        ]

    if results.get("substitution"):
        lines += ["## Substitution test", "", "```",
                  results["substitution"]["report"], "```", ""]

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> EvaluationConfig:
    """Builds an :class:`EvaluationConfig` from the command line.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed configuration.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trajectories", default="results/trajectories")
    parser.add_argument("--controller", default="results/controller")
    parser.add_argument("--controller_seed", type=int, default=0)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default="results/evaluation")
    parser.add_argument("--quality_metric", default="teacher_forced_accuracy",
                        choices=sorted(QUALITY_METRICS))
    parser.add_argument("--cost_metric", default="cost_depth_fraction",
                        choices=COST_METRICS)
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    parser.add_argument("--quality_margin", type=float, default=0.02)
    parser.add_argument("--cost_tolerance", type=float, default=0.0)
    parser.add_argument("--vsr_target", type=float, default=0.9)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)

    parsed = vars(parser.parse_args(argv))
    parsed["lambdas"] = tuple(parsed["lambdas"])
    return EvaluationConfig(**parsed)


def main(argv: list[str] | None = None) -> None:
    """Evaluates every system and writes JSON plus Markdown.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.
    """
    config = parse_args(argv)
    provenance = RunRecord.create(
        "experiments.evaluate_vertical_routing",
        config=asdict(config),
        seeds={"bootstrap": config.seed},
        inputs={
            "trajectories": config.trajectories,
            "controller": config.controller,
            "manifest": config.manifest or "not supplied",
        },
        notes=[
            "Every system is evaluated on the same held-out requests.",
            "Costs are estimated multiply-accumulates or depth fractions, "
            "never latency. See experiments/benchmark_latency.py for that.",
        ],
    )
    print(provenance.summary())
    print()

    results = evaluate(config)
    path = Path(config.out)
    provenance.write(path / "evaluation.json", payload=results)
    (path / "evaluation.md").write_text(format_markdown(results))

    print(format_markdown(results))
    print(f"wrote {path / 'evaluation.json'} and {path / 'evaluation.md'}")


if __name__ == "__main__":
    main()
