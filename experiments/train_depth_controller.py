"""Fitting the depth controller on a frozen backbone.

Post-hoc training, deliberately. Fitting the controller and the backbone
together would confound two questions — whether the probe carries enough signal
to choose a depth, and whether a backbone can be reshaped to make depth choosing
easier — and the first has to be answered before the second is interpretable. So
nothing here touches the backbone; the trajectories already contain everything.

Three targets are supported, and which one is right depends on what is being
claimed:

* **earliest sufficient depth.** For a tolerance, the shallowest tier whose
  quality is within it of a declared reference. Fits the ordinal head, whose
  cumulative outputs match the nested structure of the events.
* **per-tier utility.** Predict quality at each tier and subtract the cost at
  selection time. Fits the utility head, and is what allows the price of
  compute to change at inference on a frozen controller — the property that
  makes one checkpoint serve a whole frontier instead of one point on it.
* **continuation gain.** The best utility available deeper, minus the utility
  here. This is the quantity the stopping rule is actually about, and it is
  stored as a diagnostic even when the fitted head is one of the other two.

The controller never sees a target, a label, or a final-layer state at
inference. It sees pooled probe features and nothing else, which is enforced by
the collection schema rather than by convention.

Run it::

    python -m experiments.train_depth_controller \\
        --trajectories results/trajectories --out results/controller
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.routing import DepthController
from experiments.collect_depth_trajectories import load
from utils.provenance import RunRecord

#: Version of the saved controller layout.
SCHEMA_VERSION = 2

#: Quality columns a trajectory record can supply, and whether larger is better.
QUALITY_METRICS = {
    "teacher_forced_accuracy": True,
    "teacher_forced_top1_agreement": True,
    "teacher_forced_nll": False,
    # The only column comparable against a model from another tokenizer family.
    "bits_per_byte": False,
    "free_running_reward": True,
    "free_running_agreement": True,
}

#: Cost columns a trajectory record can supply.
COST_METRICS = ("cost_depth_fraction", "cost_macs", "kv_bytes")


@dataclass
class ControllerTrainConfig:
    """Settings for one controller fit.

    Attributes:
        trajectories: Directory written by the collector.
        out: Directory for the trained controller.
        quality_metric: Which recorded quality to fit. Naming it explicitly
            matters: a controller fitted on teacher-forced labels and reported
            on free-running quality is a different claim from one fitted and
            reported on the same thing.
        cost_metric: Which recorded cost to trade quality against.
        target: ``"utility"`` or ``"ordinal"``.
        epsilon: Tolerance for the sufficiency definition, in quality units.
        reference: ``"full_depth"`` or ``"best_endpoint"``. The second admits
            that a deeper endpoint can be *worse*, which the first cannot.
        routing_lambda: Price of a unit of normalized cost, used for the
            utility target and for every reported utility.
        hidden: Controller bottleneck width.
        epochs: Passes over the training split.
        batch_size: Requests per step.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        seeds: Seeds to repeat the fit under. Every one is reported.
    """

    trajectories: str = "results/trajectories"
    out: str = "results/controller"
    quality_metric: str = "teacher_forced_accuracy"
    cost_metric: str = "cost_depth_fraction"
    target: str = "utility"
    epsilon: float = 0.01
    reference: str = "full_depth"
    routing_lambda: float = 0.1
    hidden: int = 64
    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seeds: tuple[int, ...] = (0, 1, 2)


def stack_column(records: list[dict], name: str) -> np.ndarray:
    """Collects one per-tier column into an array.

    Args:
        records: Trajectory records.
        name: Field to collect.

    Returns:
        An array shaped ``(n_requests, n_tiers)``.

    Raises:
        ValueError: If the field is missing or empty, which happens when
            free-running labels were not collected.
    """
    values = [record.get(name) for record in records]
    if any(not value for value in values):
        raise ValueError(
            f"Field {name!r} is empty for at least one record. Free-running "
            f"labels need --max_new_tokens > 0 at collection time."
        )
    return np.asarray(values, dtype=float)


def quality_matrix(records: list[dict], metric: str) -> np.ndarray:
    """Reads a quality column, oriented so larger is always better.

    Args:
        records: Trajectory records.
        metric: A key of :data:`QUALITY_METRICS`.

    Returns:
        Quality shaped ``(n_requests, n_tiers)``.

    Raises:
        ValueError: If the metric is unknown.
    """
    if metric not in QUALITY_METRICS:
        raise ValueError(
            f"quality_metric must be one of {sorted(QUALITY_METRICS)}, got "
            f"{metric!r}."
        )
    values = stack_column(records, metric)
    # Negated where smaller is better, so every downstream comparison can
    # assume one direction. Sign confusion here would invert the whole policy.
    return values if QUALITY_METRICS[metric] else -values


def normalized_cost(records: list[dict], metric: str) -> np.ndarray:
    """Reads a cost column and scales it to ``[0, 1]``.

    Normalizing makes ``routing_lambda`` mean the same thing whether cost is
    counted in depth fractions or multiply-accumulates, so a value tuned on one
    transfers to the other.

    Args:
        records: Trajectory records.
        metric: One of :data:`COST_METRICS`.

    Returns:
        Cost shaped ``(n_requests, n_tiers)``, divided by the deepest tier's.

    Raises:
        ValueError: If the metric is unknown.
    """
    if metric not in COST_METRICS:
        raise ValueError(
            f"cost_metric must be one of {COST_METRICS}, got {metric!r}."
        )
    values = stack_column(records, metric)
    return values / np.maximum(values[:, -1:], 1e-12)


def earliest_sufficient_index(
    quality: np.ndarray,
    epsilon: float,
    reference: str = "full_depth",
) -> np.ndarray:
    """Finds the shallowest tier good enough, per request.

    Args:
        quality: Quality shaped ``(n_requests, n_tiers)``, larger better.
        epsilon: How much quality may be given up.
        reference: ``"full_depth"`` compares against the deepest tier;
            ``"best_endpoint"`` compares against whichever tier was best for
            that request, which is the honest reference when depth is not
            monotone and a deeper endpoint can be worse.

    Returns:
        Tier indices shaped ``(n_requests,)``.

    Raises:
        ValueError: If the reference is unknown.
    """
    if reference == "full_depth":
        target = quality[:, -1]
    elif reference == "best_endpoint":
        target = quality.max(axis=1)
    else:
        raise ValueError(
            f"reference must be 'full_depth' or 'best_endpoint', got "
            f"{reference!r}."
        )

    sufficient = quality >= (target[:, None] - epsilon)
    # The deepest tier is always allowed, so argmax always finds something.
    sufficient[:, -1] = True
    return sufficient.argmax(axis=1)


def per_tier_utility(
    quality: np.ndarray,
    cost: np.ndarray,
    routing_lambda: float,
) -> np.ndarray:
    """Scalarizes quality and cost at each tier.

    Args:
        quality: Quality shaped ``(n_requests, n_tiers)``.
        cost: Normalized cost, same shape.
        routing_lambda: Price of a unit of cost.

    Returns:
        Utility of the same shape.
    """
    return quality - routing_lambda * cost


def continuation_gain(
    quality: np.ndarray,
    cost: np.ndarray,
    routing_lambda: float,
) -> np.ndarray:
    """Best utility available deeper, minus the utility here.

    This is the quantity the Bellman stopping rule compares against zero:
    continue exactly when something deeper is worth more than it costs. The
    last tier has nothing beyond it and is defined as zero gain.

    Args:
        quality: Quality shaped ``(n_requests, n_tiers)``.
        cost: Normalized cost, same shape.
        routing_lambda: Price of a unit of cost.

    Returns:
        Gain shaped ``(n_requests, n_tiers)``.
    """
    utility = per_tier_utility(quality, cost, routing_lambda)
    gain = np.zeros_like(utility)
    for tier in range(utility.shape[1] - 1):
        gain[:, tier] = utility[:, tier + 1 :].max(axis=1) - utility[:, tier]
    return gain


def policy_metrics(
    chosen: np.ndarray,
    quality: np.ndarray,
    cost: np.ndarray,
    tiers: list[int],
    routing_lambda: float,
) -> dict[str, float]:
    """Scores a policy against the oracle and the best fixed depth.

    Args:
        chosen: Tier index per request.
        quality: Quality shaped ``(n_requests, n_tiers)``.
        cost: Normalized cost, same shape.
        tiers: Candidate depths, for reporting the mean depth.
        routing_lambda: Price of a unit of cost.

    Returns:
        Mean quality, cost, depth, and utility, alongside the oracle utility,
        the best fixed-depth utility, the adaptivity gain between them, and the
        router's regret against the oracle. Those last two are the pair that
        says where an underperforming system is losing: a large gain with large
        regret is a controller problem, and a small gain is an endpoint problem
        no controller can fix.
    """
    rows = np.arange(len(chosen))
    utility = per_tier_utility(quality, cost, routing_lambda)

    achieved = float(utility[rows, chosen].mean())
    oracle = float(utility.max(axis=1).mean())
    fixed = utility.mean(axis=0)
    best_fixed = float(fixed.max())

    return {
        "mean_quality": float(quality[rows, chosen].mean()),
        "mean_cost": float(cost[rows, chosen].mean()),
        "mean_depth": float(np.array(tiers)[chosen].mean()),
        "utility": achieved,
        "oracle_utility": oracle,
        "best_fixed_utility": best_fixed,
        "best_fixed_tier": int(tiers[int(fixed.argmax())]),
        "adaptivity_gain": oracle - best_fixed,
        "router_regret": oracle - achieved,
    }


def depth_confusion(
    chosen: np.ndarray,
    oracle: np.ndarray,
    n_tiers: int,
) -> list[list[int]]:
    """Counts chosen tier against oracle tier.

    Args:
        chosen: Tier index the controller picked.
        oracle: Tier index the oracle would have picked.
        n_tiers: Number of tiers.

    Returns:
        A matrix whose ``[i][j]`` entry counts requests the oracle put at tier
        ``i`` and the controller put at tier ``j``. Off-diagonal mass above the
        diagonal is over-spending; below it is under-serving, which is the
        expensive kind of error.
    """
    matrix = [[0] * n_tiers for _ in range(n_tiers)]
    for want, got in zip(oracle.tolist(), chosen.tolist()):
        matrix[int(want)][int(got)] += 1
    return matrix


def calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    bins: int = 10,
) -> float:
    """Expected calibration error of predicted probabilities.

    Args:
        probabilities: Predicted probabilities, flattened.
        outcomes: Matching binary outcomes.
        bins: Equal-width bins over ``[0, 1]``.

    Returns:
        The weighted mean absolute gap between confidence and accuracy. Zero
        means a stated 70% happens 70% of the time.
    """
    probabilities = probabilities.ravel()
    outcomes = outcomes.ravel().astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)

    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (probabilities > low) & (probabilities <= high)
        if not in_bin.any():
            continue
        total += in_bin.mean() * abs(
            probabilities[in_bin].mean() - outcomes[in_bin].mean()
        )
    return float(total)


def fit_controller(
    features: np.ndarray,
    quality: np.ndarray,
    cost: np.ndarray,
    tiers: list[int],
    config: ControllerTrainConfig,
    seed: int = 0,
    d_model: int | None = None,
    verbose: bool = False,
) -> tuple[DepthController, dict[str, float]]:
    """Trains one controller on one split.

    Args:
        features: Probe features shaped ``(n_requests, feature_dim)``.
        quality: Quality shaped ``(n_requests, n_tiers)``, larger better.
        cost: Normalized cost, same shape.
        tiers: Candidate depths.
        config: Training settings.
        seed: Random seed for initialization and batching.
        d_model: Backbone width, used only to size the controller when the
            pooling scheme is not the default. Inferred from ``features`` when
            omitted.
        verbose: Whether to print progress.

    Returns:
        A tuple ``(controller, history)`` where ``history`` holds the final
        training loss and the target used.

    Raises:
        ValueError: If the target is unknown.
    """
    if config.target not in ("utility", "ordinal"):
        raise ValueError(
            f"target must be 'utility' or 'ordinal', got {config.target!r}."
        )

    torch.manual_seed(seed)
    n_tiers = len(tiers)
    feature_dim = features.shape[1]

    # Built at the observed feature width rather than re-derived from the
    # pooling scheme, so a mismatch between collection and training surfaces as
    # a shape error instead of silent nonsense.
    controller = DepthController(
        d_model=d_model or feature_dim,
        n_tiers=n_tiers,
        hidden_dim=config.hidden,
        output=config.target,
        input_dim=feature_dim,
    )

    x = torch.tensor(features, dtype=torch.float32)
    if config.target == "utility":
        y = torch.tensor(quality, dtype=torch.float32)
    else:
        index = earliest_sufficient_index(quality, config.epsilon, config.reference)
        # Cumulative indicators: tier k is sufficient once the earliest
        # sufficient tier is at or below it. Nested by construction, which is
        # what the monotone head is built to represent.
        y = torch.tensor(
            (np.arange(n_tiers)[None, :] >= index[:, None]).astype(np.float32)
        )

    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)

    controller.train()
    loss = torch.zeros(())
    for epoch in range(config.epochs):
        order = torch.randperm(x.size(0), generator=generator)
        for start in range(0, x.size(0), config.batch_size):
            rows = order[start : start + config.batch_size]
            prediction = controller(x[rows])

            if config.target == "utility":
                loss = F.mse_loss(prediction, y[rows])
            else:
                loss = F.binary_cross_entropy(
                    prediction.clamp(1e-6, 1 - 1e-6), y[rows]
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose and (epoch + 1) % max(config.epochs // 4, 1) == 0:
            print(f"    epoch {epoch + 1:>4}  loss {float(loss.detach()):.5f}")

    controller.eval()
    return controller, {"final_loss": float(loss.detach()), "target": config.target}


@torch.no_grad()
def choose(
    controller: DepthController,
    features: np.ndarray,
    cost: np.ndarray,
    routing_lambda: float,
    sufficiency_threshold: float = 0.5,
) -> np.ndarray:
    """Applies a trained controller to pick a tier per request.

    Args:
        controller: Trained controller.
        features: Probe features.
        cost: Normalized cost shaped ``(n_requests, n_tiers)``.
        routing_lambda: Price of a unit of cost.
        sufficiency_threshold: Cutoff for the ordinal head.

    Returns:
        Tier indices shaped ``(n_requests,)``.
    """
    scores = controller(torch.tensor(features, dtype=torch.float32)).numpy()

    if controller.output == "utility":
        return (scores - routing_lambda * cost).argmax(axis=1)

    sufficient = scores >= sufficiency_threshold
    sufficient[:, -1] = True
    return sufficient.argmax(axis=1)


def split_rows(records: list[dict], split: str) -> list[int]:
    """Row indices belonging to one split.

    Args:
        records: Trajectory records.
        split: ``"train"`` or ``"validation"``.

    Returns:
        Matching row indices.
    """
    return [row for row, record in enumerate(records) if record["split"] == split]


def halve_by_source(
    records: list[dict],
    rows: list[int],
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Splits held-out rows into calibration and reporting halves.

    Calibrating and reporting on the same requests would make the reported
    numbers a description of the calibration fit rather than of held-out
    behaviour, so the split is made once, by source, before either happens.

    Args:
        records: Trajectory records.
        rows: Held-out row indices.
        seed: Shuffle seed.

    Returns:
        A tuple ``(calibration_rows, report_rows)``.
    """
    sources = sorted({records[row]["source_id"] for row in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(sources)
    held = set(sources[: max(1, len(sources) // 2)])

    calibration = [row for row in rows if records[row]["source_id"] in held]
    report = [row for row in rows if records[row]["source_id"] not in held]
    return calibration, report


def run(config: ControllerTrainConfig) -> dict:
    """Fits the controller under every seed and evaluates it.

    Args:
        config: Training settings.

    Returns:
        A results dictionary with one entry per seed plus an aggregate.
    """
    records, features, metadata = load(config.trajectories)
    tiers = metadata["tiers"]

    quality = quality_matrix(records, config.quality_metric)
    cost = normalized_cost(records, config.cost_metric)

    train_rows = split_rows(records, "train")
    held_rows = split_rows(records, "validation")
    calibration_rows, report_rows = halve_by_source(records, held_rows)

    print(
        f"{len(train_rows)} train / {len(calibration_rows)} calibration / "
        f"{len(report_rows)} report requests, tiers {tiers}"
    )

    oracle_index = per_tier_utility(
        quality, cost, config.routing_lambda
    ).argmax(axis=1)

    per_seed = []
    for seed in config.seeds:
        print(f"  seed {seed} ...")
        controller, history = fit_controller(
            features[train_rows],
            quality[train_rows],
            cost[train_rows],
            tiers,
            config,
            seed=seed,
        )

        threshold = config.epsilon and 0.5
        if config.target == "ordinal":
            threshold = _tune_threshold(
                controller,
                features[calibration_rows],
                quality[calibration_rows],
                cost[calibration_rows],
                tiers,
                config,
            )

        chosen = choose(
            controller,
            features[report_rows],
            cost[report_rows],
            config.routing_lambda,
            threshold,
        )
        metrics = policy_metrics(
            chosen,
            quality[report_rows],
            cost[report_rows],
            tiers,
            config.routing_lambda,
        )
        metrics["sufficiency_threshold"] = float(threshold)
        metrics["confusion"] = depth_confusion(
            chosen, oracle_index[report_rows], len(tiers)
        )

        if config.target == "ordinal":
            with torch.no_grad():
                probabilities = controller(
                    torch.tensor(features[report_rows], dtype=torch.float32)
                ).numpy()
            index = earliest_sufficient_index(
                quality[report_rows], config.epsilon, config.reference
            )
            outcomes = (np.arange(len(tiers))[None, :] >= index[:, None])
            metrics["calibration_error"] = calibration_error(
                probabilities, outcomes
            )

        per_seed.append(
            {"seed": seed, **history, **metrics, "controller": controller}
        )
        print(
            f"    depth {metrics['mean_depth']:.2f}  "
            f"quality {metrics['mean_quality']:.4f}  "
            f"regret {metrics['router_regret']:+.4f}  "
            f"(oracle gain {metrics['adaptivity_gain']:+.4f})"
        )

    return {
        "tiers": tiers,
        "metadata": metadata,
        "per_seed": per_seed,
        "splits": {
            "train": len(train_rows),
            "calibration": len(calibration_rows),
            "report": len(report_rows),
        },
        # Persist identities, not only counts. The evaluator must run every
        # baseline on the untouched reporting half; using all validation rows
        # would leak the ordinal threshold's calibration examples back into the
        # headline comparison.
        "split_request_ids": {
            "train": [records[row]["request_id"] for row in train_rows],
            "calibration": [
                records[row]["request_id"] for row in calibration_rows
            ],
            "report": [records[row]["request_id"] for row in report_rows],
        },
    }


def _tune_threshold(
    controller: DepthController,
    features: np.ndarray,
    quality: np.ndarray,
    cost: np.ndarray,
    tiers: list[int],
    config: ControllerTrainConfig,
) -> float:
    """Picks the ordinal head's cutoff on the calibration split.

    The ordinal head predicts sufficiency, not utility, so it has no way to
    know the price of compute. The cutoff is where that price enters, and it is
    chosen on data the reported numbers never touch.

    Args:
        controller: Trained controller.
        features: Calibration features.
        quality: Calibration quality.
        cost: Calibration cost.
        tiers: Candidate depths.
        config: Training settings supplying the price of compute.

    Returns:
        The cutoff maximizing utility on the calibration split.
    """
    best, best_utility = 0.5, -float("inf")
    for threshold in np.linspace(0.05, 0.95, 19):
        chosen = choose(
            controller, features, cost, config.routing_lambda, float(threshold)
        )
        utility = policy_metrics(
            chosen, quality, cost, tiers, config.routing_lambda
        )["utility"]
        if utility > best_utility:
            best, best_utility = float(threshold), utility
    return best


def save(results: dict, config: ControllerTrainConfig, record: RunRecord) -> Path:
    """Writes the controllers and their metrics.

    Args:
        results: Output of :func:`run`.
        config: Training settings.
        record: Provenance.

    Returns:
        The directory written to.
    """
    path = Path(config.out)
    path.mkdir(parents=True, exist_ok=True)

    summary = []
    for entry in results["per_seed"]:
        controller = entry.pop("controller")
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "state_dict": controller.state_dict(),
                "feature_dim": controller.feature_dim,
                "n_tiers": controller.n_tiers,
                "hidden_dim": controller.hidden_dim,
                "output": controller.output,
                "tiers": results["tiers"],
                "train_config": asdict(config),
                "split_request_ids": results["split_request_ids"],
                "metrics": {
                    k: v for k, v in entry.items() if k != "confusion"
                },
            },
            path / f"controller-seed{entry['seed']}.pt",
        )
        summary.append(entry)

    record.write(
        path / "run.json",
        payload={
            "tiers": results["tiers"],
            "splits": results["splits"],
            "split_request_ids": results["split_request_ids"],
            "per_seed": summary,
            "trajectory_metadata": results["metadata"],
        },
    )
    return path


def parse_args(argv: list[str] | None = None) -> ControllerTrainConfig:
    """Builds a :class:`ControllerTrainConfig` from the command line.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed configuration.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trajectories", default="results/trajectories")
    parser.add_argument("--out", default="results/controller")
    parser.add_argument(
        "--quality_metric", default="teacher_forced_accuracy",
        choices=sorted(QUALITY_METRICS),
    )
    parser.add_argument("--cost_metric", default="cost_depth_fraction",
                        choices=COST_METRICS)
    parser.add_argument("--target", default="utility",
                        choices=("utility", "ordinal"))
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--reference", default="full_depth",
                        choices=("full_depth", "best_endpoint"))
    parser.add_argument("--routing_lambda", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    parsed = vars(parser.parse_args(argv))
    parsed["seeds"] = tuple(parsed["seeds"])
    return ControllerTrainConfig(**parsed)


def main(argv: list[str] | None = None) -> None:
    """Fits the controller and writes it to disk.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.
    """
    config = parse_args(argv)
    provenance = RunRecord.create(
        "experiments.train_depth_controller",
        config=asdict(config),
        seeds={f"fit_{seed}": seed for seed in config.seeds},
        inputs={"trajectories": config.trajectories},
        notes=[
            "Backbone frozen; only the controller is fitted.",
            f"Fitted and reported on '{config.quality_metric}'.",
            "Reported on a held-out half that calibration never touched.",
        ],
    )
    print(provenance.summary())
    print()

    results = run(config)
    regrets = [entry["router_regret"] for entry in results["per_seed"]]
    gains = [entry["adaptivity_gain"] for entry in results["per_seed"]]

    path = save(results, config, provenance)
    print(
        f"\nrouter regret across seeds: "
        f"{np.mean(regrets):+.4f} +/- {np.std(regrets):.4f}"
    )
    print(f"oracle adaptivity gain:     {np.mean(gains):+.4f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
