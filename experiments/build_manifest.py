"""Turning independently trained checkpoints into a horizontal manifest.

The sharing tax asks what it costs to make one backbone serve every size: a
model trained alone at depth *d* against the depth-*d* endpoint of the shared
backbone, on the same requests. :mod:`experiments.evaluate_vertical_routing`
computes it, but only from a manifest of independently trained models, and a
manifest is not something a training run produces. This module is the step
between: collect trajectories for each independent checkpoint the same way the
shared backbone's were collected, then point this at both.

``vr-noexits`` is exactly one such model. It has a single exit on the final
layer, so its trajectories carry one tier, and it is the independently trained
counterpart of the shared backbone's deepest endpoint. That gives the sharing
tax at full depth, which is the tier the comparison turns on: if training with
six exits costs the final layer nothing, shallow-tier taxes are a refinement,
and if it costs the final layer a lot, the thesis is already in trouble.

**Why alignment is checked rather than assumed.** The comparison is paired --
it bootstraps a per-request difference -- so the two sides must describe the
same requests in the same order. Two collections with the same workload, seed
and request count do produce that, which makes it tempting to trust position.
But nothing in the file format records the agreement, a changed seed leaves no
trace, and the failure is silent: mismatched rows still subtract, still
bootstrap, and still report a confident interval around a meaningless number.
So request identity is verified explicitly, and a mismatch is an error with the
first offending row named rather than a warning.

Run it::

    python -m experiments.build_manifest \\
        --vertical results/vr-exits/trajectories \\
        --independent noexits=results/vr-noexits/trajectories \\
        --out results/horizontal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.collect_depth_trajectories import load
from experiments.train_depth_controller import (
    quality_matrix,
    split_rows,
    stack_column,
)
from utils.provenance import RunRecord


def _validation_subset(directory: str | Path) -> tuple[list[dict], dict]:
    """Loads a trajectory directory and keeps its validation rows.

    Args:
        directory: Directory written by
            :mod:`experiments.collect_depth_trajectories`.

    Returns:
        A tuple ``(records, metadata)`` where ``records`` holds only the rows
        whose split is ``"validation"``, in file order.

    Raises:
        ValueError: If the collection has no validation rows, which would make
            it unusable for a held-out comparison.
    """
    records, _features, metadata = load(directory)
    rows = split_rows(records, "validation")
    if not rows:
        raise ValueError(
            f"{directory} has no validation requests, so it cannot take part "
            f"in a held-out comparison."
        )
    return [records[row] for row in rows], metadata


def _require_same_requests(
    reference: list[dict],
    candidate: list[dict],
    reference_name: str,
    candidate_name: str,
) -> None:
    """Fails unless two collections cover identical requests in identical order.

    Args:
        reference: Validation records from the shared backbone.
        candidate: Validation records from an independently trained model.
        reference_name: Directory of the reference, for the error message.
        candidate_name: Directory of the candidate, for the error message.

    Raises:
        ValueError: If the lengths differ, or if any row disagrees about which
            request it describes.
    """
    if len(reference) != len(candidate):
        raise ValueError(
            f"{candidate_name} has {len(candidate)} validation requests but "
            f"{reference_name} has {len(reference)}. A paired comparison needs "
            f"the same requests on both sides -- collect both with the same "
            f"--n_requests, --seed and --validation_fraction."
        )

    for row, (left, right) in enumerate(zip(reference, candidate)):
        if left["request_id"] != right["request_id"]:
            raise ValueError(
                f"{candidate_name} and {reference_name} disagree at validation "
                f"row {row}: request {right['request_id']} against "
                f"{left['request_id']}. The collections were not built from "
                f"the same workload, so pairing them would compare unrelated "
                f"requests."
            )


def _tier_index(records: list[dict], tier: int | None) -> tuple[int, int]:
    """Chooses which tier of an independent model represents it.

    An independently trained model stands at one depth. Its collection usually
    carries exactly one tier, and the deepest is the right default in any case:
    that is the model at full capability, which is what the horizontal side is
    supposed to be.

    Args:
        records: Validation records for the independent model.
        tier: Explicit tier, or ``None`` to take the deepest available.

    Returns:
        A tuple ``(tier, index)`` naming the depth and its column.

    Raises:
        ValueError: If an explicit tier is not among those collected.
    """
    tiers = list(records[0]["tiers"])
    if tier is None:
        return tiers[-1], len(tiers) - 1
    if tier not in tiers:
        raise ValueError(
            f"Tier {tier} was not collected; available tiers are {tiers}."
        )
    return tier, tiers.index(tier)


def _tokenizer_id(metadata: dict, declared: str | None) -> str:
    """Names the tokenization a collection was made under.

    Trajectory metadata records the model configuration but not the tokenizer,
    so there is nothing to read directly. Vocabulary size is the surrogate:
    equal sizes are necessary for two collections to share a tokenization and
    are cheap to check, but they are not sufficient -- two different 50257-token
    vocabularies would pass. ``--tokenizer_id`` is there to state the truth when
    it is known, and this is the fallback when it is not.

    Args:
        metadata: Trajectory metadata.
        declared: Value supplied on the command line, or ``None``.

    Returns:
        The declared id, or one derived from the vocabulary size.
    """
    if declared:
        return declared
    vocab = metadata.get("model_config", {}).get("vocab_size", "unknown")
    return f"vocab-{vocab}"


def build(
    vertical: str | Path,
    independent: dict[str, str],
    out_dir: str | Path,
    quality_metric: str = "teacher_forced_accuracy",
    cost_metric: str = "cost_macs",
    tiers: dict[str, int] | None = None,
    tokenizer_id: str | None = None,
) -> Path:
    """Writes a manifest and one per-request quality file per model.

    Args:
        vertical: Trajectory directory for the shared backbone. It defines the
            request set every independent model is checked against.
        independent: Model id to trajectory directory, one entry per
            independently trained checkpoint.
        out_dir: Destination directory for ``manifest.json`` and the quality
            files it references.
        quality_metric: Column to export, read exactly as the evaluation reads
            it so the two sides cannot disagree about sign or scale.
        cost_metric: Column the manifest's scalar ``cost`` comes from, averaged
            over the validation requests.
        tiers: Optional explicit tier per model id.
        tokenizer_id: Tokenization every collection was made under. Defaults to
            a surrogate derived from vocabulary size; see :func:`_tokenizer_id`.

    Returns:
        The path of the written manifest.

    Raises:
        ValueError: If a collection is empty, misaligned, or mixes tokenizers.
    """
    reference, reference_meta = _validation_subset(vertical)
    tokenizers = {_tokenizer_id(reference_meta, tokenizer_id)}

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    entries = []
    for model_id, directory in independent.items():
        records, metadata = _validation_subset(directory)
        _require_same_requests(reference, records, str(vertical), str(directory))

        tier, index = _tier_index(records, (tiers or {}).get(model_id))
        quality = quality_matrix(records, quality_metric)[:, index]
        cost = float(stack_column(records, cost_metric)[:, index].mean())

        results_path = destination / f"{model_id}.json"
        results_path.write_text(json.dumps([float(v) for v in quality]))

        tokenizer = _tokenizer_id(metadata, tokenizer_id)
        tokenizers.add(tokenizer)
        entries.append(
            {
                "model_id": model_id,
                "tokenizer_id": tokenizer,
                "tier": int(tier),
                "cost": cost,
                "results": str(results_path),
                "quality_metric": quality_metric,
                "cost_metric": cost_metric,
                "n_requests": int(quality.shape[0]),
                "trajectories": str(directory),
            }
        )

    # Caught here rather than in load_manifest, because at this point the
    # offending directory can still be named. Quality is not comparable across
    # tokenizations and neither is per-token cost, so a mixed manifest is not a
    # comparison that can be repaired downstream.
    if len(tokenizers) > 1:
        raise ValueError(
            f"The shared backbone and the independent models were collected "
            f"under different tokenizations {sorted(tokenizers)}. Neither "
            f"quality nor cost is comparable across tokenizations. Pass "
            f"--tokenizer_id if the vocabulary sizes differ for another reason."
        )

    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(entries, indent=2))

    RunRecord.create(
        script="experiments.build_manifest",
        config={
            "vertical": str(vertical),
            "independent": independent,
            "quality_metric": quality_metric,
            "cost_metric": cost_metric,
        },
        inputs={"trajectories": [str(vertical), *independent.values()]},
        notes=[
            "Quality columns are copied from each model's own trajectory "
            "collection; this module computes nothing about the models.",
        ],
    ).write(destination / "run.json", payload={"manifest": entries})

    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the command line.

    Args:
        argv: Argument list, or ``None`` to read ``sys.argv``.

    Returns:
        The parsed arguments, with ``--independent`` reduced to a dictionary.

    Raises:
        SystemExit: On a malformed ``--independent`` entry.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vertical", required=True)
    parser.add_argument(
        "--independent",
        action="append",
        required=True,
        metavar="MODEL_ID=DIR",
        help="Independently trained model and its trajectory directory. "
             "Repeat for a family.",
    )
    parser.add_argument("--out", default="results/horizontal")
    parser.add_argument("--quality_metric", default="teacher_forced_accuracy")
    parser.add_argument("--cost_metric", default="cost_macs")
    parser.add_argument(
        "--tokenizer_id",
        default=None,
        help="Tokenization all collections share. Defaults to a surrogate "
             "derived from vocabulary size, which trajectory metadata does "
             "record.",
    )
    parser.add_argument(
        "--tier",
        action="append",
        default=[],
        metavar="MODEL_ID=DEPTH",
        help="Tier a model represents. Defaults to its deepest collected tier.",
    )
    args = parser.parse_args(argv)

    def pairs(items: list[str], flag: str) -> dict[str, str]:
        out = {}
        for item in items:
            if "=" not in item:
                parser.error(f"{flag} expects MODEL_ID=VALUE, got {item!r}.")
            key, value = item.split("=", 1)
            out[key] = value
        return out

    args.independent = pairs(args.independent, "--independent")
    args.tier = {k: int(v) for k, v in pairs(args.tier, "--tier").items()}
    return args


def main(argv: list[str] | None = None) -> None:
    """Builds the manifest and reports what went into it.

    Args:
        argv: Argument list, or ``None`` to read ``sys.argv``.
    """
    args = parse_args(argv)
    path = build(
        vertical=args.vertical,
        independent=args.independent,
        out_dir=args.out,
        quality_metric=args.quality_metric,
        cost_metric=args.cost_metric,
        tiers=args.tier,
        tokenizer_id=args.tokenizer_id,
    )
    entries = json.loads(path.read_text())

    print(f"wrote {path}")
    for entry in entries:
        print(
            f"  {entry['model_id']:<20} tier {entry['tier']:>3}  "
            f"{entry['n_requests']} requests  "
            f"mean {args.quality_metric} "
            f"{np.mean(json.loads(Path(entry['results']).read_text())):.4f}"
        )
    print(
        "\nPass it to the evaluation with "
        f"--manifest={path} to get the sharing tax."
    )


if __name__ == "__main__":
    main()
