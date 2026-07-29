"""The predeclared test that a retrofit did not damage its parent.

Hypothesis H1 says the retrofitted model's final endpoint is non-inferior to its
frozen parent within a predeclared margin. Under a frozen retrofit that holds by
construction and :func:`src.retrofit.assert_parent_preserved` proves it to the
bit. Under `selective_unfreeze`, `lora`, `qlora` or `full_finetune` it does not
hold by construction, and until this command existed there was nothing that
tested it — `retrofit_parent.py` printed the words "must be tested for
non-inferiority on held-out data" with no implementation behind them.

Those are precisely the modes anyone would reach for if frozen exits proved too
weak, which is the reason the ladder has rungs above the bottom one.

What this does, and the three things it refuses to do:

* Scores both models' **full-depth** endpoint on the same real held-out requests.
  Not the shallow exits — H1 is about the endpoint being preserved, and a
  retrofit that improved its shallow tiers while moving the endpoint has failed
  regardless of how the average looks.
* Runs a **one-sided** non-inferiority test. It does not report "the interval
  contains zero", which only says the study was too small to tell the models
  apart and would let an underpowered run support preservation.
* Resamples **documents**, not requests, so the interval is not narrowed by
  correlation between requests cut from the same document.
* States the margin as an input, never chosen after seeing the estimate.

Run it::

    python -m experiments.no_regret \\
        --parent checkpoints/vr-noexits/final.pt \\
        --candidate runs/retrofit-lora/checkpoints/retrofit.pt \\
        --data data/val.bin --eos_id 50256 \\
        --quality_margin 0.01 --run-dir runs/no-regret-lora
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.collect_depth_trajectories import (
    check_corpus_compatible,
    parse_shapes,
)
from experiments.workloads import Workload, real_text_corpus
from src.model import Transformer
from src.retrofit import LoRALinear, restore, set_lora_enabled
from utils.provenance import RunArtifacts, Seeds, file_digest
from utils.statistics import non_inferiority_test, paired_bootstrap

#: Version of the record this command writes.
SCHEMA_VERSION = 1


def load_model(path: str | Path) -> tuple[Transformer, dict]:
    """Loads a checkpoint as a frozen model for scoring.

    Args:
        path: Checkpoint written by ``training.train`` or
            ``experiments.retrofit_parent``.

    Returns:
        A tuple ``(model, blob)``.

    Raises:
        FileNotFoundError: If the checkpoint is absent.
        KeyError: If it carries no architecture.
        ValueError: If loading would leave part of the backbone at its random
            initialization, which would make every number below describe an
            untrained model without anything raising.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    blob = torch.load(path, map_location="cpu", weights_only=False)
    # restore() replays any recorded retrofit before loading, which is what makes
    # a LoRA checkpoint loadable: wrapping a projection renames its weight, so a
    # plain Transformer sees every wrapped projection as a missing key.
    return restore(path), blob


@torch.no_grad()
def score_endpoint(model: Transformer, workload: Workload) -> dict[str, np.ndarray]:
    """Scores a workload's reference continuation at full depth.

    Args:
        model: Frozen model.
        workload: Requests whose continuations are scored.

    Returns:
        Per-request ``nll`` (mean over scored positions), ``nll_sum``, and a
        scalar ``valid_tokens``. Both forms are returned because the corpus
        number is a ratio of totals while the paired test operates on
        per-request values.
    """
    sequences = workload.sequences()
    inputs, targets = sequences[:, :-1], sequences[:, 1:]
    start = workload.prompt_len - 1

    logits = model(inputs).logits[:, start:]
    gold = targets[:, start:]
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        gold.reshape(-1),
        reduction="none",
    ).view_as(gold)

    return {
        "nll": per_token.mean(dim=1).numpy(),
        "nll_sum": per_token.sum(dim=1).numpy(),
        "valid_tokens": int(gold.size(1)),
    }


def compare(
    parent: Transformer,
    candidate: Transformer,
    splits: list[tuple[str, Workload]],
) -> dict:
    """Scores both models on identical requests and pairs the results.

    Args:
        parent: The frozen reference.
        candidate: The retrofitted model.
        splits: Shape-bucketed workloads to score.

    Returns:
        A mapping with per-request arrays for both models, the document label of
        each request, and the corpus NLL of each model formed from summed
        numerators and denominators.
    """
    parent_nll, candidate_nll = [], []
    parent_sums, candidate_sums, token_counts = [], [], []
    documents, shapes = [], []

    for _, subset in splits:
        if not len(subset):
            continue
        left = score_endpoint(parent, subset)
        right = score_endpoint(candidate, subset)

        parent_nll.append(left["nll"])
        candidate_nll.append(right["nll"])
        parent_sums.append(left["nll_sum"])
        candidate_sums.append(right["nll_sum"])
        token_counts.append(
            np.full(len(subset), left["valid_tokens"], dtype=float)
        )
        documents.extend(
            subset.document_ids
            if subset.document_ids is not None
            else subset.source_ids
        )
        shapes.extend(
            [f"p{subset.prompt_len}c{subset.continuation_len}"] * len(subset)
        )

    if not parent_nll:
        raise ValueError("no requests were scored; nothing to compare.")

    counts = np.concatenate(token_counts)
    return {
        "parent_nll": np.concatenate(parent_nll),
        "candidate_nll": np.concatenate(candidate_nll),
        "documents": np.asarray(documents),
        "shapes": shapes,
        "valid_tokens": counts,
        "parent_corpus_nll": float(np.concatenate(parent_sums).sum() / counts.sum()),
        "candidate_corpus_nll": float(
            np.concatenate(candidate_sums).sum() / counts.sum()
        ),
    }


def test_preservation(
    paired: dict,
    quality_margin: float,
    resamples: int,
    seed: int,
) -> dict:
    """Runs the one-sided non-inferiority test on the endpoint.

    Quality is negative NLL, so a *higher* value is better and the candidate is
    non-inferior when the lower confidence bound on ``-(NLL_c - NLL_p)`` exceeds
    ``-quality_margin``. Cost is identical by construction — both models run the
    same architecture at full depth — so the cost arm of the test is trivially
    satisfied and is reported rather than silently dropped.

    Args:
        paired: Output of :func:`compare`.
        quality_margin: Largest acceptable NLL increase, in nats. Predeclared.
        resamples: Bootstrap replicates.
        seed: Bootstrap seed.

    Returns:
        A mapping with the test result, the clustered interval on the NLL
        difference, and the verdict.
    """
    delta = paired["candidate_nll"] - paired["parent_nll"]
    clusters = paired["documents"]

    nll_interval = paired_bootstrap(
        delta, resamples=resamples, seed=seed, clusters=clusters
    )
    zeros = np.zeros_like(delta)
    result = non_inferiority_test(
        -paired["candidate_nll"],
        -paired["parent_nll"],
        zeros,
        zeros,
        quality_margin=quality_margin,
        cost_tolerance=0.0,
        resamples=resamples,
        seed=seed,
        clusters=clusters,
    )

    return {
        "nll_difference": asdict(nll_interval),
        "quality_difference": asdict(result.quality_difference),
        "quality_margin": quality_margin,
        "passes": bool(result.quality_passes),
        "n_requests": int(delta.size),
        "n_documents": int(len(set(clusters.tolist()))),
        "perplexity_ratio": float(np.exp(nll_interval.estimate)),
    }


def check_recoverable(candidate: Transformer, parent: Transformer, probe) -> dict:
    """Checks that disabling any low-rank updates recovers the parent exactly.

    A LoRA retrofit keeps its reference computable rather than remembered. If
    switching the adapters off does *not* reproduce the parent, then something
    outside the adapters moved — and the non-inferiority result above is measuring
    a different intervention than the one being reported.

    Args:
        candidate: The retrofitted model.
        parent: The frozen reference.
        probe: Token ids to compare on.

    Returns:
        A mapping describing what was checked. ``applicable`` is ``False`` when
        the candidate holds no low-rank updates.
    """
    wrapped = sum(
        1 for module in candidate.modules() if isinstance(module, LoRALinear)
    )
    if not wrapped:
        return {"applicable": False, "lora_modules": 0}

    set_lora_enabled(candidate, False)
    with torch.no_grad():
        difference = float(
            (candidate(probe).logits - parent(probe).logits).abs().max()
        )
    set_lora_enabled(candidate, True)

    return {
        "applicable": True,
        "lora_modules": wrapped,
        "max_logit_difference": difference,
        "exact": difference == 0.0,
        "note": (
            "with the updates disabled the model is the parent to the bit"
            if difference == 0.0
            else "disabling the updates did NOT recover the parent, so something "
            "outside the adapters moved and the comparison above is not the "
            "intervention it claims to be"
        ),
    }


def report(results: dict) -> str:
    """Renders the verdict, stating it only when it was earned.

    Args:
        results: The assembled results.

    Returns:
        A markdown fragment.
    """
    test = results["non_inferiority"]
    interval = test["nll_difference"]
    lines = [
        "## No-regret test on the full-depth endpoint",
        "",
        f"| | NLL |",
        f"|---|---:|",
        f"| parent | {results['parent_corpus_nll']:.4f} |",
        f"| candidate | {results['candidate_corpus_nll']:.4f} |",
        f"| difference | {interval['estimate']:+.4f} |",
        "",
        f"Corpus NLL is `sum(nll) / sum(valid_tokens)`, not a mean of "
        f"per-request means: the requests span "
        f"{len(set(results['shapes']))} shape(s) and averaging means would "
        f"weight a short request the same as a long one.",
        "",
        f"Paired difference, resampling **documents** "
        f"({test['n_documents']} of them across {test['n_requests']} requests): "
        f"{interval['estimate']:+.4f} "
        f"[{interval['low']:+.4f}, {interval['high']:+.4f}]",
        "",
        f"Perplexity ratio {test['perplexity_ratio']:.4f}.",
        "",
        f"Predeclared margin: the candidate may lose at most "
        f"{test['quality_margin']:.4f} nats.",
        "",
    ]
    if test["passes"]:
        lines += [
            "**Non-inferior.** The lower bound on the quality difference lies "
            f"above the margin, so preservation is supported at this margin on "
            "this corpus.",
        ]
    else:
        lines += [
            "**Not non-inferior.** The lower bound does not clear the margin. "
            "This is not the same as a detected regression: it can also mean "
            "the sample is too small. Either way the no-regret claim is not "
            "supported and must not be made.",
        ]

    recoverable = results.get("recoverable", {})
    if recoverable.get("applicable"):
        lines += ["", f"Adapters disabled: {recoverable['note']}."]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the command line.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Test that a retrofit preserved its parent's endpoint."
    )
    parser.add_argument("--parent", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--data", default="data/val.bin")
    parser.add_argument("--eos_id", type=int, default=None)
    parser.add_argument("--shapes", default="64:32,128:64,256:128")
    parser.add_argument("--n_requests", type=int, default=1024)
    parser.add_argument(
        "--quality_margin",
        type=float,
        default=0.01,
        help=(
            "Largest acceptable NLL increase in nats. Declare it before "
            "looking at the estimate."
        ),
    )
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Runs the test end to end.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when the test passes, ``1`` when it does not. The exit status is
        the verdict, so a job script can gate on it rather than on a human
        reading a table.
    """
    args = parse_args(argv)

    parent, _ = load_model(args.parent)
    candidate, candidate_blob = load_model(args.candidate)

    shapes = parse_shapes(
        tuple(part.strip() for part in args.shapes.split(",") if part.strip())
    )
    buckets, corpus_metadata = real_text_corpus(
        args.data,
        shapes=shapes,
        n_requests=args.n_requests,
        eos_id=args.eos_id,
        seed=Seeds.derive(args.seed).benchmark,
    )
    splits = [("report", bucket) for bucket in buckets]
    check_corpus_compatible(parent.config, splits, corpus_metadata)
    check_corpus_compatible(candidate.config, splits, corpus_metadata)

    print(f"parent      {args.parent}")
    print(f"candidate   {args.candidate}")
    print(f"corpus      {corpus_metadata['requests_drawn']} requests, "
          f"{corpus_metadata['documents_found']} documents")
    print(f"margin      {args.quality_margin:.4f} nats (predeclared)")

    if args.dry_run:
        print("dry run: validated, nothing scored")
        return 0

    paired = compare(parent, candidate, splits)
    test = test_preservation(
        paired, args.quality_margin, args.resamples, Seeds.derive(args.seed).benchmark
    )

    probe = buckets[0].sequences()[: min(4, len(buckets[0]))]
    results = {
        "schema_version": SCHEMA_VERSION,
        "parent": {"path": args.parent, "sha256": file_digest(args.parent)},
        "candidate": {
            "path": args.candidate,
            "sha256": file_digest(args.candidate),
            "retrofit": candidate_blob.get("retrofit"),
        },
        "corpus": corpus_metadata,
        "shapes": paired["shapes"],
        "parent_corpus_nll": paired["parent_corpus_nll"],
        "candidate_corpus_nll": paired["candidate_corpus_nll"],
        "non_inferiority": test,
        "recoverable": check_recoverable(candidate, parent, probe),
    }

    print()
    print(report(results))

    if args.run_dir:
        artifacts = RunArtifacts.create(
            args.run_dir,
            script="experiments.no_regret",
            config=vars(args),
            seeds=Seeds.derive(args.seed),
            inputs={"parent": args.parent, "candidate": args.candidate,
                    "corpus": args.data},
            parent_checkpoint=args.parent,
            required=(),
        )
        artifacts.log_metric(results)
        (artifacts.run_dir / "no_regret.md").write_text(report(results) + "\n")
        print(f"\nrecord      {artifacts.run_dir}")

    return 0 if test["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
