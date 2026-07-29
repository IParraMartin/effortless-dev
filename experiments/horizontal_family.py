"""Scoring a family of independently trained models on the same requests.

This is the horizontal side of the central comparison. Until it exists there is
nothing to compare vertical routing *against*: the evaluation reports every
horizontal estimand as unavailable, and the `vertical_no_reuse_oracle_cascade`
is a same-backbone diagnostic that must not be read as a model cascade.

The Pythia suite is the right family for this. Every size shares one training
corpus, one tokenizer, one architecture and one data order, so a quality
difference between two of them is a capacity difference and not a confound. A
comparison against, say, GPT-2 small and Llama would confound capacity with
corpus, tokenizer and recipe simultaneously.

The tokenizer problem, and why it decides the metric
----------------------------------------------------

Pythia uses the GPT-NeoX tokenizer; the backbone in this repository was trained
with GPT-2's. The same string becomes different token sequences — ``"Hello
world"`` is ``[15496, 995]`` under GPT-2 and ``[12092, 1533]`` under NeoX — so
per-token loss is **not** comparable between them. A tokenizer that splits text
more finely earns a lower average loss per piece without predicting anything
better, so comparing per-token NLL across families rewards whichever tokenizer is
more granular.

The fix is to take the tokenizer out of the denominator::

    bits_per_byte = nll_sum_nats / (ln(2) * utf8_bytes(continuation))

Bytes are a property of the text, not of anyone's vocabulary. So this module
takes the *decoded text* of each request, re-encodes it with the family's own
tokenizer, scores only the continuation, and reports bits per byte. Request
identity is checked by digesting the continuation text: two models that report
the same ``text_sha256`` scored the same content, whatever ids they used to do it.

What it refuses to do
---------------------

* It will not emit a manifest without byte lengths. Without them the only
  available quality is per-token, which is the comparison that is invalid.
* It will not report a scalar cost per model. Cost depends on request shape —
  prefill scales with the prompt, decode with the growing context — so the
  manifest carries a profile keyed by shape.
* It will not silently skip a model it could not load. A family with a missing
  member is reported as incomplete, because a frontier drawn through the models
  that happened to download is not the family's frontier.

Run it::

    python -m experiments.horizontal_family \\
        --models EleutherAI/pythia-70m,EleutherAI/pythia-160m,EleutherAI/pythia-410m \\
        --data data/val.bin --eos_id 50256 \\
        --out results/pythia --run-dir runs/pythia
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from experiments.collect_depth_trajectories import parse_shapes
from experiments.workloads import RequestShape, Workload, real_text_corpus
from utils.provenance import RunArtifacts, Seeds, digest_text, hardware

#: Version of the manifest layout this module writes.
MANIFEST_SCHEMA_VERSION = 2

#: The Pythia suite, ascending. Every member shares corpus, tokenizer,
#: architecture family and data order, which is what makes a difference between
#: two of them attributable to capacity.
PYTHIA_SUITE = (
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
)

#: Family metadata recorded with every manifest, so a reader knows what is held
#: constant across the tiers and therefore what a difference between them means.
PYTHIA_FAMILY = {
    "family": "pythia",
    "corpus": "the Pile, deduplicated or not depending on the variant",
    "tokenizer": "GPT-NeoX-20B BPE",
    "held_constant": [
        "training corpus",
        "data order",
        "tokenizer",
        "architecture family",
        "training recipe",
    ],
    "varies": ["parameter count", "depth", "width"],
    "note": (
        "quality differences between tiers are capacity differences. The "
        "comparison against this repository's backbone is *not* controlled: it "
        "differs in corpus, tokenizer and recipe as well as capacity, which is "
        "why quality is reported in bits per byte and why the sharing tax "
        "estimated against it is an upper bound on what sharing costs."
    ),
}


@dataclass
class FamilyConfig:
    """Settings for one horizontal scoring run.

    Attributes:
        models: Model ids, ascending in capacity.
        data: Tokenized corpus the requests are drawn from.
        eos_id: End-of-text token of *that* corpus, for document boundaries.
        shapes: Request shapes, matching the vertical collection.
        n_requests: Requests to draw.
        out: Directory for per-model results and the manifest.
        seed: Seed for the draw. Must match the vertical collection's, or the
            two sides are not scored on the same requests.
        device: Device to score on.
        dtype: Precision to load weights in.
        batch_size: Requests scored per forward pass.
        trajectories: Optional vertical trajectory directory. When given, the
            drawn requests are checked against its records so a mismatch is an
            error rather than a silently unpaired comparison.
    """

    models: tuple[str, ...] = PYTHIA_SUITE[:3]
    data: str = "data/val.bin"
    eos_id: int | None = None
    shapes: tuple[str, ...] = ("64:32", "128:64", "256:128")
    n_requests: int = 1024
    out: str = "results/horizontal"
    seed: int = 0
    device: str = "cpu"
    dtype: str = "fp32"
    batch_size: int = 8
    trajectories: str | None = None


@dataclass
class ModelResult:
    """What one independent model achieved and cost.

    Attributes:
        model_id: Hub id.
        tokenizer_id: Tokenizer it used.
        parameters: Total parameter count.
        resident_bytes: Weight bytes at the loaded precision.
        bits_per_byte: Per-request quality, lower is better.
        text_hashes: Digest of each continuation, proving what was scored.
        cost_profile: Measured MACs and wall time per request shape.
        tier: Rank within the family, ascending in capacity.
    """

    model_id: str
    tokenizer_id: str
    parameters: int
    resident_bytes: int
    bits_per_byte: list[float]
    text_hashes: list[str]
    cost_profile: dict[str, dict[str, float]] = field(default_factory=dict)
    tier: int = 0


#: Text encoded to fingerprint a tokenizer. Fixed, and deliberately mixed —
#: punctuation, casing, a number and a word that BPE splits — so two tokenizers
#: that differ in any of those respects produce different ids.
TOKENIZER_PROBE = "The mitochondrion, in 1890, was called a bioblast; 42 units."


def tokenizer_fingerprint(tokenizer) -> str:
    """Identifies a tokenizer by what it *does*, not by where it was loaded from.

    ``tokenizer.name_or_path`` is the model id, so every Pythia size reports a
    different value while sharing one tokenizer — which made the family look
    internally inconsistent and would have rejected a perfectly controlled
    comparison. Conversely two different tokenizers could be loaded from paths
    that happen to look alike.

    Args:
        tokenizer: The tokenizer to identify.

    Returns:
        A string combining the implementation class, the vocabulary size, and a
        digest of the ids produced for :data:`TOKENIZER_PROBE`. Two tokenizers
        agreeing on all three are interchangeable for the purpose of comparing
        per-token quantities.
    """
    ids = tokenizer(TOKENIZER_PROBE, return_tensors=None)["input_ids"]
    return (
        f"{type(tokenizer).__name__}"
        f":{tokenizer.vocab_size}"
        f":{digest_text(','.join(str(value) for value in ids))[:16]}"
    )


def load_family_model(model_id: str, device: str, dtype: str):
    """Loads one member of the family.

    Args:
        model_id: Hub id.
        device: Device to place it on.
        dtype: ``"fp32"``, ``"bf16"`` or ``"fp16"``.

    Returns:
        A tuple ``(model, tokenizer)``, the model in eval mode with gradients
        disabled.

    Raises:
        ImportError: If ``transformers`` is unavailable, which is a dependency of
            this module alone and not of the repository.
        OSError: If the weights cannot be obtained. Not caught: a family with a
            missing member must not be reported as a complete frontier.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "experiments.horizontal_family needs `transformers`. It is not "
            "imported anywhere else in this repository, so the rest of the "
            "pipeline runs without it."
        ) from error

    torch_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    model.to(device).eval().requires_grad_(False)
    return model, tokenizer


@torch.no_grad()
def score_bits_per_byte(
    model,
    tokenizer,
    workload: Workload,
    batch_size: int = 8,
    device: str = "cpu",
) -> dict:
    """Scores each request's continuation in bits per byte.

    The prompt and continuation are re-encoded with *this* model's tokenizer, so
    the boundary between them is found in token space for this tokenizer rather
    than assumed from another one's. Only continuation positions contribute to the
    loss, and the denominator is the continuation's UTF-8 length, which no
    tokenizer can change.

    Args:
        model: A causal language model.
        tokenizer: Its tokenizer.
        workload: Requests carrying decoded text and byte lengths.
        batch_size: Requests per forward pass.
        device: Device to score on.

    Returns:
        A mapping with per-request ``bits_per_byte``, ``nll_sum``,
        ``scored_tokens`` and the wall time spent.

    Raises:
        ValueError: If the workload was not decoded to text, since then there is
            no string to re-encode and no byte count to divide by.
    """
    if workload.texts is None or workload.continuation_bytes is None:
        raise ValueError(
            "this workload carries no decoded text, so it cannot be scored by a "
            "model from another tokenizer family. Draw it with a tokenizer_name "
            "so continuation byte lengths are available."
        )

    per_request_bpb: list[float] = []
    per_request_nll: list[float] = []
    scored_tokens: list[int] = []
    started = time.perf_counter()

    for start in range(0, len(workload), batch_size):
        rows = range(start, min(start + batch_size, len(workload)))
        for row in rows:
            prompt, continuation = workload.texts[row]
            prompt_ids = tokenizer(prompt, return_tensors=None)["input_ids"]
            full_ids = tokenizer(prompt + continuation, return_tensors=None)[
                "input_ids"
            ]
            boundary = len(prompt_ids)
            if boundary >= len(full_ids):
                # Re-encoding merged the boundary token, leaving no continuation
                # to score. Recorded as missing rather than as zero loss.
                per_request_bpb.append(float("nan"))
                per_request_nll.append(float("nan"))
                scored_tokens.append(0)
                continue

            ids = torch.tensor([full_ids], device=device)
            logits = model(ids).logits[0, :-1].float()
            gold = ids[0, 1:]
            per_token = torch.nn.functional.cross_entropy(
                logits, gold, reduction="none"
            )
            # Positions predicting continuation tokens: the token at index
            # `boundary` is the first continuation token, predicted from index
            # boundary - 1.
            tail = per_token[boundary - 1 :]
            total = float(tail.sum())

            per_request_nll.append(total)
            scored_tokens.append(int(tail.numel()))
            per_request_bpb.append(
                total / (math.log(2.0) * workload.continuation_bytes[row])
            )

    return {
        "bits_per_byte": per_request_bpb,
        "nll_sum": per_request_nll,
        "scored_tokens": scored_tokens,
        "seconds": time.perf_counter() - started,
    }


def measure_cost_profile(
    model,
    tokenizer,
    shape: RequestShape,
    parameters: int,
    seconds: float,
    requests: int,
) -> dict[str, float]:
    """Describes what this model costs at one request shape.

    A scalar cost per model cannot support a matched-cost comparison: prefill
    scales with the prompt and decode with the growing context, so two models
    ranked one way on a short request can rank the other way on a long one.

    Args:
        model: The scored model.
        tokenizer: Its tokenizer, for the vocabulary size.
        shape: The request shape measured.
        parameters: Parameter count.
        seconds: Wall time spent scoring this shape.
        requests: Requests scored at this shape.

    Returns:
        A mapping with an analytical MAC estimate and the measured wall time per
        request, kept apart because a MAC count is not a latency and must not be
        substituted for one.
    """
    config = model.config
    layers = getattr(config, "num_hidden_layers", 0)
    width = getattr(config, "hidden_size", 0)
    vocab = getattr(config, "vocab_size", len(tokenizer))
    total = shape.total

    # Two matmul-dominated terms per block (attention projections and MLP), plus
    # the attention score/value products, plus one vocabulary projection per
    # scored position. Deliberately coarse: it is an analytical estimate, and the
    # measured time beside it is the one that supports a systems claim.
    per_token_block = 4.0 * width * width + 8.0 * width * width
    attention = 2.0 * width * total
    forward = total * layers * per_token_block + layers * attention * total
    head = total * width * vocab

    return {
        "analytical_macs": float(forward + head),
        "parameters": float(parameters),
        "seconds_per_request": seconds / max(requests, 1),
        "prompt_len": float(shape.prompt_len),
        "continuation_len": float(shape.continuation_len),
    }


def score_family(config: FamilyConfig) -> tuple[list[ModelResult], dict]:
    """Scores every member of the family on identical requests.

    Args:
        config: Run settings.

    Returns:
        A tuple ``(results, metadata)``.

    Raises:
        ValueError: If the corpus could not be decoded to text, if the family
            disagrees about its tokenizer, or if the requests do not match the
            vertical side's.
    """
    shapes = parse_shapes(config.shapes)
    buckets, corpus_metadata = real_text_corpus(
        config.data,
        shapes=shapes,
        n_requests=config.n_requests,
        eos_id=config.eos_id,
        seed=config.seed,
    )
    if corpus_metadata.get("quality_unit") != "bits_per_byte":
        raise ValueError(
            f"the corpus could not be decoded to text "
            f"({corpus_metadata.get('decode_note', 'no reason recorded')}). "
            f"Without byte lengths the only available quality is per token, "
            f"which is not comparable across tokenizer families — which is the "
            f"entire comparison this module exists to make."
        )

    if config.trajectories:
        check_requests_match(buckets, config.trajectories)

    results: list[ModelResult] = []
    for tier, model_id in enumerate(config.models):
        print(f"scoring {model_id} ...", flush=True)
        model, tokenizer = load_family_model(model_id, config.device, config.dtype)
        parameters = sum(p.numel() for p in model.parameters())
        resident = sum(p.numel() * p.element_size() for p in model.parameters())

        quality: list[float] = []
        hashes: list[str] = []
        profile: dict[str, dict[str, float]] = {}
        for bucket, shape in zip(buckets, shapes):
            scored = score_bits_per_byte(
                model, tokenizer, bucket, config.batch_size, config.device
            )
            quality.extend(scored["bits_per_byte"])
            hashes.extend(bucket.text_hashes)
            profile[shape.label()] = measure_cost_profile(
                model, tokenizer, shape, parameters, scored["seconds"], len(bucket)
            )
            finite = [v for v in scored["bits_per_byte"] if v == v]
            print(
                f"  {shape.label()}: {len(bucket)} requests, "
                f"bits/byte {np.mean(finite):.4f}"
                f"{'' if len(finite) == len(bucket) else f' ({len(bucket) - len(finite)} unscorable)'}",
                flush=True,
            )

        results.append(
            ModelResult(
                model_id=model_id,
                tokenizer_id=tokenizer_fingerprint(tokenizer),
                parameters=parameters,
                resident_bytes=resident,
                bits_per_byte=quality,
                text_hashes=hashes,
                cost_profile=profile,
                tier=tier,
            )
        )
        del model

    _check_one_tokenizer(results)
    _check_same_requests(results)

    metadata = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "corpus": corpus_metadata,
        "family": PYTHIA_FAMILY if _is_pythia(config.models) else {
            "family": "custom",
            "note": (
                "family metadata is not recorded for this model set, so what is "
                "held constant across its tiers is unknown and a difference "
                "between them cannot be attributed to capacity"
            ),
        },
        "quality_unit": "bits_per_byte",
        "quality_direction": "lower_is_better",
        "hardware": hardware(),
        "dtype": config.dtype,
        "requests": sum(len(bucket) for bucket in buckets),
        "shapes": [shape.label() for shape in shapes],
        "seed": config.seed,
    }
    return results, metadata


def check_requests_match(buckets: list[Workload], trajectories: str | Path) -> None:
    """Verifies the horizontal side drew the vertical side's requests.

    Args:
        buckets: The drawn workloads.
        trajectories: Vertical trajectory directory.

    Raises:
        ValueError: If the trajectory records carry no text digests, or if the
            digests disagree. A paired comparison over different requests is not
            a paired comparison, and the failure is invisible in the output.
    """
    from experiments.collect_depth_trajectories import load

    records, _, _ = load(trajectories)
    recorded = [record.get("text_sha256", "") for record in records]
    if not any(recorded):
        raise ValueError(
            f"{trajectories} carries no text digests, so it cannot be proved to "
            f"hold the same requests. Re-collect it with --corpus real_text on a "
            f"corpus whose tokenizer could be loaded (schema 3)."
        )

    drawn = [h for bucket in buckets for h in bucket.text_hashes]
    if sorted(recorded) != sorted(drawn):
        overlap = len(set(recorded) & set(drawn))
        raise ValueError(
            f"the drawn requests do not match {trajectories}: {overlap} of "
            f"{len(drawn)} digests are shared. The two sides must be scored on "
            f"identical requests, which means the same --data, --eos_id, "
            f"--shapes, --n_requests and --seed."
        )


def write_manifest(
    results: list[ModelResult],
    metadata: dict,
    out: str | Path,
) -> Path:
    """Writes per-model quality files and the manifest that indexes them.

    Args:
        results: Scored models.
        metadata: Run metadata.
        out: Output directory.

    Returns:
        Path to the manifest.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    entries = []
    for result in results:
        slug = result.model_id.replace("/", "_")
        quality_path = out / f"{slug}.json"
        quality_path.write_text(json.dumps(result.bits_per_byte))
        (out / f"{slug}.hashes.json").write_text(json.dumps(result.text_hashes))

        deepest = max(
            entry["analytical_macs"] for entry in result.cost_profile.values()
        )
        entries.append(
            {
                "model_id": result.model_id,
                "tokenizer_id": result.tokenizer_id,
                "tier": result.tier,
                # Retained for readers of the older manifest schema. The profile
                # below is the one a matched-cost comparison must use.
                "cost": deepest,
                "cost_profile": result.cost_profile,
                "results": str(quality_path),
                "request_hashes": str(out / f"{slug}.hashes.json"),
                "parameters": result.parameters,
                "resident_bytes": result.resident_bytes,
                "quality_unit": metadata["quality_unit"],
                "quality_direction": metadata["quality_direction"],
                "family": metadata["family"],
                "hardware": metadata["hardware"],
            }
        )

    manifest = out / "manifest.json"
    manifest.write_text(json.dumps(entries, indent=2))
    (out / "family.json").write_text(json.dumps(metadata, indent=2))
    return manifest


def report(results: list[ModelResult], metadata: dict) -> str:
    """Renders the family's frontier.

    Args:
        results: Scored models.
        metadata: Run metadata.

    Returns:
        A markdown fragment.
    """
    lines = [
        "## Independent model family",
        "",
        f"Family: **{metadata['family'].get('family')}**. "
        f"Quality is **bits per byte**, lower better — the only unit comparable "
        f"against a model that uses a different tokenizer.",
        "",
        "| model | parameters | resident | bits/byte |",
        "|---|---:|---:|---:|",
    ]
    for result in results:
        finite = [v for v in result.bits_per_byte if v == v]
        lines.append(
            f"| {result.model_id} | {result.parameters / 1e6:,.1f}M | "
            f"{result.resident_bytes / 1e6:,.0f} MB | "
            f"{np.mean(finite):.4f} |"
        )

    lines += [
        "",
        "Cost is a profile keyed by request shape, not a scalar: prefill scales "
        "with the prompt and decode with the growing context, so two models "
        "ranked one way on a short request can rank the other way on a long one.",
        "",
    ]
    held = metadata["family"].get("held_constant")
    if held:
        lines += [
            f"Held constant across tiers: {', '.join(held)}. A quality "
            f"difference between two tiers is therefore a capacity difference.",
            "",
            metadata["family"]["note"],
        ]
    return "\n".join(lines)


def _is_pythia(models: tuple[str, ...]) -> bool:
    """Whether every model named is a Pythia suite member."""
    return all("pythia" in model.lower() for model in models)


def _check_one_tokenizer(results: list[ModelResult]) -> None:
    """Rejects a family whose members disagree about the tokenizer.

    Args:
        results: Scored models.

    Raises:
        ValueError: If more than one tokenizer appears. Within a family the
            tokenizer must be shared, or a difference between tiers is not a
            capacity difference.
    """
    tokenizers = {result.tokenizer_id for result in results}
    if len(tokenizers) > 1:
        raise ValueError(
            f"the family mixes tokenizers {sorted(tokenizers)}. Within a family "
            f"the tokenizer must be constant, or a quality difference between "
            f"tiers confounds capacity with tokenization. The identity is a "
            f"behavioural fingerprint, so this means the tokenizers genuinely "
            f"encode text differently -- not merely that they were loaded from "
            f"different paths."
        )


def _check_same_requests(results: list[ModelResult]) -> None:
    """Rejects a family whose members scored different requests.

    Args:
        results: Scored models.

    Raises:
        ValueError: If the request digests differ between models.
    """
    reference = results[0]
    for result in results[1:]:
        if result.text_hashes != reference.text_hashes:
            raise ValueError(
                f"{result.model_id} scored different requests from "
                f"{reference.model_id}. Every model must see identical content "
                f"for the comparison to be paired."
            )


def parse_args(argv: list[str] | None = None) -> FamilyConfig:
    """Parses the command line.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        The configuration.
    """
    parser = argparse.ArgumentParser(
        description="Score an independently trained model family on shared requests."
    )
    parser.add_argument(
        "--models",
        default=",".join(PYTHIA_SUITE[:3]),
        help="Comma-separated hub ids, ascending in capacity.",
    )
    parser.add_argument("--data", default="data/val.bin")
    parser.add_argument("--eos_id", type=int, default=None)
    parser.add_argument("--shapes", default="64:32,128:64,256:128")
    parser.add_argument("--n_requests", type=int, default=1024)
    parser.add_argument("--out", default="results/horizontal")
    parser.add_argument("--run-dir", dest="run_dir", default=None)
    parser.add_argument(
        "--trajectories",
        default=None,
        help=(
            "Vertical trajectory directory. Supplying it checks that both sides "
            "scored identical requests, which a paired comparison requires."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="fp32", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parsed = parser.parse_args(argv)

    config = FamilyConfig(
        models=tuple(
            part.strip() for part in parsed.models.split(",") if part.strip()
        ),
        data=parsed.data,
        eos_id=parsed.eos_id,
        shapes=tuple(
            part.strip() for part in parsed.shapes.split(",") if part.strip()
        ),
        n_requests=parsed.n_requests,
        out=parsed.out,
        seed=parsed.seed,
        device=parsed.device,
        dtype=parsed.dtype,
        batch_size=parsed.batch_size,
        trajectories=parsed.trajectories,
    )
    config.run_dir = parsed.run_dir
    config.dry_run = parsed.dry_run
    return config


def main(argv: list[str] | None = None) -> int:
    """Scores the family and writes its manifest.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit status.
    """
    config = parse_args(argv)

    if getattr(config, "dry_run", False):
        shapes = parse_shapes(config.shapes)
        buckets, corpus_metadata = real_text_corpus(
            config.data,
            shapes=shapes,
            n_requests=config.n_requests,
            eos_id=config.eos_id,
            seed=config.seed,
        )
        print(f"models      {', '.join(config.models)}")
        print(f"requests    {sum(len(b) for b in buckets)} across "
              f"{len(buckets)} shape(s)")
        print(f"quality     {corpus_metadata.get('quality_unit')}")
        if corpus_metadata.get("quality_unit") != "bits_per_byte":
            print(f"REFUSED     {corpus_metadata.get('decode_note')}")
            return 1
        if config.trajectories:
            check_requests_match(buckets, config.trajectories)
            print("paired      request digests match the vertical side")
        print("dry run: validated, no model loaded")
        return 0

    results, metadata = score_family(config)
    manifest = write_manifest(results, metadata, config.out)

    print()
    print(report(results, metadata))
    print(f"\nmanifest    {manifest}")

    if getattr(config, "run_dir", None):
        artifacts = RunArtifacts.create(
            config.run_dir,
            script="experiments.horizontal_family",
            config=asdict(config),
            seeds=Seeds.derive(config.seed),
            inputs={"corpus": config.data, "manifest": str(manifest)},
            required=(),
        )
        artifacts.log_metric(
            {
                "models": [result.model_id for result in results],
                "parameters": [result.parameters for result in results],
                "bits_per_byte": [
                    float(np.nanmean(result.bits_per_byte)) for result in results
                ],
                "family": metadata["family"].get("family"),
                "requests": metadata["requests"],
            }
        )
        (artifacts.run_dir / "family.md").write_text(
            report(results, metadata) + "\n"
        )
        print(f"record      {artifacts.run_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
