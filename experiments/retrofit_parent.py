"""Turns a trained parent into an elastic model with shallow endpoints.

This is the entry point for the no-regret retrofit ladder. It loads a parent
checkpoint, adds exits, freezes or adapts what the chosen mode says, verifies the
parent's full-depth output where that verification is exact, and writes the
result with complete provenance.

It does not train. Training the retrofitted model is ``training.train`` with
``--resume_from`` pointed at the checkpoint this writes, which keeps the
adaptation decision and the optimization decision in separate records.

Examples::

    # The lower bound: what is already linearly decodable from the parent.
    python -m experiments.retrofit_parent \\
        --checkpoint checkpoints/vr-noexits/final.pt \\
        --run-dir runs/retrofit-frozen-tied \\
        --mode frozen_tied_head --exit_every 2

    # The recommended lightweight baseline.
    python -m experiments.retrofit_parent \\
        --checkpoint checkpoints/vr-noexits/final.pt \\
        --run-dir runs/retrofit-adapter \\
        --mode frozen_exit_adapter --exit_adapter_rank 32 --exit_every 2

    # Validate everything without building anything expensive.
    python -m experiments.retrofit_parent \\
        --checkpoint checkpoints/vr-noexits/final.pt \\
        --run-dir runs/retrofit-lora --mode lora --dry-run
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.config import TransformerConfig
from src.retrofit import (
    RETROFIT_MODES,
    RetrofitConfig,
    RetrofitReport,
    assert_parent_preserved,
    load_parent,
    retrofit,
)
from training.train import CHECKPOINT_SCHEMA_VERSION
from utils.provenance import RunArtifacts, Seeds, file_digest

#: Tokens used to verify that the parent's full-depth output did not move.
#: Random ids are enough: preservation is a statement about the function, and a
#: function that agrees on random inputs at every position agrees.
PROBE_ROWS = 4
PROBE_LENGTH = 64


def build(
    checkpoint: str | Path,
    retrofit_config: RetrofitConfig,
    exit_every: int,
    min_exit_layer: int | None = None,
    seed: int = 1337,
    device: str = "cpu",
) -> tuple[torch.nn.Module, torch.nn.Module, TransformerConfig, RetrofitReport]:
    """Loads a parent and builds the retrofitted model.

    Args:
        checkpoint: Parent checkpoint path.
        retrofit_config: How to adapt it.
        exit_every: Place an exit every this many layers in the retrofitted
            model. The parent typically has one exit; this is where the extra
            endpoints come from.
        min_exit_layer: Earliest layer allowed to terminate a token. Defaults to
            the parent's.
        seed: Seeds the initialization of modules the parent has no counterpart
            for, and the verification probe.
        device: Device to build on.

    Returns:
        A tuple ``(model, parent, model_config, report)``.
    """
    parent, parent_config = load_parent(checkpoint, device=device)

    model_config = replace(
        parent_config,
        exit_every=exit_every,
        min_exit_layer=(
            parent_config.min_exit_layer if min_exit_layer is None else min_exit_layer
        ),
        exit_adapter_rank=(
            retrofit_config.exit_adapter_rank
            if retrofit_config.mode == "frozen_exit_adapter"
            else 0
        ),
        tie_embeddings=not retrofit_config.untie_exit_heads,
        preservation_weight=retrofit_config.preservation_weight,
        preservation_teacher_checkpoint=str(checkpoint),
    )

    torch.manual_seed(Seeds.derive(seed).model_init)
    model, report = retrofit(parent, retrofit_config, model_config=model_config)
    model.to(device)
    return model, parent, model_config, report


def verify(
    model: torch.nn.Module,
    parent: torch.nn.Module,
    retrofit_config: RetrofitConfig,
    seed: int,
) -> dict:
    """Checks the parent's full-depth output where the check is exact.

    Args:
        model: The retrofitted model.
        parent: The frozen reference.
        retrofit_config: The mode applied, which decides whether exactness is
            claimed at all.
        seed: Seed for the probe.

    Returns:
        A record of what was verified. When the mode does not preserve the parent
        exactly, ``verified`` is ``False`` and ``reason`` says why — a run must
        not be able to report an unverified preservation claim as a verified one.

    Raises:
        AssertionError: If an exact mode failed the check.
    """
    generator = torch.Generator().manual_seed(Seeds.derive(seed).benchmark)
    probe = torch.randint(
        0, model.config.vocab_size, (PROBE_ROWS, PROBE_LENGTH), generator=generator
    )
    probe = probe.to(next(model.parameters()).device)

    if not retrofit_config.preserves_parent_exactly:
        return {
            "verified": False,
            "reason": (
                f"mode {retrofit_config.mode!r} leaves backbone weights trainable, "
                f"so the parent's output is not preserved by construction. It must "
                f"be tested for non-inferiority on held-out data after training, "
                f"against a predeclared margin."
            ),
            "recoverable": retrofit_config.parent_is_recoverable,
        }

    difference = assert_parent_preserved(model, parent, probe, tolerance=0.0)
    return {
        "verified": True,
        "max_logit_difference": difference,
        "probe_rows": PROBE_ROWS,
        "probe_length": PROBE_LENGTH,
        "note": (
            "bit-identical at construction. The claim survives training only "
            "because no parameter feeding the full-depth path is trainable; see "
            "the trainable_names in this record."
        ),
    }


def save(
    path: Path,
    model: torch.nn.Module,
    model_config: TransformerConfig,
    retrofit_config: RetrofitConfig,
    parent_checkpoint: str | Path,
) -> Path:
    """Writes the retrofitted model in the shape ``training.train`` resumes from.

    Args:
        path: Destination file.
        model: The retrofitted model.
        model_config: Its architecture.
        retrofit_config: How it was produced.
        parent_checkpoint: The parent it branched from.

    Returns:
        The path written.

    Note:
        ``completed_updates`` is zero. This is a starting point, not a partially
        trained run, and a resume from it should begin the data stream at the
        top rather than skipping ahead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": model.state_dict(),
            "optimizer": {},
            "scaler": None,
            "step": 0,
            "completed_updates": 0,
            "completed_tokens": 0,
            "model_config": model_config,
            "train_config": None,
            "retrofit": asdict(retrofit_config),
            "parent_checkpoint": str(parent_checkpoint),
            "parent_sha256": file_digest(parent_checkpoint),
            "lineage": [
                {
                    "stage": "retrofit",
                    "mode": retrofit_config.mode,
                    "parent": str(parent_checkpoint),
                }
            ],
        },
        path,
    )
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the command line.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Retrofit shallow endpoints onto a trained parent."
    )
    parser.add_argument("--checkpoint", required=True, help="Parent checkpoint.")
    parser.add_argument("--run-dir", required=True, help="Where to write the record.")
    parser.add_argument("--mode", default="frozen_exit_adapter", choices=RETROFIT_MODES)
    parser.add_argument("--exit_every", type=int, default=2)
    parser.add_argument("--min_exit_layer", type=int, default=None)
    parser.add_argument("--exit_adapter_rank", type=int, default=32)
    parser.add_argument("--unfreeze_blocks", default="", help="Comma-separated indices.")
    parser.add_argument("--unfreeze_norms_only", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_targets",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated substrings matched against projection names.",
    )
    parser.add_argument(
        "--lora_layers", default="", help="Comma-separated block indices; empty for all."
    )
    parser.add_argument("--preservation_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the checkpoint, the mode, the architecture and the "
            "trainable set, then stop without writing a model."
        ),
    )
    return parser.parse_args(argv)


def _indices(value: str) -> tuple[int, ...]:
    """Parses a comma-separated index list.

    Args:
        value: Text such as ``"0,1,4"``, possibly empty.

    Returns:
        The parsed indices, empty for empty input.
    """
    return tuple(int(part) for part in value.split(",") if part.strip())


def main(argv: list[str] | None = None) -> int:
    """Runs the retrofit end to end.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit status.
    """
    args = parse_args(argv)

    retrofit_config = RetrofitConfig(
        mode=args.mode,
        exit_adapter_rank=args.exit_adapter_rank,
        unfreeze_blocks=_indices(args.unfreeze_blocks),
        unfreeze_norms_only=args.unfreeze_norms_only,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=tuple(
            part.strip() for part in args.lora_targets.split(",") if part.strip()
        ),
        lora_layers=_indices(args.lora_layers),
        preservation_weight=args.preservation_weight,
        parent_checkpoint=args.checkpoint,
    )

    model, parent, model_config, report = build(
        args.checkpoint,
        retrofit_config,
        exit_every=args.exit_every,
        min_exit_layer=args.min_exit_layer,
        seed=args.seed,
        device=args.device,
    )

    print(f"parent      {args.checkpoint}")
    print(f"exits       layers {model_config.exit_layers}")
    print(report.summary())

    verification = verify(model, parent, retrofit_config, args.seed)
    if verification["verified"]:
        print(
            f"preserved   max logit difference "
            f"{verification['max_logit_difference']:.3e} over "
            f"{PROBE_ROWS}x{PROBE_LENGTH} probe tokens"
        )
    else:
        print(f"preserved   not by construction: {verification['reason']}")

    if args.dry_run:
        print("dry run: validated, nothing written")
        return 0

    run_dir = Path(args.run_dir)
    artifacts = RunArtifacts.create(
        run_dir,
        script="experiments.retrofit_parent",
        config={
            "retrofit": asdict(retrofit_config),
            "model": asdict(model_config),
            "exit_every": args.exit_every,
        },
        seeds=Seeds.derive(args.seed),
        inputs={"parent_checkpoint": args.checkpoint},
        parent_checkpoint=args.checkpoint,
        required=(),
    )
    artifacts.log_metric(
        {
            "stage": "retrofit",
            "mode": retrofit_config.mode,
            "trainable": report.trainable,
            "frozen": report.frozen,
            "trainable_fraction": report.trainable_fraction,
            "trainable_groups": report.trainable_groups,
            "trainable_names": list(report.trainable_names),
            "lora_modules": list(report.lora_modules),
            "notes": list(report.notes),
            "preservation": verification,
        }
    )

    path = save(
        artifacts.checkpoints_dir / "retrofit.pt",
        model,
        model_config,
        retrofit_config,
        args.checkpoint,
    )
    print(f"wrote       {path}")
    print(f"record      {run_dir}")
    print()
    print("Train it with:")
    print(
        f"  python -m training.train --resume_from={path} "
        f"--objective_version=anchored_v1 --shallow_loss_weight=0.25 ..."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
