"""Training loop for the early-exit Transformer.

Single process::

    python -m training.train --max_steps=2000

Multiple GPUs on one node::

    torchrun --standalone --nproc_per_node=4 -m training.train --max_steps=2000

The same code runs both ways: a world of size one simply skips the collectives.
Tokenize the corpus first with ``python -m training.data``.

Architecture and schedule share one command line, so a run is fully specified
in one place::

    torchrun --standalone --nproc_per_node=4 -m training.train \\
        --n_layers=12 --d_model=768 --exit_every=2 --min_exit_layer=2 \\
        --seq_len=1024 --batch_size=8 --max_steps=50000

Note:
    On machines whose loopback resolves to IPv6 first, ``--standalone`` can
    stall in rendezvous with repeated ``ip6.arpa`` resolution warnings. Pin the
    launcher to IPv4 instead::

        torchrun --nnodes=1 --nproc_per_node=2 \\
            --master_addr=127.0.0.1 --master_port=29500 -m training.train ...
"""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from training import distributed
from utils.calibration import format_sweep, recommend_threshold, sweep_thresholds
from src.config import TrainConfig, TransformerConfig, parse_configs
from training.data import build_dataloader
from src.model import Transformer
from utils.provenance import RunArtifacts
from utils.tracking import RunTracker
from src.tokenizer import config_from_tokenizer, load_tokenizer

#: Maps the config's precision name onto a torch dtype.
DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def learning_rate_at(step: int, config: TrainConfig) -> float:
    """Computes the learning rate for a step.

    Linear warmup followed by cosine decay. The warmup matters more than usual
    here: the exit modules all write into one shared output matrix, so early
    gradients from a dozen exits arrive at the same weights at once.

    Args:
        step: Zero-based optimizer step.
        config: Run settings supplying the schedule.

    Returns:
        The learning rate to use on this step.
    """
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(config.warmup_steps, 1)

    span = max(config.max_steps - config.warmup_steps, 1)
    progress = min((step - config.warmup_steps) / span, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + (config.learning_rate - config.min_lr) * cosine


def build_optimizer(
    model: torch.nn.Module,
    config: TrainConfig,
    device: torch.device,
) -> torch.optim.AdamW:
    """Creates AdamW with decay applied only where it belongs.

    Weight decay is a prior pulling weights toward zero, which is sensible for
    the matrices that mix features and actively harmful for the one-dimensional
    gain parameters in RMSNorm, whose useful values sit near one.

    Args:
        model: Model whose parameters are optimized.
        config: Run settings supplying the hyperparameters.
        device: Device training runs on, used to enable the fused kernel.

    Returns:
        The configured optimizer.
    """
    decayed, plain = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decayed if parameter.dim() >= 2 else plain).append(parameter)

    groups = [
        {"params": decayed, "weight_decay": config.weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        fused=device.type == "cuda",
    )


#: Version of the checkpoint layout. Version 1 held weights, optimizer state
#: and a step count, which is enough to continue training but not enough to
#: continue the *same* run: the data cursor, the exit rotation and every random
#: stream restarted. Version 2 adds them.
CHECKPOINT_SCHEMA_VERSION = 2


def seed_everything(seed: int) -> None:
    """Seeds every stream the process can draw from.

    Seeding only ``torch`` leaves Python's and NumPy's generators at whatever
    system entropy gave them, so two runs of the same configuration differ in
    any code path that reaches for either — and a resumed run cannot restore a
    position in a stream whose origin was never fixed.

    Args:
        seed: Value applied to Python, NumPy, and torch, including CUDA.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def random_states() -> dict[str, object]:
    """Captures every random stream the training loop draws from.

    Omitting any one of these makes resume inexact in a way that is invisible
    in the loss curve but real in the results: dropout masks repeat, and any
    future sampling decision restarts from the top of its stream.

    Returns:
        A mapping of stream name to serializable state. CUDA states are a list
        with one entry per visible device, empty off CUDA.
    """
    import random

    import numpy as np

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def restore_random_states(states: dict[str, object]) -> None:
    """Restores the streams captured by :func:`random_states`.

    Args:
        states: A mapping produced by :func:`random_states`. Missing keys are
            skipped, so a version 1 checkpoint still loads — it simply cannot
            promise exactness, which the caller reports.

    Note:
        CUDA states are restored only when the current process has at least as
        many devices as the checkpoint recorded. Restoring four devices' states
        onto two would silently drop two streams, and a resume that quietly
        differs is worse than one that says it does.
    """
    import random

    import numpy as np

    if "python" in states:
        random.setstate(_as_tuple(states["python"]))
    if "numpy" in states:
        np.random.set_state(_as_tuple(states["numpy"]))
    if "torch_cpu" in states:
        torch.set_rng_state(states["torch_cpu"].cpu())
    cuda = states.get("torch_cuda") or []
    if cuda and torch.cuda.is_available():
        if torch.cuda.device_count() < len(cuda):
            print(
                f"  warning: checkpoint holds {len(cuda)} CUDA RNG states but "
                f"this process sees {torch.cuda.device_count()} devices; CUDA "
                f"streams not restored, so resume is not exact.",
                flush=True,
            )
        else:
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda])


def _as_tuple(value):
    """Recursively converts lists back into tuples.

    ``random.setstate`` and ``numpy.random.set_state`` require tuples, and a
    round trip through some serializers turns them into lists.

    Args:
        value: A possibly nested structure.

    Returns:
        The same structure with every list replaced by a tuple.
    """
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_as_tuple(item) for item in value)
    return value


def save_checkpoint(
    path: Path,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_config: TransformerConfig,
    train_config: TrainConfig,
    scaler: torch.amp.GradScaler | None = None,
    tokens: int | None = None,
    lineage: list[dict] | None = None,
) -> None:
    """Writes everything needed to resume the same run, not merely a run.

    Args:
        path: Destination file.
        model: The unwrapped model, never the DDP wrapper.
        optimizer: Optimizer whose moments are saved alongside the weights.
        step: Number of *completed* optimizer updates. The data cursor is
            derived from this, so an off-by-one here repeats or skips a batch.
        model_config: Architecture the weights belong to.
        train_config: Run settings, stored so a resumed run is reproducible.
        scaler: Gradient scaler, whose scale factor is part of the optimization
            state under fp16.
        tokens: Tokens consumed so far, recorded rather than recomputed so that
            a run whose batch size changed mid-flight still reports a true
            budget.
        lineage: One entry per launch that contributed to this checkpoint.

    Note:
        The data cursor is deliberately absent. :class:`StatelessBlockSampler`
        derives its position from ``step``, so there is no cursor to save and no
        way for one to drift out of agreement with the update count.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "completed_updates": step,
            "completed_tokens": tokens,
            "model_config": model_config,
            "train_config": train_config,
            "seeds": asdict(train_config.seeds()),
            # Not in state_dict: registered with persistent=False, because it
            # is a schedule position rather than a parameter. It still decides
            # which exits a step scores, so a resume that drops it trains a
            # different rotation than the run it claims to continue.
            "step_counter": int(model._step_counter.item()),
            "random_states": random_states(),
            "lineage": list(lineage or []),
        },
        path,
    )


@torch.no_grad()
def evaluate(
    model: Transformer,
    loader: torch.utils.data.DataLoader,
    config: TrainConfig,
    device: torch.device,
    autocast,
    context: distributed.DistributedContext | None = None,
) -> tuple[float, dict[int, float]]:
    """Measures loss on held-out data.

    Args:
        model: The unwrapped model.
        loader: Validation loader.
        config: Run settings supplying ``eval_steps``.
        device: Device to move batches to.
        autocast: Context manager applying the configured precision.
        context: Distributed context. When present, loss sums and counts are
            reduced across every rank before reporting; without this, rank zero
            would log only its validation shard.

    Returns:
        A tuple ``(mean_loss, mean_exit_losses)`` where the second element maps
        each exit's layer index to its own cross-entropy. Every exit appears,
        not the rotating subset training scores.
    """
    was_training = model.training
    model.eval()
    total = 0.0
    per_exit_total: dict[int, float] = {}
    per_exit_count: dict[int, int] = {}
    batches = 0

    # Every exit, not the step's rotation. The rotation is deterministic in the
    # global step, so it aliases against any schedule keyed on the step: at
    # eval_every=500 with five non-final exits and a budget of two, every
    # evaluation landed on the same rotation position and scored the same two
    # exits for the whole run. There is no memory reason to sample here -- this
    # function is under no_grad, so no log-softmax is retained for a backward
    # pass that never happens.
    with model.score_all_exits():
        for inputs, targets in loader:
            if batches >= config.eval_steps:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast:
                out = model(inputs, targets=targets)
            total += float(out.loss)
            for layer, value in out.exit_losses.items():
                per_exit_total[layer] = per_exit_total.get(layer, 0.0) + value
                per_exit_count[layer] = per_exit_count.get(layer, 0) + 1
            batches += 1

    model.train(was_training)

    # Every rank receives a disjoint validation shard. Reduce numerators and
    # denominators separately so rank zero reports the whole held-out sample,
    # not whichever shard happened to be assigned rank zero. ``reduce_mean`` is
    # sufficient because the common world-size factor cancels in the ratio.
    aggregate = torch.tensor(
        [total, float(batches)], dtype=torch.float64, device=device
    )
    if context is not None:
        aggregate = distributed.reduce_mean(aggregate, context)

    global_batches = float(aggregate[1])
    if global_batches == 0:
        return float("nan"), {}

    layers = model.config.exit_layers
    exit_sums = torch.tensor(
        [per_exit_total.get(layer, 0.0) for layer in layers],
        dtype=torch.float64,
        device=device,
    )
    exit_counts = torch.tensor(
        [per_exit_count.get(layer, 0) for layer in layers],
        dtype=torch.float64,
        device=device,
    )
    if context is not None:
        exit_sums = distributed.reduce_mean(exit_sums, context)
        exit_counts = distributed.reduce_mean(exit_counts, context)

    means = {
        layer: float(exit_sums[index] / exit_counts[index])
        for index, layer in enumerate(layers)
        if float(exit_counts[index]) > 0.0
    }
    return float(aggregate[0] / aggregate[1]), means


def main(
    train_config: TrainConfig,
    model_overrides: dict[str, object] | None = None,
) -> None:
    """Runs training end to end.

    Args:
        train_config: Run settings, normally from
            :func:`config.parse_configs`.
        model_overrides: Architecture fields to override, excluding
            ``vocab_size`` and ``max_seq_len`` which are derived from the
            tokenizer and ``seq_len``.
    """
    context = distributed.setup(train_config.ddp_backend)
    device = context.device
    seeds = train_config.seeds()

    # Initialization is seeded without a rank offset, so that two arms of a
    # causal comparison can branch from the same parameters. DDP would
    # broadcast rank zero's weights anyway, but a rank offset would also make a
    # single-process run differ from rank zero of a distributed one.
    seed_everything(seeds.model_init)

    tokenizer = load_tokenizer(train_config.tokenizer_name)
    model_config = config_from_tokenizer(
        tokenizer, max_seq_len=train_config.seq_len, **(model_overrides or {})
    )

    start_step = 0
    checkpoint = None
    lineage: list[dict] = []
    if train_config.resume_from is not None:
        checkpoint = torch.load(
            train_config.resume_from, map_location="cpu", weights_only=False
        )
        model_config = checkpoint["model_config"]
        start_step = checkpoint.get("completed_updates", checkpoint["step"])
        lineage = list(checkpoint.get("lineage", []))

    model = Transformer(model_config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])

    tokens_per_step = (
        train_config.batch_size
        * train_config.grad_accum_steps
        * train_config.seq_len
        * context.world_size
    )
    tracker = RunTracker(train_config, model_config, context.is_main)

    # The local record, written whether or not a tracking service was
    # reachable. Rank zero owns it; other ranks would interleave writes into the
    # same files for no added information.
    artifacts = None
    if context.is_main:
        artifacts = RunArtifacts.create(
            Path(train_config.out_dir) / "run",
            script="training.train",
            config={
                "train": asdict(train_config),
                "model": asdict(model_config),
                "world_size": context.world_size,
                "tokens_per_step": tokens_per_step,
            },
            seeds=seeds,
            inputs={
                "train_bin": str(Path(train_config.data_dir) / "train.bin"),
                "val_bin": str(Path(train_config.data_dir) / "val.bin"),
                "tokenizer_name": train_config.tokenizer_name,
            },
            parent_checkpoint=train_config.resume_from,
            required=(),
        )
        artifacts.record_resume(
            {
                "start_update": start_step,
                "max_steps": train_config.max_steps,
                "resumed_from": train_config.resume_from,
                "world_size": context.world_size,
            }
        )
        lineage.append(
            {
                "start_update": start_step,
                "resumed_from": train_config.resume_from,
                "world_size": context.world_size,
                "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )

    if context.is_main:
        print(f"model:  {model.num_parameters() / 1e6:.1f}M parameters")
        print(f"exits:  layers {model_config.exit_layers}")
        print(f"world:  {context.world_size} process(es) on {device}")
        print(f"tokens: {tokens_per_step:,} per optimizer step")
        print(f"seeds:  {seeds}")
        if start_step:
            print(f"resume: from completed update {start_step:,}")
        if artifacts is not None:
            print(f"record: {artifacts.run_dir}")
        if tracker.enabled:
            print(f"wandb:  {train_config.wandb_project} ({train_config.wandb_mode})")

    optimizer = build_optimizer(model, train_config, device)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    raw_model = model
    if train_config.compile_model:
        model = torch.compile(model)
    if context.enabled:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if device.type == "cuda" else None,
            find_unused_parameters=train_config.find_unused_parameters,
        )

    data_dir = Path(train_config.data_dir)
    # The cursor is derived from the completed update count rather than restored
    # from the loader, which is what makes resume land on the next unseen batch
    # instead of the top of the corpus.
    train_loader = build_dataloader(
        data_dir / "train.bin",
        train_config,
        context.world_size,
        context.rank,
        start_micro_batch=start_step * train_config.grad_accum_steps,
    )
    val_loader = build_dataloader(
        data_dir / "val.bin",
        train_config,
        context.world_size,
        context.rank,
        shuffle=False,
    )

    dtype = DTYPES[train_config.dtype]
    autocast = (
        nullcontext()
        if dtype is torch.float32 or device.type == "cpu"
        else torch.autocast(device_type=device.type, dtype=dtype)
    )
    # Only fp16 needs loss scaling; bf16 has the same exponent range as fp32.
    scaler = torch.amp.GradScaler(
        device.type, enabled=dtype is torch.float16 and device.type == "cuda"
    )
    if checkpoint is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    # Ranks must not apply identical dropout masks, so this stream — unlike
    # initialization — is offset by rank on purpose.
    seed_everything(seeds.dropout + context.rank)
    if checkpoint is not None:
        if "random_states" in checkpoint:
            restore_random_states(checkpoint["random_states"])
            # raw_model, not model: under DDP or torch.compile the wrapper has
            # no such buffer, and setting an attribute on the wrapper would
            # leave the real rotation counter at zero.
            raw_model._step_counter.fill_(int(checkpoint.get("step_counter", 0)))
        elif context.is_main:
            print(
                f"  warning: {train_config.resume_from} predates checkpoint "
                f"schema {CHECKPOINT_SCHEMA_VERSION} and carries no random or "
                f"rotation state. Training continues; it is not the same run.",
                flush=True,
            )

    batches = iter(train_loader)
    exit_layers = model_config.exit_layers
    started = time.time()
    model.train()

    for step in range(start_step, train_config.max_steps):
        lr = learning_rate_at(step, train_config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        running = torch.zeros((), device=device)
        exit_totals = torch.zeros(len(exit_layers), device=device)
        exit_counts = torch.zeros(len(exit_layers), device=device)
        kv_running = torch.zeros((), device=device)
        # Tracked apart from the combined objective. The combined value is not a
        # cross-entropy and must not be reported as one, and the no-regret
        # question is about the full endpoint's own CE specifically.
        full_running = torch.zeros((), device=device)
        distill_totals = torch.zeros(len(exit_layers), device=device)
        distill_counts = torch.zeros(len(exit_layers), device=device)
        preservation_running = torch.zeros((), device=device)
        step_alpha, step_full_weight = 0.0, 0.0
        has_preservation = False

        for micro in range(train_config.grad_accum_steps):
            inputs, targets = next(batches)
            inputs, targets = inputs.to(device), targets.to(device)

            # Gradients only need reducing on the last micro-batch; syncing on
            # every one would multiply communication by grad_accum_steps.
            is_last = micro == train_config.grad_accum_steps - 1
            sync = nullcontext() if is_last or not context.enabled else model.no_sync()

            with sync, autocast:
                out = model(inputs, targets=targets)
                loss = out.loss / train_config.grad_accum_steps

            scaler.scale(loss).backward()

            running += loss.detach()
            if out.kv_loss is not None:
                kv_running += out.kv_loss / train_config.grad_accum_steps
            if out.full_loss is not None:
                full_running += out.full_loss / train_config.grad_accum_steps
            if out.preservation_loss is not None:
                preservation_running += (
                    out.preservation_loss / train_config.grad_accum_steps
                )
                has_preservation = True
            if out.shallow_alpha is not None:
                step_alpha = out.shallow_alpha
            if out.full_weight is not None:
                step_full_weight = out.full_weight
            for i, layer in enumerate(exit_layers):
                if layer in out.exit_losses:
                    exit_totals[i] += out.exit_losses[layer]
                    exit_counts[i] += 1
                if layer in out.distill_losses:
                    distill_totals[i] += out.distill_losses[layer]
                    distill_counts[i] += 1

        if train_config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), train_config.grad_clip
            )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % train_config.log_every == 0:
            # Every rank must reach these reductions: all_reduce is a
            # collective, so guarding it with `is_main` would leave rank zero
            # waiting on peers that never call it. Only the printing is
            # rank-local.
            mean_loss = float(distributed.reduce_mean(running, context))
            totals = distributed.reduce_mean(exit_totals, context)
            counts = distributed.reduce_mean(exit_counts, context)
            kv_mean = float(distributed.reduce_mean(kv_running, context))
            full_mean = float(distributed.reduce_mean(full_running, context))
            distill_sums = distributed.reduce_mean(distill_totals, context)
            distill_seen = distributed.reduce_mean(distill_counts, context)
            preservation_mean = float(
                distributed.reduce_mean(preservation_running, context)
            )

            if context.is_main:
                per_exit = " ".join(
                    f"L{layer}:{float(totals[i] / counts[i]):.3f}"
                    for i, layer in enumerate(exit_layers)
                    if counts[i] > 0
                )
                elapsed = time.time() - started
                # Throughput belongs in the log, not only in the tracker. It is
                # the number that decides whether a run fits its wall clock, and
                # needing a browser open to read it is a poor trade when the
                # decision is usually made in the first minute.
                throughput = (
                    (step + 1 - start_step) * tokens_per_step / max(elapsed, 1e-6)
                )

                metrics = {
                    # Deliberately not called "ce": under either objective this
                    # is a weighted sum of several terms.
                    "train/objective": mean_loss,
                    "train/loss": mean_loss,
                    "train/full_ce": full_mean,
                    "train/full_weight": step_full_weight,
                    "train/shallow_alpha": step_alpha,
                    "train/grad_clip": train_config.grad_clip,
                    "train/lr": lr,
                    "train/tokens": (step + 1) * tokens_per_step,
                    "train/tokens_per_sec": throughput,
                    **(
                        {"train/preservation_kl": preservation_mean}
                        if has_preservation
                        else {}
                    ),
                    **(
                        {"train/kv_reconstruction": kv_mean}
                        if model_config.learned_kv_propagation
                        else {}
                    ),
                }
                tracker.log(metrics, step=step)
                if artifacts is not None:
                    # Both indices, because the earlier runs reported one and
                    # left readers to guess: the zero-based index a dashboard
                    # shows is one less than the number of updates completed,
                    # and the token budget follows the latter.
                    artifacts.log_metric(
                        {
                            "global_step_index": step,
                            "completed_updates": step + 1,
                            "objective_version": model_config.objective_version,
                            "scored_exits": [
                                layer
                                for i, layer in enumerate(exit_layers)
                                if counts[i] > 0
                            ],
                            **metrics,
                            **{
                                f"train/exit_ce_L{layer}": float(totals[i] / counts[i])
                                for i, layer in enumerate(exit_layers)
                                if counts[i] > 0
                            },
                            **{
                                f"train/distill_L{layer}": float(
                                    distill_sums[i] / distill_seen[i]
                                )
                                for i, layer in enumerate(exit_layers)
                                if distill_seen[i] > 0
                            },
                        }
                    )
                tracker.log_exit_losses(
                    {
                        layer: float(totals[i] / counts[i])
                        for i, layer in enumerate(exit_layers)
                        if counts[i] > 0
                    },
                    step=step,
                )
                # Flushed explicitly: stdout is block-buffered when redirected
                # to a file, which is how long runs are usually launched, and
                # an unflushed log is indistinguishable from a hung job.
                kv_note = (
                    f"  kv {kv_mean:.4f}"
                    if model_config.learned_kv_propagation
                    else ""
                )
                remaining = (train_config.max_steps - step - 1) * tokens_per_step
                eta_hours = remaining / max(throughput, 1e-6) / 3600

                print(
                    f"step {step:>6}  loss {mean_loss:.4f}  lr {lr:.2e}"
                    f"{kv_note}  {throughput:,.0f} tok/s  eta {eta_hours:.1f}h"
                    f"\n          exits {per_exit}",
                    flush=True,
                )

        # Periodic work is keyed on steps *completed*, so that --save_every=30
        # writes after the 30th step rather than before the 31st.
        completed = step + 1

        if train_config.eval_every and completed % train_config.eval_every == 0:
            val_loss, val_exits = evaluate(
                raw_model,
                val_loader,
                train_config,
                device,
                autocast,
                context,
            )
            if context.is_main:
                summary = " ".join(
                    f"L{layer}:{value:.3f}" for layer, value in sorted(val_exits.items())
                )
                print(f"  eval  loss {val_loss:.4f}  exits {summary}", flush=True)
                tracker.log({"eval/loss": val_loss}, step=step)
                tracker.log_exit_losses(val_exits, step=step, prefix="eval")
                if artifacts is not None:
                    artifacts.log_metric(
                        {
                            "global_step_index": step,
                            "completed_updates": completed,
                            "eval/loss": val_loss,
                            **{
                                f"eval/exit_ce_L{layer}": value
                                for layer, value in sorted(val_exits.items())
                            },
                        }
                    )

        if (
            train_config.grad_diagnostics_every
            and completed % train_config.grad_diagnostics_every == 0
            and len(exit_layers) > 1
        ):
            _report_gradient_conflict(
                raw_model,
                val_loader,
                device,
                context,
                tracker,
                artifacts,
                step,
                completed,
            )

        if train_config.sweep_every and completed % train_config.sweep_every == 0:
            _report_sweep(
                raw_model, val_loader, model_config, device, context, tracker, step
            )

        if train_config.save_every and completed % train_config.save_every == 0:
            if context.is_main:
                path = Path(train_config.out_dir) / f"step-{completed:06d}.pt"
                save_checkpoint(
                    path,
                    raw_model,
                    optimizer,
                    completed,
                    model_config,
                    train_config,
                    scaler=scaler,
                    tokens=completed * tokens_per_step,
                    lineage=lineage,
                )
                print(f"  saved {path}", flush=True)
            distributed.barrier(context)

    if context.is_main:
        path = Path(train_config.out_dir) / "final.pt"
        save_checkpoint(
            path,
            raw_model,
            optimizer,
            train_config.max_steps,
            model_config,
            train_config,
            scaler=scaler,
            tokens=train_config.max_steps * tokens_per_step,
            lineage=lineage,
        )
        print(f"done, saved {path}", flush=True)
        if artifacts is not None:
            artifacts.record_resume(
                {
                    "finished": True,
                    "completed_updates": train_config.max_steps,
                    "completed_tokens": train_config.max_steps * tokens_per_step,
                    "final_checkpoint": str(path),
                }
            )

    tracker.finish()
    distributed.cleanup(context)


def _report_gradient_conflict(
    model: Transformer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    context: distributed.DistributedContext,
    tracker: RunTracker | None,
    artifacts: RunArtifacts | None,
    step: int,
    completed: int,
) -> None:
    """Measures and records how the shallow objective opposes the full one.

    This is the evidence that decides whether gradient-surgery machinery is
    warranted. Adding PCGrad or a compromise gradient before knowing the cosine
    is adding complexity on a guess, and the simplest method that clears the
    no-regret gate should win.

    Args:
        model: The unwrapped model.
        loader: Loader to draw one held-out batch from.
        device: Device to move the batch to.
        context: Distribution context; only the main process measures, since the
            diagnostic is descriptive and one rank's batch answers the question.
        tracker: Optional experiment tracker.
        artifacts: Optional run directory, which receives the full per-group
            record. The scalars go to the tracker; the layerwise table is too
            wide for a dashboard and belongs in the raw records.
        step: Zero-based step index, for the tracker's x-axis.
        completed: Number of completed updates, for the record.
    """
    if not context.is_main:
        return

    inputs, targets = next(iter(loader))
    diagnostics = model.gradient_diagnostics(inputs.to(device), targets.to(device))

    print(
        f"  grad  cosine {diagnostics.cosine:+.4f}  "
        f"shallow/full norm {diagnostics.norm_ratio:.3f}  "
        f"groups in conflict {diagnostics.negative_fraction:.0%}",
        flush=True,
    )
    scalars = {
        "grad/cosine": diagnostics.cosine,
        "grad/norm_ratio": diagnostics.norm_ratio,
        "grad/full_norm": diagnostics.full_norm,
        "grad/shallow_norm": diagnostics.shallow_norm,
        "grad/negative_fraction": diagnostics.negative_fraction,
    }
    if tracker is not None:
        tracker.log(scalars, step=step)
    if artifacts is not None:
        path = artifacts.raw_records_dir / "gradient_diagnostics.jsonl"
        with path.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "global_step_index": step,
                        "completed_updates": completed,
                        **scalars,
                        "layer_cosine": diagnostics.layer_cosine,
                        "layer_norm_ratio": diagnostics.layer_norm_ratio,
                    }
                )
                + "\n"
            )


def _report_sweep(
    model: Transformer,
    loader: torch.utils.data.DataLoader,
    model_config: TransformerConfig,
    device: torch.device,
    context: distributed.DistributedContext,
    tracker: RunTracker | None = None,
    step: int = 0,
) -> None:
    """Prints the accuracy-versus-depth tradeoff on one validation batch.

    Run periodically so the exits' usefulness can be watched as it develops,
    rather than discovered at the end.

    Args:
        model: The unwrapped model.
        loader: Validation loader to draw a batch from.
        model_config: Architecture, for ``min_exit_layer``.
        device: Device to move the batch to.
        context: Distribution context; only the main process prints.
        tracker: Optional experiment tracker to mirror the curve to.
        step: Optimizer step the sweep describes.
    """
    if not context.is_main:
        return

    inputs, targets = next(iter(loader))
    model.eval()
    stats = model.exit_statistics(inputs.to(device), targets.to(device))
    model.train()

    points = sweep_thresholds(
        stats, [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0], model_config.min_exit_layer
    )
    print(format_sweep(points, stats), flush=True)
    pick = recommend_threshold(points, max_accuracy_drop=0.01)
    print(
        f"  suggested exit_threshold {pick.threshold:.2f} "
        f"-> mean depth {pick.mean_exit_depth:.2f}/{model_config.n_layers}, "
        f"{pick.compute_saved:.1%} of blocks skipped",
        flush=True,
    )

    if tracker is not None:
        tracker.log_sweep(points, step=step)
        tracker.log(
            {
                "sweep/best_threshold": pick.threshold,
                "sweep/compute_saved": pick.compute_saved,
                "sweep/accuracy": pick.accuracy,
                "sweep/mean_exit_depth": pick.mean_exit_depth,
            },
            step=step,
        )


if __name__ == "__main__":
    main(*parse_configs())
