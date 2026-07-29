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

import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from training import distributed
from utils.calibration import format_sweep, recommend_threshold, sweep_thresholds
from src.config import TrainConfig, TransformerConfig, parse_configs
from training.data import build_dataloader
from src.model import Transformer
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


def save_checkpoint(
    path: Path,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_config: TransformerConfig,
    train_config: TrainConfig,
) -> None:
    """Writes everything needed to resume.

    Args:
        path: Destination file.
        model: The unwrapped model, never the DDP wrapper.
        optimizer: Optimizer whose moments are saved alongside the weights.
        step: Number of completed optimizer steps.
        model_config: Architecture the weights belong to.
        train_config: Run settings, stored so a resumed run is reproducible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "model_config": model_config,
            "train_config": train_config,
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

    # Ranks share a model init but must not walk the data in lockstep.
    torch.manual_seed(train_config.seed + context.rank)

    tokenizer = load_tokenizer(train_config.tokenizer_name)
    model_config = config_from_tokenizer(
        tokenizer, max_seq_len=train_config.seq_len, **(model_overrides or {})
    )

    start_step = 0
    checkpoint = None
    if train_config.resume_from is not None:
        checkpoint = torch.load(
            train_config.resume_from, map_location="cpu", weights_only=False
        )
        model_config = checkpoint["model_config"]
        start_step = checkpoint["step"]

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

    if context.is_main:
        print(f"model:  {model.num_parameters() / 1e6:.1f}M parameters")
        print(f"exits:  layers {model_config.exit_layers}")
        print(f"world:  {context.world_size} process(es) on {device}")
        print(f"tokens: {tokens_per_step:,} per optimizer step")
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
    train_loader = build_dataloader(
        data_dir / "train.bin", train_config, context.world_size, context.rank
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

    batches = _infinite(train_loader)
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
            for i, layer in enumerate(exit_layers):
                if layer in out.exit_losses:
                    exit_totals[i] += out.exit_losses[layer]
                    exit_counts[i] += 1

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

                tracker.log(
                    {
                        "train/loss": mean_loss,
                        "train/lr": lr,
                        "train/tokens": (step + 1) * tokens_per_step,
                        "train/tokens_per_sec": throughput,
                        **(
                            {"train/kv_reconstruction": kv_mean}
                            if model_config.learned_kv_propagation
                            else {}
                        ),
                    },
                    step=step,
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

        if train_config.sweep_every and completed % train_config.sweep_every == 0:
            _report_sweep(
                raw_model, val_loader, model_config, device, context, tracker, step
            )

        if train_config.save_every and completed % train_config.save_every == 0:
            if context.is_main:
                path = Path(train_config.out_dir) / f"step-{completed:06d}.pt"
                save_checkpoint(
                    path, raw_model, optimizer, completed, model_config, train_config
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
        )
        print(f"done, saved {path}", flush=True)

    tracker.finish()
    distributed.cleanup(context)


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


def _infinite(loader: torch.utils.data.DataLoader):
    """Cycles a loader forever, reshuffling each epoch.

    Args:
        loader: Loader to repeat.

    Yields:
        Batches, restarting when the loader is exhausted. The distributed
        sampler is re-seeded per epoch so ranks do not repeat the same shard
        order.
    """
    epoch = 0
    while True:
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


if __name__ == "__main__":
    main(*parse_configs())
