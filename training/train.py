"""Training loop for the decoder-only Transformer.

Single process::

    python -m training.train --max_steps=2000

Multiple GPUs on one node::

    torchrun --standalone --nproc_per_node=4 -m training.train --max_steps=2000

The same code runs both ways: a world of size one simply skips the collectives.
Tokenize the corpus first with ``python -m training.data``.

Architecture and schedule share one command line, so a run is fully specified in
one place::

    torchrun --standalone --nproc_per_node=4 -m training.train \\
        --n_layers=12 --d_model=768 --seq_len=1024 --batch_size=8 --max_steps=50000

Note:
    On machines whose loopback resolves to IPv6 first, ``--standalone`` can stall
    in rendezvous. Pin the launcher to IPv4 instead::

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

from src.config import TrainConfig, TransformerConfig, parse_configs
from src.model import Transformer
from src.tokenizer import config_from_tokenizer, load_tokenizer
from training import distributed
from training.data import build_dataloader
from utils.tracking import RunTracker

#: Maps the config's precision name onto a torch dtype.
DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

#: Version of the checkpoint layout. Bumped if the saved fields change.
CHECKPOINT_SCHEMA_VERSION = 1


def learning_rate_at(step: int, config: TrainConfig) -> float:
    """Computes the learning rate for a step.

    Linear warmup followed by cosine decay to ``min_lr``.

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


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy, and torch (including CUDA).

    Args:
        seed: Value applied to every generator.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def random_states() -> dict[str, object]:
    """Captures every random stream the training loop draws from.

    Omitting any one of these makes resume inexact in a way invisible in the
    loss curve but real in the results: dropout masks repeat, and any sampling
    decision restarts from the top of its stream.

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
            skipped.
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
        if torch.cuda.device_count() >= len(cuda):
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
    if isinstance(value, (list, tuple)):
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
) -> None:
    """Writes everything needed to resume the same run.

    Args:
        path: Destination file.
        model: The unwrapped model, never the DDP or compile wrapper.
        optimizer: Optimizer whose moments are saved alongside the weights.
        step: Number of *completed* optimizer updates. The data cursor is
            derived from this, so an off-by-one here repeats or skips a batch.
        model_config: Architecture the weights belong to.
        train_config: Run settings, stored so a resumed run is reproducible.
        scaler: Gradient scaler, whose scale factor is part of the optimization
            state under fp16.

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
            "model_config": model_config,
            "train_config": train_config,
            "random_states": random_states(),
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
) -> float:
    """Measures mean cross-entropy on held-out data.

    Args:
        model: The unwrapped model.
        loader: Validation loader.
        config: Run settings supplying ``eval_steps``.
        device: Device to move batches to.
        autocast: Context manager applying the configured precision.
        context: Distributed context. When present, loss sums and counts are
            reduced across every rank; without it, rank zero reports only its
            own validation shard.

    Returns:
        The mean loss across ranks, or NaN if no batches were seen.
    """
    was_training = model.training
    model.eval()
    total = 0.0
    batches = 0
    for inputs, targets in loader:
        if batches >= config.eval_steps:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        with autocast:
            out = model(inputs, targets=targets)
        total += float(out.loss)
        batches += 1
    model.train(was_training)

    # float32, not float64: the sum is a handful of loss scalars where the
    # precision difference is immaterial, and MPS has no float64 dtype.
    aggregate = torch.tensor(
        [total, float(batches)], dtype=torch.float32, device=device
    )
    if context is not None:
        aggregate = distributed.reduce_mean(aggregate, context)
    if float(aggregate[1]) == 0:
        return float("nan")
    return float(aggregate[0] / aggregate[1])


def main(
    train_config: TrainConfig,
    model_overrides: dict[str, object] | None = None,
) -> None:
    """Runs training end to end.

    Args:
        train_config: Run settings, normally from
            :func:`src.config.parse_configs`.
        model_overrides: Architecture fields to override, excluding
            ``vocab_size`` and ``max_seq_len`` which are derived from the
            tokenizer and ``seq_len``.
    """
    context = distributed.setup(train_config.ddp_backend)
    device = context.device
    seeds = train_config.seeds()

    # Initialization is seeded without a rank offset so a single-process run and
    # rank zero of a distributed one start from identical weights.
    seed_everything(seeds.model_init)

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
        print(f"world:  {context.world_size} process(es) on {device}")
        print(f"tokens: {tokens_per_step:,} per optimizer step")
        if start_step:
            print(f"resume: from completed update {start_step:,}")
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
    if checkpoint is not None and "random_states" in checkpoint:
        restore_random_states(checkpoint["random_states"])

    batches = iter(train_loader)
    started = time.time()
    model.train()

    for step in range(start_step, train_config.max_steps):
        lr = learning_rate_at(step, train_config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        running = torch.zeros((), device=device)
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

        if train_config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), train_config.grad_clip
            )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % train_config.log_every == 0:
            # Every rank must reach this reduction: all_reduce is a collective,
            # so guarding it with is_main would leave rank zero waiting on peers
            # that never call it. Only the printing is rank-local.
            mean_loss = float(distributed.reduce_mean(running, context))
            if context.is_main:
                elapsed = time.time() - started
                throughput = (
                    (step + 1 - start_step) * tokens_per_step / max(elapsed, 1e-6)
                )
                eta_hours = (
                    (train_config.max_steps - step - 1)
                    * tokens_per_step
                    / max(throughput, 1e-6)
                    / 3600
                )
                tracker.log(
                    {
                        "train/loss": mean_loss,
                        "train/lr": lr,
                        "train/tokens": (step + 1) * tokens_per_step,
                        "train/tokens_per_sec": throughput,
                    },
                    step=step,
                )
                # Flushed explicitly: stdout is block-buffered when redirected to
                # a file, and an unflushed log is indistinguishable from a hung
                # job.
                print(
                    f"step {step:>6}  loss {mean_loss:.4f}  lr {lr:.2e}  "
                    f"{throughput:,.0f} tok/s  eta {eta_hours:.1f}h",
                    flush=True,
                )

        # Periodic work is keyed on steps *completed*, so --save_every=30 writes
        # after the 30th step rather than before the 31st.
        completed = step + 1

        if train_config.eval_every and completed % train_config.eval_every == 0:
            val_loss = evaluate(
                raw_model, val_loader, train_config, device, autocast, context
            )
            if context.is_main:
                print(f"  eval  loss {val_loss:.4f}", flush=True)
                tracker.log({"eval/loss": val_loss}, step=step)

        if train_config.save_every and completed % train_config.save_every == 0:
            if context.is_main:
                path = Path(train_config.out_dir) / f"step-{completed:06d}.pt"
                save_checkpoint(
                    path, raw_model, optimizer, completed,
                    model_config, train_config, scaler=scaler,
                )
                print(f"  saved {path}", flush=True)
            distributed.barrier(context)

    if context.is_main:
        path = Path(train_config.out_dir) / "final.pt"
        save_checkpoint(
            path, raw_model, optimizer, train_config.max_steps,
            model_config, train_config, scaler=scaler,
        )
        print(f"done, saved {path}", flush=True)

    tracker.finish()
    distributed.cleanup(context)


if __name__ == "__main__":
    main(*parse_configs())
