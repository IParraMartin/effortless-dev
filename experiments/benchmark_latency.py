"""Measuring whether saved arithmetic becomes saved time.

It usually does not, and this file exists to say so with numbers rather than to
hope nobody asks. A depth-capped request performs strictly less work, but the
work it skips has to be *on the critical path* for anyone to notice, and three
things routinely stop that happening:

* **Batching.** A batch runs a layer for whoever still needs it, so one deep
  request in a batch of shallow ones pays for the whole batch. Grouping by
  depth fixes this in principle and costs scheduling latency in practice.
* **Kernel efficiency.** Half as many blocks over the same tensors is not half
  the time when each launch has fixed overhead, which is exactly the regime a
  small model on a laptop sits in.
* **The parts routing does not remove.** The vocabulary head, the probe, the
  controller, and the sampling loop are all paid regardless.

Everything reported here is a *measurement*. The analytical multiply-accumulate
counts from :mod:`utils.costs` appear alongside, in their own columns, and the
gap between them is reported as the systems realization gap rather than
averaged away.

Run it::

    python -m experiments.benchmark_latency --out results/latency
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from src.config import RoutingConfig, TransformerConfig
from src.model import Transformer
from experiments.workloads import DEMO_MODEL
from utils.costs import AnalyticalCostModel
from utils.provenance import RunRecord
from utils.statistics import systems_realization_gap


@dataclass
class BenchmarkConfig:
    """Settings for one benchmark sweep.

    Attributes:
        out: Directory for results.
        batch_sizes: Batch sizes to sweep.
        prompt_lens: Prompt lengths to sweep.
        new_tokens: Generation lengths to sweep.
        repeats: Timed repetitions per configuration.
        warmup: Untimed repetitions before each configuration, so kernel
            selection, memory pools, and any compilation are paid before the
            clock starts.
        seed: Seed for the model and the benchmark order shuffle.
        device: ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"``.
        checkpoint: Model to benchmark. ``None`` uses the small demonstration
            architecture, which is fine for checking the harness and useless as
            a latency measurement of anything real — a 64-wide toy and a 768-wide
            model do not share a kernel regime, and the vocabulary head that
            dominates a real decode step is 24k multiply-accumulates in one and
            39M in the other.
    """

    out: str = "results/latency"
    batch_sizes: tuple[int, ...] = (1, 4, 16)
    prompt_lens: tuple[int, ...] = (16, 48)
    new_tokens: tuple[int, ...] = (8, 32)
    repeats: int = 5
    warmup: int = 2
    seed: int = 0
    device: str = "auto"
    checkpoint: str | None = None


@dataclass
class Measurement:
    """Timings for one configuration.

    Attributes:
        depth: Executed depth, or ``"routed"``.
        batch_size: Requests per batch.
        prompt_len: Prompt length.
        new_tokens: Tokens generated.
        ttft_p50: Median time to first token, in milliseconds.
        ttft_p95: 95th percentile time to first token.
        tpot_p50: Median time per output token after the first.
        tpot_p95: 95th percentile time per output token.
        total_p50: Median end-to-end latency.
        total_p95: 95th percentile end-to-end latency.
        tokens_per_second: Throughput at the median end-to-end latency.
        peak_memory_bytes: Peak allocator memory, where the device reports it.
        kv_bytes: Cache bytes materialized.
        estimated_macs: Analytical multiply-accumulates for the same work.
        route_distribution: Requests per chosen depth.
        samples: Raw end-to-end timings, kept so the summary can be rechecked.
    """

    depth: int | str
    batch_size: int
    prompt_len: int
    new_tokens: int
    ttft_p50: float
    ttft_p95: float
    tpot_p50: float
    tpot_p95: float
    total_p50: float
    total_p95: float
    tokens_per_second: float
    peak_memory_bytes: int
    kv_bytes: int
    estimated_macs: float
    route_distribution: dict[int, int] = field(default_factory=dict)
    samples: list[float] = field(default_factory=list)


def resolve_device(name: str) -> torch.device:
    """Chooses the device to benchmark on.

    Args:
        name: ``"auto"`` or an explicit device string.

    Returns:
        The device.
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Waits for queued work, so a timer measures execution and not enqueueing.

    Args:
        device: Device to wait on. A no-op on CPU.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def percentile(values: list[float], fraction: float) -> float:
    """Reads a percentile from a small sample.

    Args:
        values: Observations.
        fraction: Percentile in ``[0, 1]``.

    Returns:
        The value at that percentile, by nearest rank. With the handful of
        repetitions a laptop sweep affords, a P95 is a description of the worst
        sample and should be read as such.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def peak_memory(device: torch.device) -> int:
    """Reads peak allocator memory since the last reset.

    Args:
        device: Device to query.

    Returns:
        Bytes, or zero where the device does not report it. CPU allocation is
        not tracked by torch, so zero there means "unavailable", not "none".
    """
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    return 0


def reset_memory(device: torch.device) -> None:
    """Resets the peak-memory counter where one exists.

    Args:
        device: Device to reset.
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


@torch.inference_mode()
def time_generation(
    model: Transformer,
    prompts: torch.Tensor,
    new_tokens: int,
    config: BenchmarkConfig,
) -> tuple[list[float], list[float], list[float]]:
    """Times prefill and decode separately, over several repetitions.

    Time to first token and time per output token are measured apart because
    routing affects them differently: a shallower prefill shortens the first,
    while the second is dominated by per-step overhead that routing does not
    touch.

    Args:
        model: Model with a router attached.
        prompts: Prompt ids.
        new_tokens: Tokens to generate.
        config: Benchmark settings.

    Returns:
        A tuple ``(ttft, tpot, total)`` of per-repetition milliseconds, where
        ``tpot`` is per generated token after the first.
    """
    device = prompts.device

    for _ in range(config.warmup):
        model.generate_routed(prompts, max_new_tokens=new_tokens, temperature=0.0)
    synchronize(device)

    ttft, tpot, total = [], [], []
    for _ in range(config.repeats):
        gc.collect()
        synchronize(device)

        start = time.perf_counter()
        model.generate_routed(prompts, max_new_tokens=1, temperature=0.0)
        synchronize(device)
        first = time.perf_counter()

        model.generate_routed(
            prompts, max_new_tokens=new_tokens, temperature=0.0
        )
        synchronize(device)
        end = time.perf_counter()

        ttft.append((first - start) * 1e3)
        total.append((end - first) * 1e3)
        tpot.append(
            ((end - first) * 1e3 - (first - start) * 1e3)
            / max(new_tokens - 1, 1)
        )
    return ttft, tpot, total


def measure(
    model: Transformer,
    depth: int | str,
    batch_size: int,
    prompt_len: int,
    new_tokens: int,
    config: BenchmarkConfig,
    device: torch.device,
) -> Measurement:
    """Benchmarks one configuration.

    Args:
        model: Model with a router already attached.
        depth: Depth label for the row.
        batch_size: Requests per batch.
        prompt_len: Prompt length.
        new_tokens: Tokens to generate.
        config: Benchmark settings.
        device: Device in use.

    Returns:
        The populated measurement.
    """
    generator = torch.Generator().manual_seed(config.seed)
    prompts = torch.randint(
        0, model.config.vocab_size, (batch_size, prompt_len), generator=generator
    ).to(device)

    reset_memory(device)
    ttft, tpot, total = time_generation(model, prompts, new_tokens, config)

    out = model.generate_routed(
        prompts, max_new_tokens=new_tokens, temperature=0.0
    )
    median_total = percentile(total, 0.5) + percentile(ttft, 0.5)

    return Measurement(
        depth=depth,
        batch_size=batch_size,
        prompt_len=prompt_len,
        new_tokens=new_tokens,
        ttft_p50=percentile(ttft, 0.5),
        ttft_p95=percentile(ttft, 0.95),
        tpot_p50=percentile(tpot, 0.5),
        tpot_p95=percentile(tpot, 0.95),
        total_p50=median_total,
        total_p95=percentile(total, 0.95) + percentile(ttft, 0.95),
        tokens_per_second=(
            batch_size * new_tokens / max(median_total / 1e3, 1e-9)
        ),
        peak_memory_bytes=peak_memory(device),
        kv_bytes=sum(out.trace.kv_bytes),
        estimated_macs=out.estimated_macs["total"],
        route_distribution=out.trace.depth_distribution,
        samples=total,
    )


def run(config: BenchmarkConfig) -> dict:
    """Sweeps every depth and shape, in randomized order.

    Order is randomized because a laptop's thermal state drifts over a sweep,
    and running full depth last would systematically hand it the worst clocks.

    Args:
        config: Benchmark settings.

    Returns:
        Results ready to serialize.
    """
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)

    if config.checkpoint:
        blob = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
        model = Transformer(blob["model_config"])
        incompatible = model.load_state_dict(blob["model"], strict=False)
        if incompatible.missing_keys:
            raise ValueError(
                f"{config.checkpoint} is missing "
                f"{len(incompatible.missing_keys)} parameter(s) this "
                f"architecture needs, starting with "
                f"{incompatible.missing_keys[:3]}."
            )
        model = model.to(device).eval()
    else:
        model = Transformer(TransformerConfig(**DEMO_MODEL)).to(device).eval()

    cost_model = AnalyticalCostModel.from_config(model.config)
    tiers = model.config.exit_depths

    # Checked here rather than left to fail deep inside a decode loop, several
    # minutes into a queued job, with a stack trace that does not name the flag
    # at fault.
    limit = model.config.max_seq_len
    longest = max(config.prompt_lens) + max(config.new_tokens)
    if longest > limit:
        raise ValueError(
            f"prompt_lens up to {max(config.prompt_lens)} plus new_tokens up to "
            f"{max(config.new_tokens)} needs {longest} positions, but this "
            f"model was built for {limit}. Lower --prompt_lens/--new_tokens, or "
            f"benchmark a model trained at a longer context."
        )

    jobs = [
        (depth, batch, prompt, tokens)
        for depth in tiers
        for batch in config.batch_sizes
        for prompt in config.prompt_lens
        for tokens in config.new_tokens
    ]
    order = torch.randperm(len(jobs), generator=torch.Generator().manual_seed(
        config.seed
    )).tolist()

    measurements: list[Measurement] = []
    for position, index in enumerate(order, start=1):
        depth, batch, prompt, tokens = jobs[index]
        model.attach_router(
            RoutingConfig(
                routing_mode="fixed", fixed_depth=depth, probe_depth=1
            )
        )
        print(
            f"  [{position}/{len(jobs)}] depth {depth} batch {batch} "
            f"prompt {prompt} tokens {tokens}",
            flush=True,
        )
        measurements.append(
            measure(model, depth, batch, prompt, tokens, config, device)
        )

    return {
        "device": str(device),
        "tiers": list(tiers),
        "model_config": asdict(model.config),
        "head_to_block_ratio": cost_model.head_to_block_ratio,
        "measurements": [asdict(m) for m in measurements],
        "realization": realization_table(measurements, model.config.n_layers),
    }


def realization_table(
    measurements: list[Measurement],
    full_depth: int,
) -> list[dict]:
    """Compares each depth's arithmetic saving against its latency saving.

    Args:
        measurements: Every measured configuration.
        full_depth: Depth used as the baseline.

    Returns:
        One row per shallow configuration, with both savings and the gap.
    """
    baseline = {
        (m.batch_size, m.prompt_len, m.new_tokens): m
        for m in measurements
        if m.depth == full_depth
    }

    rows = []
    for m in measurements:
        key = (m.batch_size, m.prompt_len, m.new_tokens)
        reference = baseline.get(key)
        if reference is None or m.depth == full_depth:
            continue

        gap = systems_realization_gap(
            theoretical_saving=1.0 - m.estimated_macs / reference.estimated_macs,
            measured_saving=1.0 - m.total_p50 / reference.total_p50,
        )
        rows.append(
            {
                "depth": m.depth,
                "batch_size": m.batch_size,
                "prompt_len": m.prompt_len,
                "new_tokens": m.new_tokens,
                "kv_saving": 1.0 - m.kv_bytes / max(reference.kv_bytes, 1),
                **gap,
            }
        )
    return rows


def format_markdown(results: dict) -> str:
    """Renders the benchmark as a readable summary.

    Args:
        results: Output of :func:`run`.

    Returns:
        A Markdown document.
    """
    lines = [
        "# Measured latency of depth-capped generation",
        "",
        f"- device: `{results['device']}`",
        f"- depths: {results['tiers']}",
        f"- one vocabulary head costs "
        f"{results['head_to_block_ratio']:.2f} blocks at this width",
        "",
        "## Timings",
        "",
        "| depth | batch | prompt | tokens | TTFT p50 | TTFT p95 | "
        "TPOT p50 | TPOT p95 | total p50 | tok/s | KV bytes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in sorted(
        results["measurements"],
        key=lambda m: (m["batch_size"], m["prompt_len"], m["new_tokens"], m["depth"]),
    ):
        lines.append(
            f"| {m['depth']} | {m['batch_size']} | {m['prompt_len']} | "
            f"{m['new_tokens']} | {m['ttft_p50']:.2f} | {m['ttft_p95']:.2f} | "
            f"{m['tpot_p50']:.3f} | {m['tpot_p95']:.3f} | "
            f"{m['total_p50']:.2f} | {m['tokens_per_second']:.1f} | "
            f"{m['kv_bytes']:,} |"
        )

    lines += [
        "",
        "## Realization: does the arithmetic saving reach the clock?",
        "",
        "| depth | batch | prompt | tokens | MAC saving | latency saving | "
        "KV saving | gap | ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        results["realization"],
        key=lambda r: (r["batch_size"], r["prompt_len"], r["new_tokens"], r["depth"]),
    ):
        lines.append(
            f"| {row['depth']} | {row['batch_size']} | {row['prompt_len']} | "
            f"{row['new_tokens']} | {row['theoretical_saving']:.1%} | "
            f"{row['measured_saving']:.1%} | {row['kv_saving']:.1%} | "
            f"{row['realization_gap']:+.1%} | {row['realization_ratio']:.2f} |"
        )

    lines += [
        "",
        "A ratio near one means the arithmetic saving reached the clock; near "
        "zero means it did not; negative means routed inference was slower. "
        "The KV column is different in kind — cache memory falls exactly in "
        "proportion to depth because the entries are never allocated, so it is "
        "the one saving that does not depend on kernel behaviour.",
        "",
        "### What this sweep does and does not cover",
        "",
        "Every batch here runs at a **single depth**, which is the favourable "
        "case and the reason the ratios are as high as they are. It is the "
        "right measurement for the claim that a depth-capped request is "
        "genuinely cheaper, and it is *not* a measurement of a live server, "
        "where requests arrive at mixed depths and either get bucketed — "
        "adding scheduling delay and shrinking each kernel — or share a batch "
        "at the deepest depth present, which gives back most of the saving. "
        "Continuous batching under a realistic arrival process has not been "
        "benchmarked.",
        "",
        "This is also the point where request-level routing and token-level "
        "early exit differ most. Token-level exiting cannot deliver a "
        "batch-level saving at all, because the layer still runs for whichever "
        "row has not exited; request-level routing at least *can*, because a "
        "bucket is uniform for its whole lifetime.",
        "",
        "**These numbers are from a toy model on one laptop.** They establish "
        "that the measurement exists and is wired up, not what routing is "
        "worth on serving hardware.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    """Builds a :class:`BenchmarkConfig` from the command line.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed configuration.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="results/latency")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--prompt_lens", type=int, nargs="+", default=[16, 48])
    parser.add_argument("--new_tokens", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default=None)

    parsed = vars(parser.parse_args(argv))
    for key in ("batch_sizes", "prompt_lens", "new_tokens"):
        parsed[key] = tuple(parsed[key])
    return BenchmarkConfig(**parsed)


def main(argv: list[str] | None = None) -> None:
    """Runs the sweep and writes JSON plus Markdown.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.
    """
    config = parse_args(argv)
    provenance = RunRecord.create(
        "experiments.benchmark_latency",
        config=asdict(config),
        seeds={"model": config.seed, "prompts": config.seed,
               "order": config.seed},
        inputs={"checkpoint": config.checkpoint or "demonstration architecture"},
        notes=[
            "Measured wall-clock. Never mix these columns with the estimated "
            "multiply-accumulates from utils/costs.py.",
            "Benchmark order is randomized to spread thermal drift.",
            "Peak memory is unavailable on CPU and is reported as zero there.",
        ],
    )
    print(provenance.summary())
    print()

    results = run(config)
    path = Path(config.out)
    provenance.write(path / "latency.json", payload=results)
    (path / "latency.md").write_text(format_markdown(results))

    print()
    print(format_markdown(results))
    print(f"wrote {path / 'latency.json'} and {path / 'latency.md'}")


if __name__ == "__main__":
    main()
