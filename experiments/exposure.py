"""Does fitting the propagation adapters on their own outputs help?

The adapters that repair a propagated hidden state have to be fitted on *some*
distribution of exit states. The cheap choice is the states a full-depth pass
produces, which are free because training computes them anyway. The honest
objection is that no such state exists at inference: by the time a token
reaches its exit, every layer beneath it has attended over a cache already full
of approximations, so the state handed to the adapter is not the clean one it
was fitted on.

This experiment measures whether that mismatch matters, and whether closing it
helps. Three arms share one frozen backbone, so the only thing that varies is
how the adapters were fitted:

    ``plain``      no adapters at all, the CALM-style baseline
    ``teacher``    adapters fitted on full-depth states
    ``simulated``  adapters fitted on states early exiting really produces

Run from the repository root::

    python -m experiments.exposure
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from src.config import TransformerConfig
from src.model import Transformer
from utils.drift import measure_drift
from utils.provenance import RunRecord

#: Backbone used by every arm. Large enough that depth buys accuracy, small
#: enough to train on a laptop.
MODEL = dict(
    vocab_size=256,
    d_model=96,
    n_layers=8,
    n_heads=4,
    n_kv_heads=2,
    max_seq_len=160,
    min_exit_layer=1,
)


def induction_corpus(
    n_sequences: int = 512,
    half_length: int = 48,
    vocab_size: int = 256,
) -> torch.Tensor:
    """Builds a corpus whose second half is only predictable by attending.

    Each sequence is a random block concatenated with itself, so predicting any
    token in the second half means finding its earlier occurrence and reading
    off what followed. That is the induction pattern, and it makes attention
    load-bearing by construction.

    The choice matters more than it looks. On a corpus of independent random
    tokens, attention carries almost no information, hidden states barely
    depend on the cache, and corrupting the cache therefore changes nothing
    measurable. Any experiment about cache quality run on such a corpus reports
    a null result by construction rather than by fact.

    Args:
        n_sequences: Number of sequences.
        half_length: Length of the repeated block.
        vocab_size: Token vocabulary size.

    Returns:
        Token ids shaped ``(n_sequences, 2 * half_length)``.
    """
    block = torch.randint(0, vocab_size, (n_sequences, half_length))
    return torch.cat([block, block], dim=1)


def train_backbone(steps: int = 800, seed: int = 0) -> tuple[Transformer, torch.Tensor]:
    """Trains a backbone on the induction task.

    Args:
        steps: Optimizer steps.
        seed: Random seed.

    Returns:
        A tuple ``(model, corpus)``. The model has adapters enabled but
        untouched, so it starts exactly at plain propagation.
    """
    torch.manual_seed(seed)
    model = Transformer(
        TransformerConfig(
            **MODEL,
            learned_kv_propagation=True,
            kv_propagation_weight=0.0,
        )
    )
    corpus = induction_corpus(vocab_size=MODEL["vocab_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(steps):
        batch = corpus[torch.randint(0, corpus.size(0), (16,))]
        # kv_propagation_weight is zero here so the backbone is trained without
        # any adapter influence; the arms below fit the adapters afterwards.
        out = model(batch[:, :-1], targets=batch[:, 1:])
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()

    model.eval()
    return model, corpus


def fit_adapters(
    model: Transformer,
    corpus: torch.Tensor,
    exposure: str,
    steps: int = 400,
    seed: int = 1,
) -> Transformer:
    """Fits the propagation adapters on a frozen backbone.

    Freezing the backbone is what makes the comparison clean: the arms cannot
    differ because one of them happened to train the transformer better. Only
    the adapter weights move, and only the source of their input states
    differs.

    Args:
        model: Model to copy and fit. Left untouched.
        corpus: Token corpus to draw batches from.
        exposure: ``"teacher"`` or ``"simulated"``.
        steps: Optimizer steps.
        seed: Random seed.

    Returns:
        A fitted copy of ``model``.
    """
    torch.manual_seed(seed)
    fitted = copy.deepcopy(model)
    fitted.config.kv_exposure = exposure
    fitted.eval()  # keeps the simulation free of dropout

    parameters = [
        parameter
        for block in fitted.blocks
        for parameter in block.kv_adapter.parameters()
    ]
    optimizer = torch.optim.AdamW(parameters, lr=3e-3)

    for _ in range(steps):
        batch = corpus[torch.randint(0, corpus.size(0), (16,))]
        inputs = batch[:, :-1]
        with torch.no_grad():
            hidden = fitted._run_blocks(inputs)
        valid = torch.ones_like(inputs, dtype=torch.bool)

        loss = fitted._propagation_loss(hidden, valid, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return fitted


def strip_adapters(model: Transformer) -> Transformer:
    """Builds the genuine no-adapter arm by removing the adapters.

    A zero-initialized adapter *is* the identity, so it is tempting to use the
    untouched backbone as the ``plain`` baseline and rely on that. Relying on
    it is the problem: the equality holds only as long as nothing in the
    codebase perturbs those weights, and this repository has already had one
    bug where a model-wide initialization pass silently overwrote them. An arm
    that is supposed to represent "no learned propagation" should not be able
    to acquire any, so the modules are removed and the configuration flag is
    turned off rather than assumed inert.

    Args:
        model: Model to copy. Left untouched.

    Returns:
        A copy with ``learned_kv_propagation`` disabled and every
        ``kv_adapter`` set to ``None``, so :meth:`DecoderBlock.repair` returns
        its input by construction rather than by arithmetic.
    """
    plain = copy.deepcopy(model)
    plain.config.learned_kv_propagation = False
    plain.config.kv_propagation_weight = 0.0
    for block in plain.blocks:
        block.kv_adapter = None
    return plain


@torch.no_grad()
def reconstruction_error(
    model: Transformer,
    inputs: torch.Tensor,
    exposure: str,
    seed: int = 7,
) -> float:
    """Scores the adapters against one distribution of exit states.

    Evaluating each arm under *both* distributions is what exposes the gap:
    adapters fitted on clean states can look excellent on clean states and no
    better than nothing on the states they will actually receive.

    Args:
        model: Model to score.
        inputs: Token ids.
        exposure: Distribution to draw source states from.
        seed: Seed fixed so every arm sees the same sampled exit depths.

    Returns:
        Relative squared reconstruction error. Zero is perfect; the value plain
        propagation achieves is the number to beat.
    """
    original = model.config.kv_exposure
    model.config.kv_exposure = exposure
    torch.manual_seed(seed)

    hidden = model._run_blocks(inputs)
    valid = torch.ones_like(inputs, dtype=torch.bool)
    loss = model._propagation_loss(hidden, valid, inputs)

    model.config.kv_exposure = original
    return float(loss)


@torch.no_grad()
def gap_scan(model: Transformer, corpus: torch.Tensor) -> str:
    """Traces the exposure gap against the two things that should widen it.

    A single gap measurement cannot distinguish "this effect is negligible"
    from "this effect is negligible *here*". Both axes scanned below are ones
    that grow in real deployments: contexts are longer than a toy probe, and
    useful early exiting stops tokens far shallower than the midpoint. If the
    gap is flat in both, it is genuinely small; if it climbs, the null result
    above is a statement about scale rather than about the method.

    Args:
        model: Model to measure.
        corpus: Token corpus to slice probes from.

    Returns:
        A multi-line table.
    """
    lines = [f"{'context':>9}  {'exit pattern':>16}  {'state gap':>10}"]
    lines.append("-" * len(lines[0]))

    for length in (16, 32, 64, 96):
        probe = corpus[:8, :length]
        gap, _ = exposure_gap(model, probe)
        lines.append(f"{length:>9}  {'uniform random':>16}  {gap:>10.4f}")

    # Forcing every position to the same depth measures nothing: with no
    # disparity, no token ever attends to another token's approximation, and
    # the gap is exactly zero by construction. The quantity that drives it is
    # the *spread* of exit depths, so that is what is scanned here.
    probe = corpus[:8]
    deep = model.config.n_layers - 1
    for fraction in (0.1, 0.25, 0.5, 0.75, 0.9):
        exits = torch.full(probe.shape, deep, dtype=torch.long)
        shallow = torch.rand(probe.shape) < fraction
        exits[shallow] = model.config.min_exit_layer
        gap, _ = exposure_gap(model, probe, exits=exits)
        lines.append(
            f"{probe.size(1):>9}  {f'{fraction:.0%} shallow':>16}  {gap:>10.4f}"
        )
    return "\n".join(lines)


@torch.no_grad()
def exposure_gap(
    model: Transformer,
    inputs: torch.Tensor,
    seed: int = 7,
    exits: torch.Tensor | None = None,
) -> tuple[float, float]:
    """Measures how far the deployed exit states sit from the clean ones.

    This is the premise of the whole experiment, so it is measured rather than
    assumed. If the two distributions coincide there is no gap to close, and a
    null result downstream says nothing about the method.

    Args:
        model: Model to measure.
        inputs: Token ids.
        seed: Seed fixing the sampled exit depths.
        exits: Explicit exit depth per position. When ``None``, depths are
            sampled uniformly.

    Returns:
        A tuple ``(state_gap, task_accuracy)``. The first is the relative
        difference between simulated and full-depth exit states; the second is
        next-token accuracy, included so a gap measured on a model that has not
        learned the task can be recognized as meaningless.
    """
    torch.manual_seed(seed)
    if exits is None:
        exits = model._sample_exit_layers(inputs.shape, inputs.device)

    hidden = model._run_blocks(inputs)
    stacked = torch.stack(hidden)
    index = exits.unsqueeze(0).unsqueeze(-1).expand(1, *exits.shape, stacked.size(-1))
    clean = torch.gather(stacked, 0, index).squeeze(0)

    deployed = model.simulate_early_exit(inputs, exits)
    gap = float(
        (deployed - clean).pow(2).sum().sqrt() / clean.pow(2).sum().sqrt().clamp(min=1e-9)
    )

    logits = model.exit_modules[-1](hidden[-1])
    accuracy = float((logits[:, :-1].argmax(-1) == inputs[:, 1:]).float().mean())
    return gap, accuracy


#: Seeds this experiment consumes, named by what each one controls. Recorded
#: rather than passed around as bare literals, so a rerun is specified by the
#: output file instead of by reading the source.
SEEDS = {"backbone": 0, "adapter_fit": 1, "evaluation": 7}


def main(out_path: str | Path = "results/exposure.json") -> None:
    """Runs the three-arm comparison, prints it, and records it.

    Args:
        out_path: Where to write the machine-readable record. The console
            output is a rendering of the same numbers; the file is what can be
            compared against a later run.
    """
    record = RunRecord.create(
        "experiments.exposure",
        config={"model": MODEL, "backbone_steps": 800, "adapter_steps": 400},
        seeds=SEEDS,
        inputs={"corpus": "induction_corpus (synthetic, generated in-process)"},
        notes=[
            "Toy scale: 8 layers, d_model 96, synthetic induction corpus.",
            "The 'plain' arm has its adapters removed, not zeroed.",
        ],
    )
    results: dict[str, object] = {}
    print(record.summary())
    print()

    print("training backbone ...")
    base, corpus = train_backbone(seed=SEEDS["backbone"])
    probe = corpus[:8]

    gap, accuracy = exposure_gap(base, probe)
    results["backbone_accuracy"] = accuracy
    results["exposure_gap"] = gap
    print(f"\nbackbone next-token accuracy: {accuracy:.3f}")
    print(f"exposure gap (deployed vs clean exit states): {gap:.4f}")
    if gap < 0.02:
        print("  WARNING: the two distributions nearly coincide, so any")
        print("  downstream difference between the arms is uninformative.")

    print("\n=== does the gap widen under harsher conditions? ===")
    print(gap_scan(base, corpus))

    arms = {
        "plain": strip_adapters(base),
        "teacher": fit_adapters(base, corpus, "teacher", seed=SEEDS["adapter_fit"]),
        "simulated": fit_adapters(
            base, corpus, "simulated", seed=SEEDS["adapter_fit"]
        ),
    }

    print("\n=== reconstruction error, scored under both distributions ===")
    print("(lower is better; 'plain' is what doing nothing achieves)")
    header = f"{'arm':>10}  {'on teacher states':>18}  {'on deployed states':>19}"
    print(header)
    print("-" * len(header))
    reconstruction: dict[str, dict[str, float]] = {}
    for name, model in arms.items():
        teacher = reconstruction_error(model, probe, "teacher")
        deployed = reconstruction_error(model, probe, "simulated")
        reconstruction[name] = {"teacher": teacher, "deployed": deployed}
        print(f"{name:>10}  {teacher:>18.4f}  {deployed:>19.4f}")
    results["reconstruction_error"] = reconstruction

    print("\n=== deployed cache key error by layer (threshold 0.6) ===")
    profiles = {
        name: measure_drift(model, probe, threshold=0.6)
        for name, model in arms.items()
    }
    results["cache_key_error"] = {
        name: [float(e) for e in profile.cache_error]
        for name, profile in profiles.items()
    }
    for name, profile in profiles.items():
        layers = " ".join(f"{float(e):.3f}" for e in profile.cache_error)
        print(f"{name:>10}  {layers}")

    print("\n=== end-to-end quality at matched depth ===")
    header = (
        f"{'arm':>10}  {'threshold':>9}  {'mean depth':>11}  "
        f"{'agreement':>10}  {'mean KL':>9}"
    )
    print(header)
    print("-" * len(header))
    end_to_end = []
    for threshold in (0.4, 0.6, 0.8):
        for name, model in arms.items():
            profile = measure_drift(model, probe, threshold=threshold)
            end_to_end.append(
                {
                    "arm": name,
                    "threshold": threshold,
                    "mean_exit_depth": profile.mean_exit_depth,
                    "agreement": profile.agreement_rate,
                    "mean_kl": profile.mean_kl,
                }
            )
            print(
                f"{name:>10}  {threshold:>9.2f}  {profile.mean_exit_depth:>11.2f}  "
                f"{profile.agreement_rate:>10.3f}  {profile.mean_kl:>9.4f}"
            )
        print()
    results["end_to_end"] = end_to_end

    written = record.write(out_path, payload=results)
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
