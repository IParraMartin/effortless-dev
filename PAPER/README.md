# Vertical routing — working notes

One transformer that can be executed to a chosen depth, and the question of
whether choosing that depth per request can stand in for keeping several
independently trained models of different sizes.

These notes are the write-up of a working codebase, not a finished paper. Where
a result is negative, corrected, or retracted it is recorded as such.

| Document | Contents |
|---|---|
| [METHOD.md](METHOD.md) | Architecture, request-level routing, the KV-cache problem, what is prior work |
| [EXPERIMENTS.md](EXPERIMENTS.md) | What is measured, how, and how to reproduce it |
| [FINDINGS.md](FINDINGS.md) | Results, with every claim labelled by evidential status |
| [ROADMAP.md](ROADMAP.md) | The paper: target claim, baselines, main figure, order of operations |

## How to read a claim here

Every numbered claim in [FINDINGS.md](FINDINGS.md) carries one of these labels.
They are not decoration; several claims have moved between them.

| Label | Means |
|---|---|
| **ESTABLISHED** | Verified by a test or measured on held-out data. Still at toy scale. |
| **CORRECTED** | Previously reported differently. The old number, the cause, and the new number are all stated. |
| **RETRACTED** | The measurement was invalid. Kept, struck through, with the reason. |
| **HYPOTHESIS** | Believed, argued for, not measured. |
| **PLANNED** | An experiment that has not been run. |

## The reframing

This repository began as a study of **token-level early exit**: every token
decides for itself whether to stop, using an entropy threshold. That work is
intact and still runs. It is no longer the primary method, for two measured
reasons and one structural one.

- Its compute saving does not become a latency saving at batch size above one,
  because the layer runs for whoever has not exited.
- It saves **no** key/value memory, because entries for skipped layers still
  have to be synthesized and stored.
- Its central approximation — propagating a hidden state upward to stand in for
  layers that never ran — is a source of error that request-level routing does
  not have at all.

**Request-level vertical routing** decides once, before generation, from a
cheap probe of the prompt, and then never executes or allocates anything above
the chosen depth. It gives up per-token granularity. In exchange every executed
layer sees exact keys and values, cache memory falls exactly in proportion to
depth, and a bucket of requests at one depth is uniform for its whole lifetime,
so the saving can reach the clock.

## Position relative to prior work

The token-level mechanism is **not novel**. Per-token adaptive depth with
per-layer exits is Elbayad et al. (2020); the decoder-only formulation with
hidden-state propagation, threshold calibration, and softmax confidence is CALM
(Schuster et al., 2022). LayerSkip (Elhoushi et al., 2024) contributes the
shared exit head and the early-exit training loss.

"One model, many sizes" is also prior art — MatFormer, Once-for-All, LayerDrop.
Those are competitors, not related work, and the paper's claim has to be stated
against them rather than against the full model.

What is less well covered, and is what this repository is now built around:

1. **Request-level depth as a substitute for a model family.** The early-exit
   literature benchmarks against the same model at full depth, rarely against
   *just training a smaller model*, which is what practitioners actually do.
2. **Honest cost accounting.** A vocabulary projection costs several blocks at
   realistic widths, so a policy that reads the softmax at every checkpoint can
   cost more than the depth it saves. The controller here never does.
3. **A substitution test that can fail.** Predeclared margins, paired
   bootstrap, and a harness that will not print "equivalent" unless the test
   passes.

> A literature check is still outstanding. Work from 2025–2026 has not been
> surveyed and any of this may already exist.

## What is not established

Two gaps, stated up front because everything else reads differently in their
light.

**The oracle gain is real but not reachable from the prompt.** A per-request
oracle beats the best fixed depth by +0.051, and even beats full depth outright
because depth is not monotone. But that oracle chooses by knowing the outcome. A
cross-fitted ceiling restricted to the probe features does not beat the best
fixed depth at all, and the trained controller is already at that ceiling (§7).
The prompt predicts request *difficulty* perfectly and still says almost nothing
about whether extra depth will help — the headroom is determined by the
continuation, which routing decides before seeing.

Getting even that much took four workload constructions, three of which failed
— two of them by producing a convincing depth gradient that turned out to be
memorization or a positional shortcut (§6). That is itself a warning about the
research direction: whether *real* prompts separate by required depth is
unestablished and is not something a corpus can be built to guarantee.

**The horizontal side has not been run.** No family of independently trained
models exists, so the sharing tax, the substitution ratio, and horizontal
regret are all unmeasured. The harness accepts a manifest and prints "not
evaluated" until one is supplied.

## Scale warning

Every number in these notes comes from models of at most 8 layers and 96 hidden
units, trained on synthetic corpora on a laptop with no GPU. They establish
that the mechanisms are correct and are suggestive about magnitudes. They do
**not** establish how any of this behaves at useful scale.
