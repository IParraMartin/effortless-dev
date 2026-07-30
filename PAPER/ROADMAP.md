# Roadmap: from working code to an empirical paper

The target claim, stated so it can be falsified:

> **Choosing how much of one elastic backbone to execute, per request, matches
> a family of independently trained same-family models on the quality-per-cost
> frontier — at a fraction of the training and storage cost, with proportional
> key/value memory savings, and with the saving realized in measured latency.**

Four conjuncts, each of which can fail separately. That is the point of writing
it this way.

## 1. Why this framing

The early-exit literature almost always benchmarks against *the same model at
full depth*. It rarely benchmarks against **just training a smaller model**,
which is the first thing a skeptical reader asks and the thing practitioners
actually do: Llama 1B/3B/8B/70B, Qwen 0.5B/1.5B/7B. The real baseline is a model
family, and it is largely unaddressed.

The reason to expect adaptivity to win is a property of **the request
distribution, not of architectures**. Difficulty is heavy-tailed: a large share
of traffic is formulaic and a minority carries real uncertainty. Any fixed-size
model spends the same compute on both, so uniform allocation is wasteful by
construction.

### Why *request*-level, and not per token

This is the reframing, and it is driven by three measurements rather than by
taste.

| | Token-level | Request-level |
|---|---|---|
| K/V memory saved | none (finding 9) | `1 − d/L` exactly (finding 2) |
| Latency at batch > 1 | none | realization ratio 0.90–0.92 (finding 4) |
| Cache approximation | propagated states | none (finding 1) |
| Decision cost | a softmax per checkpoint | one probe, one small MLP |

Per-token allocation is the more elegant idea and it is the one that cannot be
served. A layer runs for whoever in the batch has not exited, so its FLOP saving
does not become time; and its cache must be complete, so its memory saving is
zero. Request-level routing gives up granularity and gets a saving that
survives contact with a batch.

## 2. Positioning, honestly

"One model, many sizes" is not new. These are competitors, not related work:

| Prior approach | What it gives | What it does not |
|---|---|---|
| MatFormer, Once-for-All, slimmable nets | Many submodels from one training | Allocation still fixed per deployment |
| LayerDrop | Depth pruning at inference | Same |
| Distillation | A strong small model | One point per training run |
| Speculative decoding | Latency at unchanged quality | Does not reduce total compute |
| CALM, Depth-Adaptive Transformer, LayerSkip | Per-token depth | No K/V saving, no batched latency saving |

The differentiator is **per-request allocation with a real memory and latency
saving**. That makes MatFormer and LayerDrop *required* baselines, and it makes
the systems measurements load-bearing rather than an appendix.

> Literature check outstanding: 2025–2026 work has not been surveyed.

## 3. The main result

One figure. X-axis measured cost per request, Y-axis quality. The *differences*
between the curves are the scientific content:

| Curve | Isolates |
|---|---|
| (a) Independently trained models of increasing size | The staircase to beat |
| (b) Fixed endpoints of one elastic backbone | **Sharing tax** — cost of one backbone serving every size |
| (c) Learned request-level routing, swept over λ | **Adaptivity gain** |
| (dashed) Request-level oracle | Ceiling on (c) |
| (dotted) Best static mixture at matched cost | The floor (c) must clear to have shown anything |

`(a) − (b)` is the tax, `(c) − (b)` is the gain, and `(c) − (dotted)` is the
evidence that the controller reads the request rather than merely hitting a
budget. **How that decomposition scales is the interesting question**: if the
gain grows and the tax shrinks with model size, that is a claim about where the
field should go.

Two arguments to make in prose:

- **Total training cost to the whole frontier.** The family costs the sum of its
  runs; the elastic backbone costs one, plus multi-exit overhead. Storage
  likewise — and residency separately from storage, since a depth bucket only
  saves resident parameters if the server can avoid loading upper layers.
- **A runtime knob.** The family gives N discrete points chosen at deployment.
  With a utility-head controller, λ moves at inference on frozen weights, so one
  checkpoint exposes a continuum retunable per request against a live SLA.

## 4. What makes it credible

| Requirement | Minimum credible | Status |
|---|---|---|
| Scale | ~100M–1B params, a few B tokens, 3–4 sizes | not started |
| Independent model family | trained at matched budgets | **not started — this is the gap** |
| Evaluation | downstream zero-shot, not perplexity alone | not started |
| **Measured latency** | end-to-end, batch 1 and batched | done at toy scale only |
| K/V memory | measured, not derived | done |
| Statistics | paired bootstrap, predeclared margins | done |

The second row is the paper. Everything else in this repository is
infrastructure for it.

## 5. Order of operations

Ordered by how much each result can change the plan per unit of compute.

1. **Measure the sharing tax.** Cheap and decisive. Train one model with exits
   and one without at matched budget; compare **final-layer** quality. Multi-exit
   training can degrade the top layer, because shallow exits pull representations
   toward early linear decodability. If the tax is large, the thesis loses to
   "train one good model and distill", and that must be known first.
2. **Train the smallest credible model family** — three sizes at matched token
   budgets — and run the substitution test. This is the experiment the paper is.
3. **Measure whether a request-level depth signal exists on real prompts at
   all.** This is written third because it needs a trained model, and it has a
   strong claim to being first. Finding 6 is the reason: four deliberate
   attempts to *manufacture* a depth separation, knowing exactly what was
   wanted, produced one usable instrument — after doubling the training budget,
   and with a 3-point spread. Two of the other three produced a convincing
   gradient that was an artefact, one of memorization and one of a positional
   shortcut, and both looked *better* for being wrong. Real text is not
   hand-built and comes with no such knob.
4. **Continuous batching under mixed-depth arrival.** Finding 4 measures the
   favourable case. This decides deployability.

Steps 1 and 3 are days. Either can redirect or kill the project. Neither depends
on any of the propagation work.

## 6. Risks

- **The sharing tax may sink the claim.** Step 1 exists to find out first.
- **Real prompts may not separate by depth.** Finding 6 is the warning: three
  of four hand-built workloads produced no usable gradient, and two of those
  looked convincing until held out. Real text gets no such hand-building.
- **The controller may not find the gain even when it exists.** Finding 7
  measures an oracle gain of +0.05 and a learned router that captures none of
  it — it does not beat a coin flip at matched cost. Whether that is a probe
  problem, a data-volume problem, or a parameterization problem is unresolved
  and is the cheapest experiment left.
- **Adaptivity gain may collapse at scale.** Larger models may be confidently
  cheap about almost everything.
- **Bucketing may eat the latency saving.** Grouping by depth is what makes the
  saving real at batch > 1, and it also adds scheduling delay and shrinks each
  kernel.
- **The oracle is optimistic.** The token-level oracle inherits a full-depth
  replay assumption; the request-level oracle requires having run every
  endpoint. Both bound policies, neither is attainable.

## 7. Scope discipline

Learned KV propagation, exposure matching, and periodic refresh are **correct,
complete, and side quests**. Finding 11's correction makes the adapters more
effective than previously reported, and finding 12 still caps what that can buy.
Keep them as a correctness component of the token-level path. Do not build the
paper on them.

Token-level routing, if it returns, returns as an *increment on top of* a
working request-level system — decided inside a request's already-chosen
maximum depth — and only with active-row compaction, cheap checkpoint
controllers, and continuous-batching benchmarks. Not as the method.
