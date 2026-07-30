# Current findings

Every claim carries an evidential label. See [README.md](README.md) for what
each one means. All numbers come from models of at most 8 layers on synthetic
corpora, trained on a laptop; treat magnitudes as suggestive and directions as
reasonably solid.

---

## Part I — request-level vertical routing

### 1. Depth-capped execution is exact — ESTABLISHED

Running the first `d` blocks and stopping reproduces a reference execution of
those blocks to floating-point noise, at every depth, in prefill and in
incremental decode, and under grouped-query attention with 1, 2, and 4 key/value
heads. Full-depth routed generation reproduces the pre-existing generation path
**token for token**.

This is worth stating plainly because it is the structural advantage over
token-level early exit: **request-level routing involves no cache
approximation**. Every layer that runs sees exactly the keys and values it
would have seen at full depth, because every layer below it also ran. The
propagation machinery in Part II exists only because the token-level path
cannot say this.

### 2. Cache memory falls exactly in proportion to depth — ESTABLISHED

A request routed to depth `d` allocates `d/L` of the full cache, measured, with
bytes per unit of depth identical across depths to the byte. Writing to a layer
above the cap raises rather than silently succeeding.

This is the one saving that does not depend on kernel behaviour, and it is
exactly the saving token-level early exit does **not** get: propagating states
upward still materializes every layer (finding 9).

### 3. One vocabulary projection per generated token — ESTABLISHED

Counted, not asserted: every readout in the model goes through one instrumented
call site. A routed generation of `n` tokens performs `n` projections, over the
final prompt position only, with no shallower exit evaluated on the way past.

The arithmetic that makes this matter: at the repository's 768-wide default
with a 52k vocabulary, one projection costs 39.9M multiply-accumulates against
7.1M for an entire block. **The head is 5.6 blocks.** A policy that consults the
softmax at three candidate depths spends more on deciding than two blocks of
depth would have cost. This is the reason the controller reads hidden states.

### 4. The arithmetic saving reaches the clock — ESTABLISHED, narrow conditions

Measured on CPU with a toy model, 19-token prompts, 16 generated tokens,
uniform-depth batches. Every depth against full depth:

| depth | batch | MAC saving | latency saving | KV saving | realization ratio |
|---:|---:|---:|---:|---:|---:|
| 1/6 | 1 | 82.7% | 76.0% | 83.3% | 0.92 |
| 2/6 | 1 | 66.1% | 60.5% | 66.7% | 0.92 |
| 3/6 | 1 | 49.6% | 44.6% | 50.0% | 0.90 |
| 4/6 | 1 | 33.1% | 30.6% | 33.3% | 0.92 |
| 1/6 | 8 | 82.7% | 75.3% | 83.3% | 0.91 |
| 2/6 | 8 | 66.1% | 60.0% | 66.7% | 0.91 |
| 3/6 | 8 | 49.6% | 45.0% | 50.0% | 0.91 |
| 4/6 | 8 | 33.1% | 29.8% | 33.3% | 0.90 |

Ratios of 0.90–0.92 across the board, and **they do not collapse at batch 8**,
which is the point of difference from token-level exiting. Over the whole sweep
(both batch sizes, 4- and 16-token generations) the ratio ranges 0.82–1.00, with
the worst cases at 4 generated tokens where fixed prefill overhead is a larger
share of the total.

The conditions are narrow and matter more than the numbers. Every batch here
runs at a single depth. A live server sees mixed depths and must either bucket
them — adding scheduling delay and shrinking each kernel — or run the batch at
the deepest depth present, which gives most of the saving back. Continuous
batching under a realistic arrival process has **not** been benchmarked.

### 5. Escalation from retained state is exact — ESTABLISHED

Continuing a shallow request through the upper blocks reproduces full-depth
execution exactly, and a cache widened by escalation decodes identically to one
that was deep from the start. The strategy is to retain the boundary activation
for **every** prompt position and replay only the suffix over it; the cost is
recorded as `backfill_tokens` and `backfill_blocks` rather than described as
free. Supplying only the last position's state raises, because upper blocks
would otherwise attend over holes.

### 6. A *generalizing* request-level depth gradient was hard to produce at toy scale — ESTABLISHED, and it is a warning

This started as housekeeping — build a corpus, run the pipeline — and became
the most consequential methodological result here. Four constructions were
built and measured. **Three failed outright, each differently, and the fourth
worked only at twice the training budget and yields a small gradient.**

| # | Construction | Outcome | Why |
|---|---|---|---|
| 1 | Induction across the stack | 1.000 at **every** depth | two layers suffice for induction — a known result that should have been anticipated |
| 2 | Memorization | 0.47 / 0.82 / 0.87 on **train**, chance on held-out | memorization does not generalize; depth was buying storage, and the deepest endpoint was the *worst* on unseen data |
| 3 | Fixed-offset repetition | flat | solvable by "copy from exactly *k* back" — one layer, no lookup |
| 4 | Value-conditioned indirection (read an offset token, then index by it) | 0.018 ≈ chance at every depth | genuinely needs two composed steps, and was simply not learned at `d_model=64` |

Construction 3 with the offset *varied*, forcing a content-based match, is the
one shipped. Its behaviour depends sharply on two things that are easy to treat
as incidental:

| Training | Held-out depth gradient |
|---|---|
| fixed corpus, 3000 steps | 0.29 → 0.24, **decreasing** — the gradient on the *training* split was 0.50 → 0.94, i.e. memorization |
| resampled, 3000 steps | flat (≈0.52 at every depth) — the rule was barely learned |
| resampled, 6000 steps | 0.838 → 0.868, monotone and real |

Only the third is both correct methodology and a usable instrument, and it took
twice the training budget to get there. The gradient it produces is *small*: a
3-point spread over six depths. Whether that is a property of this task or of
this scale is not established.

Two things follow, and the second is the important one.

**Methodological.** A depth gradient must come from a rule whose required
composition depth differs, must survive being held out, and must not be
substitutable by a positional shortcut. Any of the three can fail silently, and
two of them produce numbers that look *better* when they fail.

**For the research direction.** ROADMAP §6 lists "real prompts may not separate
by depth" as a risk. This finding sharpens it. Four deliberate attempts to
*manufacture* such a separation, with full knowledge of what was wanted,
produced one usable instrument — and only after doubling the training budget,
and with a 3-point spread. Real text is not hand-built and comes with no such
knob. Before investing in a model family, **measure whether a request-level
depth signal exists at all**. That is step 3 of the roadmap and has a good claim
to being step 1.

A caveat that cuts the other way: everything above is at `d_model=64` with six
layers. Larger models have more room for depth to matter, and the tasks that
separate by depth in real corpora are plausibly ones this scale cannot learn at
all — construction 4 is exactly that shape. The finding is "hard to produce
here", not "does not exist".

### 7. Most of the "oracle gain" is unreachable, and the controller is near optimal — CORRECTED

**Previously reported here:** an oracle adaptivity gain of +0.051 that the
learned router failed to capture, diagnosed as a controller problem.

**The diagnosis was wrong, and the harness caused it.** The plain oracle takes
`max` over candidates *per request*, which requires knowing how each one turned
out. No policy can have that. Judged against it, a router is charged for
information it was never given.

The decomposition, on 128 held-out requests:

| λ | plain oracle | reachable ceiling | best fixed | **learnable** gain | unreachable |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.8719 | 0.8176 | 0.8212 | **−0.004** | +0.054 |
| 0.20 | 0.8482 | 0.7918 | 0.8046 | **−0.013** | +0.056 |

The reachable ceiling is a deliberately over-powered per-tier quality regressor
— wider and deeper than the controller under test — fitted on the probe features
by 5-fold cross-fitting, so its prediction for a request never saw that request.
**It does not beat the best fixed depth either.** Router regret against it is
−0.012 to +0.005, i.e. the controller is already at the ceiling.

Three independent checks agree:

1. **The features are not the problem.** A small network recovers the
   ground-truth easy/hard label from the probe features with **1.000** held-out
   accuracy. The signal the prompt determines is fully present.
2. **The gain is not maximization bias.** Choosing the tier with one quality
   measurement and scoring with an independent one (teacher-forced vs
   free-running, correlation 0.87) retains +0.043 of the +0.051. So the chosen
   tier really is better; the oracle is not just picking noise.
3. **But the choice is not a function of the prompt.** The best policy allowed
   to depend only on easy/hard — the one thing the prompt fixes, and which is
   perfectly predictable — is +0.008 above the best fixed depth at λ=0.1 and
   **exactly zero** at λ=0.2. The oracle sends 210 of 256 requests to depth 1
   and the remaining 46 are an idiosyncratic minority.

So the headroom is real, stable, and **determined by the continuation rather
than by the prompt**. Request-level routing cannot reach it by construction,
because the decision is made before the continuation exists.

**What this changes.** `oracle − best_fixed` is the wrong ceiling to quote for a
router, and quoting it turns a near-optimal policy into an apparent failure. The
harness now reports a cross-fitted **conditional oracle** alongside it and splits
the gain into learnable and unreachable parts; the Markdown output tells the
reader to read the learnable column. Proposition 1 in the math foundations is
still true — it is a statement about ideal utilities — but its *empirical*
counterpart is not a bound on what any router can achieve.

This also sharpens finding 6. The workload has a depth gradient, and the prompt
predicts difficulty perfectly, and there is *still* almost nothing to route on,
because knowing a request is hard does not tell you that extra depth will help
*this* request. Whether real prompts do better is unmeasured and is the question
that matters.

## Part II — token-level early exit

This is the original work. It is correct, it is kept, and findings 9–12 are why
it is no longer the primary method.

### 8. The tradeoff curve is well behaved — ESTABLISHED

On a capacity-limited corpus where depth buys accuracy (per-exit accuracy
0.06 → 0.31 → 0.60 → 0.71 → 0.73 → 0.73):

| threshold | mean depth | accuracy | compute saved |
|---|---|---|---|
| 0.00 | 6.00/6 | 0.7349 | 0.0% |
| 0.20 | 5.46/6 | 0.7349 | **9.0%** |
| 0.40 | 4.61/6 | 0.7261 | 23.2% |
| 0.70 | 2.42/6 | 0.4453 | 59.7% |
| 1.00 | 2.00/6 | 0.3086 | 66.7% |

Accuracy and NLL both move monotonically, which is the check that the
uncertainty measure is sane.

**Caveat:** this is the calibration sweep, which replays a full-depth pass and
therefore overstates achievable quality.

### 9. Token-level exiting saves no cache memory — ESTABLISHED

Structural, not empirical. A token that exits at layer `L` still needs entries
at every layer above it, or later tokens cannot attend to it. Those entries are
synthesized rather than computed, which saves the attention and feed-forward
work but stores exactly as many bytes as full depth.

Compare finding 2. This is the clearest single reason the request-level
formulation is the better basis for a serving claim.

### 10. Readout redundancy is large; the entropy policy captures little of it — ESTABLISHED

| policy | mean depth | blocks saved | tokens stopped early |
|---|---|---|---|
| `teacher_forced_top1_agreement_oracle_exact_cache` | 3.13/6 | **47.8%** | 95.6% |
| entropy threshold, no accuracy loss | 5.46/6 | 9.0% | — |

The heuristic captures 19% of that headroom.

**The name is the finding.** Every qualifier is load bearing and dropping any
one overstates the result:

- *teacher forced* — the token sequence is given, so this is not what free
  generation would do;
- *top-1 agreement* — sufficiency means reproducing the final exit's argmax
  **for the current token**, and matching the current token does not imply the
  two paths stay together, because the shallow state also enters the cache that
  later tokens read;
- *exact cache* — it replays a full-depth pass, so every exit saw true keys and
  values.

What it does bound is *readout redundancy*: how much of the stack is already
carrying the final answer. That is a real and useful diagnostic. It is not a
serving result, and an earlier version of these notes leaned on it harder than
the qualifiers allow.

### 11. Learned KV propagation helps the cache more than previously reported — CORRECTED

**Previously reported:** deep-layer cache error reduced 12–16%, reconstruction
error 0.1359 → 0.1102 (−19%).

**Cause of the correction.** Two bugs, both fixed. The model-wide
initialization pass overwrote the adapters' deliberate zero initialization, so
they were *not* the identity at the start; and the backbone was trained with
the propagation loss active, so the "plain" baseline had already been shaped by
adapter training. The baseline was contaminated. The `plain` arm now has its
adapters **removed**, not zeroed, so it cannot acquire any.

**Now measured:**

```
cache key error   L0     L1     L2     L3     L4     L5     L6     L7
plain            0.000  0.000  0.000  0.472  0.504  0.627  0.696  0.678
learned          0.000  0.000  0.000  0.299  0.350  0.443  0.506  0.511
                                      -37%   -31%   -29%   -27%   -25%
```

Reconstruction error 0.1850 → 0.1081, a 42% reduction rather than 19%.

**The conclusion is unchanged and is the point.** End-to-end agreement moves
0.956 → 0.958 at threshold 0.4 — within noise. The cache is now measurably
*much* better and the predictions are still not measurably better. Finding 11
explains why. A correction that doubles the size of an effect while leaving the
conclusion intact is the most useful kind.

### 12. Hidden states are far more robust to cache corruption than the cache error suggests — ESTABLISHED

The central quantitative result of Part II, and what explains findings 11
and 13. With cached keys off by 50% in relative L2, **exit hidden states move
only 1.5%**. The residual stream dominates each layer's attention contribution.

The gap is driven by exit-depth **disparity**, not depth:

| exit pattern | state gap |
|---|---|
| all positions at the same depth | **0.0000** (exactly) |
| 10% shallow | 0.0123 |
| 50% shallow | **0.0377** |
| 90% shallow | 0.0255 |

Zero for uniform depth is structural: if every position stops at the same
layer, nothing below any exit is corrupted and no token attends to another
token's approximation.

**This is also an argument for request-level routing**, and it was not
recognized as one at the time. A request routed to one depth has *no*
disparity within itself, so it sits at the exactly-zero row by construction.

### 13. Drift saturates rather than compounding — ESTABLISHED

Teacher-forced, mean depth 6.2/8, agreement 0.85:

| positions | 0–16 | 16–32 | 32–48 | 48–64 | 64–80 | 80–96 | 96–112 |
|---|---|---|---|---|---|---|---|
| mean KL | 0.143 | 0.156 | 0.199 | **0.247** | 0.238 | 0.204 | 0.211 |

Error rises over roughly 60 positions and then plateaus. Needs checking on a
real corpus and a longer context.

### 14. Exposure-matched adapter fitting shows no benefit — ESTABLISHED at this scale

| arm | error on teacher states | error on deployed states |
|---|---|---|
| plain | 0.1850 | 0.1852 |
| teacher | 0.1081 | 0.1082 |
| simulated | 0.1079 | 0.1080 |

Identical to three decimals on both distributions.

**A fair test that returned a genuine null,** because the premise was measured
rather than assumed: the exposure gap is 3.1% (finding 12), while the adapters
change reconstruction error by 42%. An input shift an order of magnitude
smaller than the effect being fitted should not be expected to matter.

Practical conclusion: teacher-forced fitting is sufficient. The scale-dependent
conclusion is weaker — deeper models, real corpora, and longer contexts all
plausibly widen the gap.

### 15. Periodic refresh is not a good mitigation — ESTABLISHED

Compute-matched against simply lowering the threshold: marginal gains at long
periods (+0.007 to +0.017), a clear **loss** at short ones (−0.038 at period 4).
Free-running generation diverges at tokens 2–9, so a refresh at period 8 or 4
arrives after the damage. Only period 2 helps materially, and by then mean depth
is 7.4 of 8 and the savings are gone.

The direction this points: refresh on a **staleness estimate** rather than a
clock.

---

## Bugs the measurements caught that reasoning had not

Recorded because each one produced a plausible-looking number first.

1. **Adapters were not the identity at initialization.** `Module.apply` visited
   them after their explicit zero-init. Everything downstream of "an untrained
   adapter reproduces plain propagation exactly" was false.
2. **The control arm was trained.** The exposure backbone ran with the
   propagation loss active, so `plain` was not plain.
3. **Hidden-state supervision did not transfer.** Halved the training objective
   and moved real cache error by 0.1%. RMSNorm discards the magnitude that
   dominated the L2 error; supervising the post-norm direction fixed it.
4. **The adapter was applied at gap 0**, corrupting a layer that was exact by
   construction (0.000 → 0.142).
5. **A collective inside `if is_main`** deadlocked DDP — rank 0 waited forever
   for peers that never called it.
6. **A RoPE offset read from the cache mid-stack**, so deeper layers rotated at
   position *n+1*.
7. **Padding by repetition destroyed an induction corpus**, making held-out
   accuracy sit at chance and look like a modelling failure.

---

## Open directions, ranked

1. **Train a family of independent models and run the substitution test.**
   Everything else is preparation for this. The harness is ready, the manifest
   format is defined, and until it runs the central question is unanswered.
   — PLANNED
2. **Measure the sharing tax.** Train one multi-exit model and one plain model
   at matched budget and compare *final-layer* quality. Cheap, decisive, and if
   the tax is large the thesis loses to "train one good model and distill".
   — PLANNED
3. **Re-measure everything on real text at 100M+.** Every workload here is
   synthetic and the depth structure was hand-built. Whether real prompts carry
   a request-level depth signal at all is the load-bearing empirical
   assumption. — PLANNED
4. **Continuous batching under mixed-depth arrival.** Finding 4 is measured in
   the favourable case. This is what decides deployability. — PLANNED
5. **Learned halting for the token-level path**, supervised by the oracle in
   finding 10. Still the largest headroom *within* the token-level formulation,
   but that formulation now has findings 9 and 4 against it. — PLANNED

Demoted: learned KV propagation and exposure matching. They are correct and
finding 11's correction makes them *more* effective than reported, but finding
11 caps what that effectiveness can buy. Keep the adapters as a correctness
component of the token-level path, not as a contribution.

See [ROADMAP.md](ROADMAP.md) for how these feed into a paper.
