# 2026-07-29

This is the research diary: objective, hypotheses, experiment registry, and a
dated decision log. **`START_HERE.md` is the plan** — read that first if you want
to know what to run. `CURRENT.md` holds the state of the runs; `MIGRATIONS.md`
holds schema and default changes. The newest decision-log entry is at the bottom.

The registry below was cut from twelve experiments to three on 2026-07-29. The
cut entries are retained with their reasons rather than deleted.

## Project identity

**Working title:** *Elastic Without Regret: Can Vertical Routing Replace a Portfolio of Language Models?*

**Project name:** Effortless Vertical Routing

**Current phase:** Scope cut to three experiments (2026-07-29). The first two run
on the checkpoint already on disk; the third is one GPU job. No endpoint has yet
been scored on real held-out text. See `START_HERE.md`.

## Objective of the paper

The paper studies whether a single language model with multiple usable depth endpoints can replace some of the work normally handled by routing requests among separately trained models.

The central comparison is:

- **Horizontal routing:** select one model from a portfolio of independently trained models.
- **Vertical routing:** select how much depth to execute inside one shared backbone.
- **Vertical cascade:** begin at a shallow endpoint and reuse the already-computed prefix when escalating deeper.
- **Hybrid routing:** choose a model or specialist horizontally, then choose depth vertically within it.

The primary objective is to determine the region in which vertical routing is a practical substitute for horizontal routing. The paper should not claim that depth replaces model diversity universally. It should estimate the boundary: adjacent, same-family, general-purpose capacity tiers may be substitutable; heterogeneous specialists, modalities, tools, training data, or alignment policies may retain complementary errors that depth alone cannot reproduce.

## Current thesis

> A strong full-depth parent can be retrofitted with shallow endpoints and a request-level value-of-depth controller so that the resulting elastic model preserves the parent’s full-depth quality, exposes several useful quality–cost operating points, and recovers a measurable fraction of the quality–cost advantage of routing among independent same-family models.

This is a hypothesis, not yet an established result.

**Narrowed 2026-07-29.** The thesis previously ended with a further clause: *"Reusable
probe computation and depth-capped K/V caches can provide systems benefits that
horizontal cascades cannot obtain."* That clause is withdrawn. K/V memory under a
depth cap is verified to fall exactly `1 − d/L`, but turning an allocation saving
into a *serving* benefit needs the benchmark that was cut, so the paper claims the
invariant and no systems consequence of it.

## Paper innovations to target

### 1. Vertical substitution as the research object

The paper makes vertical and horizontal routing directly comparable at matched measured cost. The main output is not only an early-exit accuracy curve; it is an estimate of how much of a horizontal model portfolio’s frontier can be recovered by one elastic backbone.

### 2. No-regret retrofit

Rather than relying only on multi-exit pretraining from scratch, the main method starts from a strong pretrained parent and adds shallow endpoints while constraining the original full-depth endpoint not to regress. This isolates the value of elasticity from the damage caused by a poorly balanced joint objective.

### 3. Reusable shallow probing

The first few backbone layers act as a request probe. Their hidden states and K/V entries are reused when the request continues to its selected endpoint. The probe is productive model computation rather than a separate router whose work is discarded.

### 4. Measured-cost value-of-depth prediction

The controller estimates the expected marginal quality gain from additional depth and compares it with the actual incremental systems cost. It is not based only on output entropy. Its training target and deployment cost definition must match.

### 5. A decomposition of where routing gains come from

The evaluation separates:

- **endpoint or sharing tax:** fixed endpoint quality versus an independent model at comparable cost;
- **vertical adaptivity gain:** learned request-dependent depth versus the best request-independent depth or exact static mixture;
- **horizontal complementarity gap:** extra gain available only from independent models’ diverse errors;
- **router regret:** learned policy versus the best policy available from the same information;
- **systems realization gap:** theoretical compute savings versus measured latency, throughput, memory, and energy savings;
- **vertical substitution ratio:** the fraction of horizontal frontier improvement recovered vertically.

### 6. Token-level extension — cut

Token-level depth adaptation and learned K/V propagation were held as a later
extension. **Cut to future work on 2026-07-29**, and cut rather than deferred: the
diagnostic that would have gated them — token-level outcome-oracle gain beyond the
request-level cap — was never built, so nothing is waiting on a measurement. The
code remains in `src/model.py` and `experiments/exposure.py` and still runs.

## Research questions

Narrowed by the 2026-07-29 scope cut. The status column is the current one; the
questions are kept in full so the narrowing is legible.

| | question | status |
|---|---|---|
| 1 | Can shallow endpoints be added without degrading the parent's full-depth quality? | **live** — experiment A1 |
| 2 | At equal measured cost, how close are shared depth endpoints to independently trained same-family models? | **live** — A3 |
| 3 | Is there per-request heterogeneity in required depth, after controlling for a strong static mixture? | **live** — A2 |
| 4 | Can a controller infer that heterogeneity from reusable shallow hidden states? | **live** — A2 |
| 5 | How much of theoretical MAC and K/V saving becomes actual latency, throughput, memory and energy? | **partial** — K/V memory verified exactly; latency, throughput, goodput and energy unmeasured and not pursued |
| 6 | When does horizontal model complementarity remain irreducible? | **dropped** — needs heterogeneous specialists, not one same-family suite |
| 7 | Does request-level selection provide most of the gain, or is token-level adaptation necessary? | **dropped to future work** — its gating diagnostic was not built |

## Primary hypotheses

- **H1 — no-regret endpoint:** the retrofitted model's final endpoint is non-inferior to its frozen parent within a predeclared margin.
  *Instrument:* exact by construction in the frozen modes (`assert_parent_preserved`); `experiments/no_regret.py` for the rest. **Tested by A1.**
- **H2 — useful tiers:** at least two shallow endpoints lie on the empirical quality–cost frontier after including all readout and controller costs.
  *Instrument:* `pareto_frontier`, with adapter and LoRA MACs charged. **Tested by A1.**
- **H3 — learnable adaptivity:** a cross-fitted request router beats the best exact static mixture at matched cost on an untouched reporting split.
  *Instrument:* `probe_policy_gain` and the matched-cost mixture comparison, document-clustered. **Tested by A2.**
- **H4 — partial substitution:** the vertical substitution ratio is materially above zero for adjacent same-family model tiers.
  *Instrument:* `bootstrap_substitution_ratio`. **Tested by A3, and may be unreportable:** matching the token budget compresses the Pythia frontier — measured 0.016 bits/byte between 70m and 160m at `step1000` against 0.111 at `main` — and the ratio divides by that. If the denominator is too small, report the tax alone.
- **H5 — systems realization: dropped.** The serving benchmark is cut. K/V memory is verified exactly; nothing else is claimed.
- **H6 — boundary: dropped.** Requires heterogeneous specialists rather than one same-family suite.

## Evidence ledger

**This section is a snapshot dated 2026-07-29, taken before the implementation
work recorded in the decision logs below.** Several of its entries have since
changed status — exact resume, for instance, is now established by test, and the
independent family now exists. It is kept unedited because the decision logs refer
back to it. **For the live state of the evidence, read `CURRENT.md`.**

### Established by code inspection or deterministic tests

- The original routed generation path recomputed the shallow prompt probe even though its MAC counters treated the probe as reused.
- The original two model configurations did not receive identical backbone initializations from the same seed because constructing different exit sets consumed different random draws before the global initialization pass.
- The original trajectory cost model charged one too many incremental decode forwards for a generation of `N` tokens.
- The original latency summary added TTFT to an end-to-end duration that already included TTFT.
- The original distributed evaluation reported only the local rank’s validation shard rather than a global reduction.
- The original controller evaluation mixed calibration and reporting examples and selected a controller checkpoint by filename order.
- The original static-mixture baseline added avoidable Monte Carlo assignment noise.
- The corrective branch compiles and passes all 192 tests discovered in the supplied source tree when unavailable network-facing libraries are replaced by lightweight import stubs for tests that do not use them.

### Observed in the supplied run diary, but not independently reproduced here

- Both arms processed 2,499,608,576 tokens.
- The final-only arm reports held-out CE 3.2024; the six-exit arm reports held-out final CE 3.2768.
- The reported gap is 0.0744 nats, corresponding to a 7.72% perplexity ratio.
- The six-exit run took 11.68% longer and reported 10.46% lower token throughput.
- The six-exit trailing training CE is almost unchanged from depth 10 to depth 12.
- K/V cache allocation matched the analytical depth-cap formula in the reported audit.

### Not established

- A causal “sharing tax.” The arms differ in both backbone initialization and objective.
- That depth 10 is free. It is only close to depth 12 within the same degraded six-exit model.
- Held-out quality at depths 6, 8, and 10. Historical evaluation aliased with exit rotation and did not score them.
- Confidence intervals, seed variability, or a predeclared non-inferiority result.
- Learnable routing gain on real text.
- A true comparison with independent shallow models.
- Production latency or throughput benefit.
- Trained learned K/V propagation.
- Exact resume equivalence after preemption.

## Decision log — 2026-07-29

### Material received

- Updated repository source.
- `CURRENT.md` summarizing `vr-noexits` and `vr-exits` Savio runs.
- No checkpoints, raw per-request output, W&B history export, profiler trace, or hardware record.

### Results reinterpreted

1. **Previous claim:** “The sharing tax at full depth is real.”
   
   **Current status:** downgraded to “the selected six-exit training recipe produced a final-endpoint degradation in one reported run pair.” The comparison is not causal because common initialization was not preserved and the optimization objectives differ sharply.

2. **Previous claim:** “Depth 10 is 17% cheaper for nothing measurable.”
   
   **Current status:** retracted. Depth 10 and depth 12 are close only inside the six-exit model. Depth 10’s trailing CE is 0.0792 nats worse than the final-only model’s depth-12 trailing CE, an 8.24% perplexity ratio. Depth 10 also lacks held-out evaluation in the historical run.

3. **Previous claim:** reported routed MACs reflect probe reuse.
   
   **Current status:** retracted for the original implementation. The probe was executed, discarded, and recomputed. A patch now reuses its hidden state and depth-capped K/V cache.

4. **Previous claim:** matched token budget makes the pair a clean sharing comparison.
   
   **Current status:** retracted. Token exposure is matched, but initialization and objective are not.

5. **Previous claim:** 38,140 steps generated 2,499,608,576 tokens at 65,536 tokens per step.
   
   **Current status:** clarified. The token count equals exactly 38,141 optimizer updates. `38,140` is probably the zero-based final logging index. Future logs must distinguish `completed_updates` from `global_step_index`.

### Focus added

- No-regret retrofit from a common pretrained parent.
- Explicit comparison against independent same-family model tiers.
- Value-of-depth controller trained on real held-out trajectories.
- Exact static-mixture and feature-conditional policy baselines.
- Serving-level benchmarking with continuous batching and tail latency.
- Causal separation of endpoint tax, adaptivity, complementarity, and systems realization.

### Experiments added

- Re-evaluate every exit from both existing checkpoints on identical real held-out requests.
- Add post-hoc readouts to the final-only checkpoint to measure natural intermediate decodability.
- Run common-parent frozen-exit retrofit.
- Sweep anchored shallow-loss strength while fixing final CE coefficient at one.
- Compare frozen exits, selective unfreezing, LoRA, QLoRA at larger scale, and full fine-tuning.
- Train controllers with full-information supervised utility labels before attempting RL.
- Compare with a controlled horizontal family such as Pythia.
- Benchmark actual probe reuse and depth-bucketed serving.

### Experiments deleted or renamed

- The current `horizontal_cascade` must not be reported as a horizontal model cascade; it uses vertical endpoint outcomes and oracle escalation. Rename it `vertical_no_reuse_oracle_cascade` until independent models and a deployable escalation rule exist.
- The current `conditional_oracle` must not be described as a mathematical ceiling. Rename it `cross_fitted_probe_policy` or `estimated_attainable_policy`.
- Synthetic trajectory collection is retained only as a unit/mechanism test, not as evidence for the paper.

### Experiments deprioritized

- Learned K/V propagation and token-level routing move behind the request-level go/no-go gate.
- RL, PPO, or contextual-bandit training moves behind a strong full-information supervised controller.
- QLoRA moves behind a model scale at which frozen 4-bit backbone storage is actually a binding memory constraint.

## Active experiment registry

Cut to three on 2026-07-29. Each remaining experiment tests one claim, and each
can kill the next — which is why they run in order.

| ID | Experiment | Claim | Status | Kill condition |
|---|---|---|---|---|
| **A1** | Frozen retrofit of the final-only checkpoint, exits trained | no regret; useful tiers | Ready, no cluster cost | Shallow tiers no better than chance: the parent's intermediate states carry nothing and the method has no basis |
| **A2** | Real-text trajectories, controller, evaluation | learnable adaptivity | Ready | `probe-policy gain` at zero: requests do not differ in required depth |
| **A3** | Pythia 70m/160m/410m at `step1000` | substitution | Ready, one GPU job | Frontier too compressed to divide by: report the tax, not a ratio |

### Cut, with reasons

Retained so the history is legible and so a later reader knows these were
decisions rather than oversights.

| Former ID | Experiment | Why cut |
|---|---|---|
| E00 | Apply the critical-fixes patch | **Completed** 2026-07-29 (`5a8f3fb`), not cut |
| E01 | Re-score both existing checkpoints at every depth | Folded into A2, which scores endpoints on real held-out requests anyway |
| E02, E03 | Post-hoc readouts; frozen-exit retrofit | Merged into **A1** |
| E04, E05 | Anchored-objective sweep; selective unfreeze / LoRA | Cut. Both are *method* comparisons within the retrofit ladder. The paper needs one working rung, not a ladder survey. The machinery stays available. |
| E06 | Trajectory and controller study | Became **A2** |
| E07 | Independent horizontal family | Became **A3**, reduced from five Pythia tiers to three: 1b and 1.4b sit far past the backbone's capacity and do not inform substitution |
| — | Four Pile + NeoX scratch arms (64 GPU-hours), approved earlier the same day | **Cut.** They measure the cost of multi-exit *pre-training*, which is not the proposed method: a frozen parent pays no sharing tax by construction. Returns only if A1 shows frozen tiers are unusable, in which case trained exits become necessary and sharing becomes a real cost. |
| E08 | Serving benchmark | Cut. A second paper's worth of work. K/V memory is verified exactly; latency, throughput, goodput and energy are reported as unmeasured rather than estimated. |
| E09 | Distribution shift and safety by endpoint | Cut. Separate contribution. |
| E10 | Token-level routing, learned K/V propagation | Cut to future work. Its gating diagnostic was not built either, so the branch is closed rather than pending. |
| E11 | RL or contextual bandit controller | Cut to future work. Never justified before a supervised controller works. |

### What the cut costs

Conditions 6 through 10 of the brief's definition of done are not met and are not
being pursued: no serving benchmark, no energy, no continuous batching, no
token-level analysis, no multi-seed scratch comparison. The paper is smaller. It
is complete at that size — three claims, three experiments, each failure
informative — rather than eleven half-built ones.

## Claim discipline

Use these labels in notes and drafts:

- **Proved:** follows from stated assumptions by mathematics.
- **Tested invariant:** verified by deterministic implementation tests.
- **Observed:** read from one or more run records.
- **Estimated:** accompanied by a sampling method and uncertainty interval.
- **Established:** replicated and supports the predeclared claim.
- **Hypothesis:** not yet supported.
- **Retracted/corrected:** retained in the diary with the reason.

Do not convert analytical MAC savings into latency claims. Do not call a learned probe a ceiling. Do not call an endpoint “free” without a predeclared non-inferiority test against the proper comparator.

## Diary update procedure

Every substantive change should update this file.

1. Change the top-level title to the new date.
2. Preserve the project objective unless the objective itself changes; record that change explicitly.
3. Move contradicted claims into the new entry’s corrected/retracted section rather than deleting history.
4. Add, remove, rename, or gate experiments in the registry.
5. State which artifacts support each new claim: checkpoint, commit, config, data hash, raw result, script, and statistical report.
6. Keep one predeclared primary endpoint and cost operating point per study.

### Entry template

```markdown
## Decision log — YYYY-MM-DD

### Material received
- ...

### New observations
- Claim:
- Evidence artifact:
- Scope and uncertainty:

### Corrected or retracted
- Previous claim:
- New status:
- Reason:

### Focus changed
- Added:
- Removed:
- Why:

### Experiment registry changes
- Added:
- Completed:
- Failed:
- Gated/deprioritized:

### Next decision gate
- Required evidence:
- Go condition:
- Stop or pivot condition:
```

---

## Decision log — 2026-07-29 (implementation)

Second entry for the same date. The entry above is the review's reading of the
material; this one records what was built in response to it, in the repository at
commit `5f0dd02`.

### Material received

- The July 29 review package, including `09_CRITICAL_FIXES.patch`.
- No new checkpoints, trajectories, or cluster runs. Nothing here is a new
  empirical result, and no claim below is upgraded on the strength of code.

### Code changed

**E00 — critical fixes applied** (`5a8f3fb`). The patch applies cleanly to the
tree at `d1abd53`; seven defects with regression tests. Test discovery goes
184 → 192, which reconciles the review's count with `CURRENT.md`'s: the supplied
tree had 184 and the patch adds 8.

**P0.1 — run artifact contract** (`137b1f8`). `utils/provenance.RunArtifacts`
owns a directory: `resolved_config.json`, `command.txt`, `environment.json`,
`hardware.json`, `git_commit.txt`, `git_diff.patch`,
`parent_checkpoint.sha256`, `data_manifest.json`, `seeds.json`,
`resume_chain.jsonl`, `metrics.jsonl`, `raw_records/`, `checkpoints/`. Written
through `os.replace`, so a scheduler kill leaves the previous contents rather
than a prefix that still parses. Inputs recorded by digest. `environment.json`
carries the installed package set. A run that cannot state its commit, config,
seeds and command refuses to start. Credentials handled by allowlist, with a
test that plants an API key in the environment and asserts no written file
contains it. Training writes `metrics.jsonl` locally as well as to W&B, and
records `global_step_index` and `completed_updates` separately — the earlier runs
reported one number and left the reader to reconcile 38,140 logged steps against
38,141 updates' worth of tokens.

**P0.2 — exact resume** (`137b1f8`). The data cursor is removed rather than
serialized: `training.data.StatelessBlockSampler` makes block order a pure
function of the seed and the global position. Checkpoint schema 2 adds the
scaler, completed updates and tokens, seeds by purpose, the exit-rotation counter,
every random stream, and the launch lineage. Seeds are separated into six named
streams; `model_init` is no longer offset by rank, because two arms meant to
branch from a common parent cannot do so if their constructors consumed different
streams.

**P1 — anchored objective** (`a291ed3`). `objective_version="anchored_v1"` fixes
the full-depth coefficient and normalizes only across the shallow exits.
`shallow_loss_weight=0.0` is exactly a final-only run — no shallow readout is
computed at all. Schedules, frozen-parent distillation, a preservation KL,
top-k distillation that pools the tail rather than truncating it. Every term is
logged separately, and the combined value is logged as `objective`, never as CE.
`gradient_diagnostics()` measures conflict between the full and shallow terms;
off by default.

**P2 — retrofit ladder** (`6e8e5e2`). `src/retrofit.py` and
`experiments/retrofit_parent.py`: seven modes from `frozen_tied_head` to
`full_finetune`, with an audit of exactly what each one made trainable.
`assert_parent_preserved` raises rather than returning a number a caller could
ignore.

**Renames** (`5f0dd02`). `conditional_oracle` → `cross_fitted_probe_policy`;
`horizontal_cascade` → `vertical_no_reuse_oracle_cascade`.

### Tests

```bash
python -m unittest discover -s tests -t .
```

184 (before the patch) → 192 (patch) → 250 (P0) → 306 (P1) → 358 (P2). All
passing on CPU. The count is still written by hand here; CI should generate it.

Two acceptance tests are worth naming because they found real bugs rather than
confirming intent:

- `tests/test_resume.ExactResume` runs the real entry point in real processes —
  100 updates uninterrupted against 50 plus 50 with a process kill in between —
  and compares consumed blocks, scored exits, parameters, optimizer moments, the
  rotation counter, and the next draw from every random stream.
  `OmittingOneComponent` then disables each saved component in turn and asserts
  the comparison notices, so the acceptance test cannot pass with a field that is
  dead code.
- `tests/test_retrofit.FrozenModesPreserveTheParentExactly` checks preservation
  **after** optimizer steps, with a companion test asserting the shallow exits
  did move.

### New observations

These are properties of the implementation, established by deterministic tests.
None is an empirical result about language.

- **Claim:** the legacy objective gives the final endpoint `12/42 = 0.285714` of
  the hard-target coefficient at six exits.
  **Artifact:** `TransformerConfig.exit_weights`,
  `tests/test_objective.TheEndpointIsAnchored`.
  **Scope:** arithmetic. It confirms the review's calculation from the code
  rather than from the paper.

- **Claim:** under the anchored objective with `exits_per_step`, the unbiased
  estimator's gradient averaged over one rotation equals the gradient of scoring
  every shallow exit, to 1.5e-7 relative error; the legacy `fixed_total`
  estimator is off by 4.1e-2.
  **Artifact:** `tests/test_objective.SampledShallowExits`.
  **Scope:** one architecture (12 layers, six exits, budget 2), float32, one
  batch. The rotation's equal coverage was checked exhaustively for every
  `(n_shallow, budget)` pair up to 8.

- **Claim:** in the three frozen retrofit modes the parent's full-depth logits
  are bit-identical after optimizer steps that do move the shallow exits.
  **Artifact:** `tests/test_retrofit`.
  **Scope:** a tiny CPU model. The property is structural — no parameter feeding
  the full-depth path is trainable — so it should hold at scale, but it has not
  been run at scale.

### Corrected or retracted

1. **Previous claim** (`CURRENT.md`, "Established results"): "The sharing tax at
   full depth is real … +0.075 nats."
   **New status:** retracted as a causal claim, retained as an observation about
   one run pair. Two confounds, both now confirmed against the code rather than
   inferred: the arms did not share a backbone initialization (exit construction
   consumed random draws before the global initialization pass — fixed by the
   patch), and the six-exit arm gave its final endpoint 0.2857 of the hard-target
   weight against the other arm's 1.0.
   **Reason:** the difference is real; its attribution to sharing is not
   identified. The defensible statement is the review's: under one seed and this
   specific six-exit recipe, the final endpoint has higher CE than a final-only
   run at matched token exposure.

2. **Previous claim** (`CURRENT.md`): "Depth 10 is 17% cheaper for nothing
   measurable."
   **New status:** retracted. Depth 10 and 12 are close only inside the degraded
   six-exit model; depth 10's trailing CE is 0.0792 nats worse than the
   final-only model's depth-12 trailing CE, and depth 10 was never scored on
   held-out data because evaluation aliased with the exit rotation.

3. **Previous claim:** routed MAC counts reflected probe reuse.
   **New status:** retracted for the pre-patch implementation, which executed
   the probe and then recomputed it from block zero. Every routed compute figure
   from before `5a8f3fb` is optimistic by one probe pass.

4. **Previous claim:** the matched token budget makes the pair a clean sharing
   comparison.
   **New status:** retracted. Token exposure matched; initialization and
   objective did not.

5. **Previous claim:** "38,140 steps generated 2,499,608,576 tokens."
   **New status:** clarified. `2,499,608,576 / 65,536 = 38,141` updates, so
   38,140 is the zero-based final logging index. Both fields are now written.

6. **Previous naming:** `conditional_oracle` described as a ceiling.
   **New status:** corrected to `cross_fitted_probe_policy`. It is the
   out-of-fold performance of one learner from one model class; a better class can
   beat it and make regret against it negative. The +0.051 / +0.008 comparison
   stands, and remains a toy-workload number.

7. **Previous naming:** `horizontal_cascade`.
   **New status:** corrected to `vertical_no_reuse_oracle_cascade`. It uses one
   backbone's endpoints and an outcome-based escalation rule.

### Focus changed

- **Added:** the retrofit ladder as the primary method, with the frozen modes as
  the no-regret lower bound.
- **Added:** the anchored objective as the instrument for any future scratch
  comparison, so a sharing question can be asked without the weight confound.
- **Removed:** nothing. The token-level and RL work stays gated where the review
  put it.

### Experiment registry changes

- **E00 completed.** Patch applied, 192 tests passing, routed block execution
  matched against counters by the patch's own regression test.
- **E01 unblocked in code, still blocked on data.** Every depth can now be
  scored on identical requests with a global reduction, but the collector still
  builds `mixed_difficulty_corpus`. P3 is the next piece of work and E01 cannot
  run before it.
- **E02, E03, E04, E05 now runnable.** `experiments/retrofit_parent.py` plus
  `--objective_version=anchored_v1` is the machinery each of them needed.
- **E06 through E11 unchanged.**

### Next decision gate

**Required evidence:** a real-text trajectory collection (P3), then E01 —
both existing checkpoints scored at every depth on the same untouched held-out
requests, with per-document records.

**Go condition:** at least two shallow endpoints on the quality–cost frontier
after full cost accounting, including the vocabulary head.

**Stop or pivot condition:** if the frozen retrofit of the final-only parent
(E02/E03) shows shallow endpoints no better than the six-exit model's, the
sharing question was never the binding constraint and the interesting result is
about intermediate decodability instead.

### What was not built

Stated so the gap is not mistaken for completion. P3 through P9 are untouched:
real-text trajectory collection, endpoint reanalysis and the clustered bootstrap,
controller schema 3 and deployment parity, the independent horizontal family,
the serving benchmark with continuous batching, token-level routing, and RL. The
cross-cutting CLI contract from §5 of the brief — `--config`, `--run-dir`,
`--dry-run` on every research command — exists only on
`experiments/retrofit_parent.py`.

Consequently, none of the ten conditions in the definition of done is met. The
first two are now *checkable* rather than met: exact resume is verified by test,
and the parent's no-regret property is verified by construction in the frozen
modes — but neither has been exercised on the cluster, and no endpoint has been
scored on real held-out requests.

---

## Decision log — 2026-07-29 (real-text collection)

Third entry for the date. Records the work that followed an audit of whether the
implementation could answer the research questions. It could not, for one uniform
reason and four specific defects, all now closed except where noted.

### Material received

None. No new runs, no new checkpoints. Nothing below is an empirical result.

### The audit that prompted this

Every one of RQ1–RQ7 routes through per-request endpoint quality, and
`collect_depth_trajectories.collect()` built `mixed_difficulty_corpus`
unconditionally. So all seven were blocked on one function, and the phases
delivered before this one (P0–P2) had deferred it. That ordering came from the
review's own priority list and from the coding-agent prompt, but it should have
been checked against the goal before the budget was spent rather than after.

Four further defects would have produced wrong answers even once real text was
available:

1. no instrument for H1 above the frozen retrofit rungs;
2. every interval resampled requests i.i.d., so intervals over correlated
   requests were too narrow — which for a one-sided non-inferiority test biases
   toward *passing*;
3. NLL stored as a per-request mean, making a corpus NLL a mean of means;
4. the controller's live cost semantics could silently differ from the ones it
   was selected under.

### Code changed

`cb42653`. `workloads.real_text_corpus`; schema 2 trajectory records carrying
document identity and NLL sums; clustered `paired_bootstrap` and
`non_inferiority_test` wired through the evaluation; `experiments/no_regret.py`;
retrofit-module MACs in the cost model; `DepthController.cost_metric` enforcement;
`retrofit.restore`; `check_corpus_compatible`.

### Tests

358 → 414. The no-regret failure path is unit-tested rather than demonstrated on
a checkpoint, because an untrained toy model sits at the entropy floor: scaling
every backbone weight by 1.35 moved its NLL by 0.0005 nats, so no fixture of that
kind can express a regression a real margin would catch.

### New observations

- **Claim:** a mean of per-request NLL means is not the corpus NLL, and the gap
  is material. On a two-shape fixture the two differed by 0.0037 nats.
  **Scope:** arithmetic plus one fixture. Stated because the effect this project
  is trying to measure elsewhere is 0.0744 nats, so an aggregation error of this
  size is not negligible relative to the signal.

- **Claim:** clustering changes interval width by a large factor when requests
  repeat within a document. On constructed data with total within-cluster
  correlation, the clustered interval is more than 1.5× wider.
  **Artifact:** `tests/test_realtext.ClusteredIntervals`.

- **Claim:** retrofit modules add 0.06% (rank-32 exit adapter, depth 2) to 0.59%
  (rank-8 LoRA on four projections, depth 12) to endpoint MACs.
  **Scope:** analytical, at `d_model=768`, `V=52000`. Small, and it lands on the
  cheap end of the frontier where the shallow endpoints being justified live.

### Corrected or retracted

Nothing retracted. Three implementation defects found by running the new code:

1. **A LoRA checkpoint could not be reloaded at all.** Wrapping a projection
   renames its weight, so every wrapped projection read as a missing key and
   `retrofit_parent.py` was writing checkpoints nothing could read.
2. **The collector died mid-collection** on a shape exceeding the model's
   context, after completing the shapes that fit — leaving output a resume would
   treat as complete.
3. **An out-of-vocabulary token** surfaced from inside the embedding lookup.

### Experiment registry changes

- **E01 unblocked.** Every depth can be scored on identical real held-out
  requests with document-clustered intervals.
- **E02, E03, E04, E05, E06 runnable.** The pipeline was verified end to end on a
  fixture corpus: collection → controller → evaluation → no-regret test.
- **E07 (independent horizontal family) is now the largest blocked item**, and it
  blocks the paper's central claim. No independent models, no adapter for a
  public family.
- **E08 (serving benchmark) blocked on code** for continuous batching, depth
  queues, goodput, energy, and counter/profiler parity.
- **E10 gated and its gate is not implemented**: the token-level
  outcome-oracle-gain diagnostic does not exist.

### Research question status

| | answerable now | blocked on |
|---|---|---|
| RQ1 no-regret endpoint | yes | — |
| RQ2 endpoints vs independent models | no | independent family (E07) |
| RQ3 heterogeneity beyond static mixture | yes | — |
| RQ4 can a controller read it | yes | — |
| RQ5 systems realization | K/V only | serving benchmark (E08) |
| RQ6 irreducible complementarity | no | independent family (E07) |
| RQ7 request vs token level | no | gating diagnostic (E10) |

H1, H2 and H3 have instruments. H4, H5 and H6 do not.

### Next decision gate

**Required evidence:** RQ1, RQ3 and RQ4 answered on FineWeb-Edu with the existing
checkpoints, plus a frozen-adapter retrofit of `vr-noexits`.

**Go condition:** at least two shallow endpoints on the frontier after full cost
accounting, and a positive `probe-policy gain` whose clustered interval excludes
zero.

**Stop or pivot condition:** `probe-policy gain` near zero. Request-level routing
then has no case, and E07/E08 should not be built for it.

---

## Decision log — 2026-07-29 (scope cut)

Fourth entry for the date. The previous three added capability; this one removes
scope. Nothing was deleted from the record — the cut experiments are listed above
with their reasons.

### Why

The project had accumulated twelve experiments, four documents, two training
objectives, three distinct meanings of "K/V cache", and a 64-GPU-hour cluster plan
approved earlier the same day. None of it was wrong and none of it was navigable.
The registry had become a list of things that could be done rather than a plan.

Volume was the failure. Eleven half-built experiments support no claim; three
complete ones support three.

### Research questions, narrowed

Of the seven questions in this file, three are retained as claims the paper makes:
no-regret endpoints, useful tiers, and partial substitution against a matched
family. RQ5 (systems realization) is answered only for K/V memory and reported as
unmeasured elsewhere. RQ7 (token versus request level) is closed to future work.
RQ6 (irreducible complementarity) needs heterogeneous specialists rather than one
same-family suite, and is out of scope.

Hypotheses H1, H2 and H3 have instruments and experiments. H4 has an instrument
and an experiment but may be unreportable if the matched frontier is too narrow to
divide by. H5 and H6 are dropped.

### The load-bearing argument for the largest cut

The four Pile + NeoX scratch arms were approved, scripted, and then cut within the
same day. The argument that removed them:

> The sharing tax is not a claim in the thesis. The thesis is *retrofit a trained
> parent*. If the parent is frozen, no sharing tax exists — that is the content of
> the no-regret framing. Measuring the tax would price a method the project is not
> proposing.

The contingency is explicit rather than hopeful: a frozen retrofit gives only what
is linearly decodable from the parent's intermediate states, which may be too
weak. A1 measures exactly that, for free, before any cluster time is spent. If A1
fails, trained exits become necessary, sharing becomes a real cost, and the Pile
arms return — with `jobs/controlled_arms.sh` and `jobs/prepare_pile.sh` already
written and validated.

### Code changed

None. This entry is scope and documentation only. Every cut experiment's machinery
remains in the repository and tested: the anchored objective, the full retrofit
ladder including LoRA, the controlled-arm job scripts, and the Pile preparation
script are all present and green. Cutting an experiment did not mean deleting the
means to run it.

### Added

`START_HERE.md`. One question, three commands in order, each with its kill
condition, plus the four things most likely to confuse a later reader: the three
meanings of "K/V cache", the two objectives and which to use, the retraction of
the two runs on disk, and why cross-tokenizer quality must be bits per byte.

The gap it fills is that `DESCRIPTION.md`, `CURRENT.md` and `MIGRATIONS.md` are
all *records*. None of them said what to run.

### Next decision gate

**Required evidence:** A1 — a frozen retrofit of `vr-noexits/final.pt` with its
exits trained, endpoints scored on real held-out text.

**Go condition:** at least two shallow endpoints on the quality–cost frontier
after full cost accounting, with the parent's logits bit-identical.

**Stop or pivot condition:** shallow endpoints no better than the deepest one is
worth using, or no better than chance. The first means routing has nothing to
choose between; the second means the parent's intermediate states are not
decodable and the retrofit framing fails. Either sends the project back to trained
exits and reinstates the Pile arms.
