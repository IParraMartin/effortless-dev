# 2026-07-29

This is the research diary: objective, hypotheses, experiment registry, and a
dated decision log. `CURRENT.md` holds the state of the runs; `MIGRATIONS.md`
holds schema and default changes. The newest decision-log entry is at the bottom.

## Project identity

**Working title:** *Elastic Without Regret: Can Vertical Routing Replace a Portfolio of Language Models?*

**Project name:** Effortless Vertical Routing

**Current phase:** After the first matched-token training pair; before a causal sharing-tax estimate or a production-valid routing benchmark.

## Objective of the paper

The paper studies whether a single language model with multiple usable depth endpoints can replace some of the work normally handled by routing requests among separately trained models.

The central comparison is:

- **Horizontal routing:** select one model from a portfolio of independently trained models.
- **Vertical routing:** select how much depth to execute inside one shared backbone.
- **Vertical cascade:** begin at a shallow endpoint and reuse the already-computed prefix when escalating deeper.
- **Hybrid routing:** choose a model or specialist horizontally, then choose depth vertically within it.

The primary objective is to determine the region in which vertical routing is a practical substitute for horizontal routing. The paper should not claim that depth replaces model diversity universally. It should estimate the boundary: adjacent, same-family, general-purpose capacity tiers may be substitutable; heterogeneous specialists, modalities, tools, training data, or alignment policies may retain complementary errors that depth alone cannot reproduce.

## Current thesis

> A strong full-depth parent can be retrofitted with shallow endpoints and a request-level value-of-depth controller so that the resulting elastic model preserves the parent’s full-depth quality, exposes several useful quality–cost operating points, and recovers a measurable fraction of the quality–cost advantage of routing among independent same-family models. Reusable probe computation and depth-capped K/V caches can provide systems benefits that horizontal cascades cannot obtain.

This is a hypothesis, not yet an established result.

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

### 6. Optional token-level extension

Token-level depth adaptation and learned K/V propagation remain a later extension. They are not the headline contribution until request-level routing is correct, useful, and measurable on serving hardware.

## Research questions

1. Can shallow endpoints be added without degrading the parent model’s full-depth quality?
2. At equal measured cost, how close are shared depth endpoints to independently trained same-family models?
3. Is there meaningful per-request heterogeneity in the depth required for good predictions after controlling for a strong static mixture?
4. Can a controller infer that heterogeneity from reusable shallow hidden states on untouched requests and distribution shifts?
5. How much of theoretical MAC and K/V savings becomes actual TTFT, TPOT, throughput, memory, and energy improvement?
6. When does horizontal model complementarity remain irreducible?
7. Does request-level depth selection provide most of the gain, or is token-level adaptation necessary?

## Primary hypotheses

- **H1 — no-regret endpoint:** the retrofitted model’s final endpoint is non-inferior to its frozen parent within a predeclared margin.
- **H2 — useful tiers:** at least two shallow endpoints lie on the empirical quality–cost frontier after including all readout and controller costs.
- **H3 — learnable adaptivity:** a cross-fitted request router beats the best exact static mixture at matched cost on an untouched reporting split.
- **H4 — partial substitution:** the vertical substitution ratio is materially above zero for adjacent same-family model tiers.
- **H5 — systems realization:** measured serving benefits preserve a meaningful fraction of analytical compute savings at relevant batch sizes and arrival rates.
- **H6 — boundary:** heterogeneous specialists retain a larger complementarity gap than adjacent same-family general-purpose models.

## Evidence ledger

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

| ID | Experiment | Status | Success criterion |
|---|---|---|---|
| E00 | Apply critical correctness patch | Ready | Clean apply; all tests pass; routed block execution equals counters. |
| E01 | Re-score existing checkpoints on real held-out data | Blocked on checkpoints | Every depth, same documents, global DDP reduction, per-document records. |
| E02 | Post-hoc readouts on final-only checkpoint | Planned | Determines whether depth saturation exists without joint exit training. |
| E03 | Common-parent frozen-exit retrofit | Planned | Useful shallow tiers with exactly unchanged parent output. |
| E04 | Anchored objective sweep | Planned | Improves shallow endpoints while meeting full-depth non-inferiority. |
| E05 | Selective unfreeze / LoRA retrofit | Planned | Beats frozen exits at equal adaptation cost without parent regression. |
| E06 | Real-text trajectory and controller study | Planned | Cross-fitted router beats exact static mixture on untouched requests. |
| E07 | Independent horizontal family | Planned | Cost-matched vertical substitution ratio with confidence bands. |
| E08 | Serving benchmark | Planned | Measured TTFT/TPOT/throughput/KV/energy frontier on identical hardware. |
| E09 | Distribution shift and safety | Planned | Controller remains calibrated or safely escalates under shift. |
| E10 | Token-level routing and K/V propagation | Gated | Only proceed if request routing has residual learnable headroom. |
| E11 | RL or constrained bandit controller | Gated | Only proceed if supervised controller leaves reward-specific headroom. |

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
