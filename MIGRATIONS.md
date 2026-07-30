# Migrations

Schema and default changes, newest first. Every entry says what changed, what
breaks, and how to reproduce the old behaviour, because a run that cannot be
reproduced after a refactor has stopped being evidence.

---

## Training objective: `legacy_normalized` → `anchored_v1`

**Status:** both available. `legacy_normalized` remains the default.

### Why

The legacy objective normalizes *every* exit weight to sum to one. With exits at
depths 2/4/6/8/10/12 and `exit_loss_weighting="linear"`, the coefficients are:

| depth | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| weight | 0.047619 | 0.095238 | 0.142857 | 0.190476 | 0.238095 | **0.285714** |

The final endpoint receives `12/42 = 0.2857` of the hard-target coefficient.
The other 0.7143 goes to shallow exits, each of which also carries a
distillation term at weight 0.5.

So the `vr-exits` and `vr-noexits` arms did not compare *sharing against no
sharing*. They compared a conventional language-model objective against a
multitask objective in which the endpoint under test had been down-weighted by a
factor of 3.5. The +0.0744 nats those runs reported is a real number about that
pair; it is not a measurement of what parameter sharing costs, and it should not
be quoted as one.

### What `anchored_v1` does

```text
L = full_loss_weight * CE_full
  + alpha(t) * sum_{d<L} w_d * (CE_d + distill_weight * T^2 * KL_d)
  + preservation_weight * KL(parent_full || this_full)
```

`full_loss_weight` defaults to 1.0 and is never renormalized. The `w_d`
normalize over the **shallow exits only**. Adding an exit therefore cannot take
coefficient away from the endpoint the no-regret claim is about, and
`shallow_loss_weight` is one number that can be swept.

### Reproducing an existing run

Nothing to do. `objective_version` defaults to `"legacy_normalized"`, and every
field introduced with the anchored objective is ignored under it — including
`preservation_weight`, which a legacy run will not apply even if it is set. The
existing `jobs/train.sh` invocations are unchanged.

### Translating a configuration

| legacy field | anchored equivalent |
|---|---|
| `exit_loss_weighting="linear"` | `shallow_weighting="linear"` |
| `exit_loss_weighting="uniform"` | `shallow_weighting="uniform"` |
| `exit_loss_weighting="final_only"` | `shallow_loss_weight=0.0` |
| `self_distill_weight` | `distill_weight` |
| `self_distill_temperature` | `distill_temperature` |

There is no anchored setting that reproduces the legacy *coefficients*, and that
is deliberate: the legacy split of the endpoint's weight is the thing being
removed. The nearest comparison at matched shallow emphasis is
`shallow_loss_weight = sum(legacy shallow weights) = 0.714286` at six exits.

`shallow_loss_weight=0.0` is exactly a final-only run — not approximately. No
shallow readout is computed, so it matches a single-exit model in gradient, in
memory, and in wall clock, which is what makes it a usable control arm.

### Sampling estimator

Under `exits_per_step`, the anchored objective defaults to
`shallow_estimator="unbiased"`: sampled exits are scaled by
`n_shallow / n_sampled`. The rotation covers each shallow exit exactly `budget`
times in `n_shallow` consecutive steps, so the gradient averaged over one
rotation equals that of scoring every exit — verified to 1e-7 relative error in
`tests/test_objective.py`.

The legacy behaviour, `fixed_total`, redistributes the whole shallow weight
across whichever exits were picked. It holds the per-step loss scale constant,
which is easier on gradient clipping, but it is biased: measured 4% relative
gradient error over a rotation at six exits with a budget of two. Legacy runs
keep it regardless of what `shallow_estimator` says, so their reproduction does
not depend on this default.

### When the default should flip

The original condition was "once a matched final-only arm has been compared
against it on the cluster". That comparison was the four Pile scratch arms, **cut
on 2026-07-29**, so the condition can no longer be met as written.

The replacement: `anchored_v1` should become the default once experiment A1 has
trained a retrofit's exits with it end to end and the parent's endpoint has been
verified unchanged. That is a weaker check — it confirms the objective trains
something usable, not that it beats the legacy weighting on a matched pair — and
it is the strongest check the narrowed plan supports.

Until then every run should pass `--objective_version=anchored_v1` explicitly, so
its record says which objective produced it.

---

## Removed: `experiments.build_manifest`

**Status:** deleted 2026-07-29. Recoverable from git history.

It built a horizontal manifest out of the **vertical** model's own trajectories,
labelling one backbone's depth endpoints as though they were independent models.
That is the substitution the review prohibited outright: no vertical endpoint data
may be reused as independent-model data. It existed only because no model family
was available.

**Replacement:** `experiments.horizontal_family`, which scores real Pythia
checkpoints and writes a manifest with per-shape costs, a stated quality unit and
direction, per-request content digests, the revision and token budget of each
checkpoint, and family metadata recording what is held constant across tiers.

Manifests written by the old module are still *readable* — `load_manifest` accepts
entries with a scalar `cost` and warns that a matched-cost comparison built on one
is matched only on average. They will now fail
`check_units_comparable`, which refuses a manifest that does not state its quality
unit. That refusal is the point: a manifest of vertical endpoints compared against
vertical endpoints in an unstated unit is exactly the meaningless comparison the
guard exists to stop.

`jobs/sharing_tax.sh` was deleted with it — it collected trajectories from real
checkpoints and pointed them at the synthetic corpus, so its confidence intervals
were intervals about a token pattern.

---

## Checkpoints: schema 1 → schema 2

**Status:** schema 2 is written. Schema 1 loads, with a printed warning.

### Why

Schema 1 held weights, optimizer state and a step count. That is enough to
continue training, but not enough to continue *the same* training: the data
cursor, the exit rotation and every random stream restarted from the top. A
resumed run therefore repeated data the model had already seen, trained a
different exit schedule, and reported a token budget it had not consumed. None
of it is visible in the loss curve.

### What is added

`schema_version`, `scaler`, `completed_updates`, `completed_tokens`, `seeds`,
`step_counter` (the exit-rotation position — a non-persistent buffer, so
`state_dict` never held it), `random_states` (Python, NumPy, CPU torch, and every
CUDA device), and `lineage` (one entry per launch).

The data cursor is deliberately **not** stored.
`training.data.StatelessBlockSampler` derives block order from the seed and the
global position, so resuming means constructing it with a different start. There
is no cursor that can drift out of agreement with the update count.

### Loading a schema 1 checkpoint

It works. `completed_updates` falls back to `step`, and the missing random and
rotation state produce:

```text
warning: <path> predates checkpoint schema 2 and carries no random or
rotation state. Training continues; it is not the same run.
```

Treat any run continued that way as a new run for reporting purposes.

---

## Data loader: epoch-cycled → unbounded

**Status:** done. `training.data.build_dataloader` returns an unbounded loader.

### What changed

`DistributedSampler` plus a cycling wrapper is replaced by
`StatelessBlockSampler`. The loader no longer ends, so there is no epoch to set
and `training.train._infinite` is gone. `len(sampler)` raises `TypeError` rather
than returning a plausible number, since an unbounded stream has no length and a
caller that got one would treat a single epoch as the whole run.

Callers pass `start_micro_batch=completed_updates * grad_accum_steps` to resume.

### One consequence worth knowing

The loader now owns an explicit `torch.Generator`. Constructing a DataLoader
iterator draws one value from the *global* generator to seed its workers; with
that draw in the global stream, restoring a random state and then creating the
iterator shifted every subsequent dropout mask by one, and a resumed run
diverged from the run it restored. The explicit generator removes the coupling.

---

## Seeds: one value → six named streams

**Status:** done, backward compatible.

`TrainConfig.seed` still specifies a run on its own; the six streams
(`model_init`, `data_order`, `dropout`, `exit_sampling`, `controller`,
`benchmark`) are derived from it and are individually settable through
`--model_init_seed` and friends.

**One behaviour changed.** Initialization used to be seeded as `seed + rank`. It
is now seeded as `model_init` with no rank offset. Two arms meant to branch from
a common parent cannot do so if their constructors consumed different random
streams, and a rank offset guaranteed they did. DDP broadcasts rank zero's
weights anyway, so no distributed run is affected in its outcome — but rank
zero's own initialization changes, so a *single-process* run with a given seed no
longer reproduces bit-for-bit against the old code. Data and dropout streams are
still offset by rank, on purpose and now on the record.

`exit_sampling` is recorded and unused: the rotation is a deterministic function
of the step counter, which is how every rank agrees without communicating. The
stream exists so that a future estimator which samples exits cannot borrow the
data or dropout stream to do it.
