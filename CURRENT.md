# 2026-07-29

State of the experiment after the first matched-budget training pair. Numbers
here were read from the W&B run history via the API, not from the rendered
report, and every claim below names where it came from.

**How to update:** replace the date above and rewrite the sections in place when
a new run lands. Move anything that a later run contradicts into
*Corrected/retracted* rather than deleting it — a result that was wrong once is
worth being able to find again.

---

## Runs

Both arms completed. The budget matched exactly, which is what makes the
comparison legitimate at all.

| | `vr-noexits` | `vr-exits` |
|---|---|---|
| state | finished | finished |
| steps | 38,140 | 38,140 |
| tokens | 2,499,608,576 | 2,499,608,576 |
| runtime | 65,755 s (18.3 h) | 73,437 s (20.4 h) |
| throughput | 38,023 tok/s | 34,045 tok/s |
| exits | 1 (final layer) | 6 (depths 2/4/6/8/10/12) |
| `self_distill_weight` | 0.0 | 0.5 |
| `exits_per_step` | — | 2 |

Architecture: 12 layers, `d_model` 768, 12 heads, `seq_len` 1024, GPT-2
tokenizer, 65,536 tokens/step. Corpus: FineWeb-Edu, 4,095,609,674 train tokens
and 2,098,344 validation tokens, so neither run wrapped the corpus.

`noexits` is not a separate code path: `exit_every = n_layers` leaves exactly one
exit, on the final layer, which is an ordinary language model.

---

## Established results

### The sharing tax at full depth is real

Training with six exits cost the final layer roughly **0.075 nats**.

| | depth-12 CE | perplexity |
|---|---|---|
| `vr-noexits` (independently trained) | 3.2024 eval / 3.1782 train | 24.59 |
| `vr-exits` (shared backbone) | 3.2768 eval / 3.2560 train | 26.49 |
| **tax** | **+0.0744 eval / +0.0778 train** | **+7.7%** |

Two estimates computed by different routes — held-out evaluation, and a
40-point trailing mean of training CE — agree to within 0.003 nats. The scatter
on those trailing windows is ±0.05. Train figures use the same trailing mean on
both arms.

This is a point estimate from one seed. It is **not** the predeclared
non-inferiority test, which needs the paired per-request bootstrap from
`jobs/sharing_tax.sh`.

### The depth curve is flat above depth 8

`vr-exits`, trailing-mean training CE:

| depth | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| CE | 4.0547 | 3.4855 | 3.3465 | 3.2997 | 3.2574 | 3.2560 |
| Δ from previous | — | −0.569 | −0.139 | −0.047 | −0.042 | **−0.001** |

The last two blocks buy 0.0014 nats, well inside the ±0.05 scatter. Depths 10
and 12 are the same model for practical purposes.

**This is the finding that most affects the research question.** A fixed
depth-10 endpoint is a strong baseline needing no controller at all: 17% cheaper
for nothing measurable. The router's gain has to be measured against that, not
against depth 12. And the shared backbone at depth 10 (3.2574) is still 0.079
nats behind the independently trained model at depth 12 (3.1782).

The token-level threshold sweep points the same way: `vr-exits` routed at mean
depth 9.94 (threshold 0.2, 17.1% compute saved) reaches accuracy 0.3470, against
0.3545 for `vr-noexits` at full depth.

### K/V memory falls exactly as proved

`04_math_foundations.md` §7 proves cache memory under a request-level depth cap
falls by exactly `1 − d/L`, and lists the outstanding obligation as *"verify
implementation does not allocate upper K/V."* Now verified rather than asserted:
across 12 configurations spanning every depth and two batch sizes, measured
cache bytes equal the analytical prediction **to the byte**, and the realized
saving equals `1 − d/L` to twelve decimal places.

The measurement is read from the cache itself and the prediction from
`AnalyticalCostModel`; the two reach the comparison by separate code paths.
Reported by `experiments/benchmark_latency.py` under *"K/V memory: does the
depth cap reach the allocator?"*.

---

## Corrected while reading these runs

### Half the exits were never evaluated

Depths **6, 8 and 10 were never scored on held-out data**, across all 76
evaluations of a 38,140-step run.

The exit rotation is deterministic in the global step: with five non-final exits
and `exits_per_step=2`, the choice depends only on `step % 5`. `eval_every=500`
is divisible by 5, so every evaluation landed on the same rotation position and
scored the same two shallow exits plus the final one.

The rotation itself is fine — consecutive training steps do cover every exit.
Applying it to evaluation was the error, since `evaluate()` runs under
`no_grad` and the memory reason for sampling does not exist there. Fixed in
`cb1eb57`: `model.score_all_exits()` suspends it and `evaluate()` holds it.

No checkpoint is affected. Only what was logged about them was incomplete —
which is why the depth curve above comes from training CE.

### K/V byte accounting was off by one position

Predicted cache bytes assumed `prompt_len + generated` cached positions. The
last token emitted is never fed back, so nothing attends to it and its keys and
values are never written. Measured across four request shapes, cached positions
are exactly `prompt_len + generated − 1`.

Fixed in `09d191a`, in both `benchmark_latency` and
`collect_depth_trajectories`. The latter mattered more: `kv_bytes` is a
selectable routing cost metric, so the error entered controller training
whenever it was chosen.

Found by the K/V audit on its first real run, which is the argument for having
built it.

---

## Not established

- **The sharing tax has no confidence interval.** One seed per arm, point
  estimates only. `jobs/sharing_tax.sh` produces the paired per-request
  bootstrap.
- **`learnable gain` is unmeasured on real text.** This is the go/no-go number.
  On the toy workload the plain oracle showed +0.051 of headroom while the
  reachable ceiling showed +0.008, so 85% of the apparent gain was unreachable
  by construction. Judging the controller against the plain oracle reports a
  near-optimal policy as a failure.
- **Sharing tax at shallow tiers.** `vr-noexits` gives one independent model, at
  depth 12. Depths 2/4/6/8/10 have no independent counterpart, so they will
  appear under `sharing_tax_unmatched_tiers`. Measuring them needs the
  horizontal family.
- **The K/V propagation strategy has never been trained.**
  `learned_kv_propagation` defaults to `False` and neither run overrode it, so
  the propagator was inert. It is the token-level (Phase 7) approximation, and
  §7 notes it materializes all `L` layers and therefore saves no memory. A
  separate run is required; `experiments/exposure.py` has the arms for it.
- **Latency on serving hardware.** `benchmark_latency` has only been run on a
  toy model on a laptop. It establishes that the measurement is wired up, not
  what routing is worth.

---

## Known gaps in the record

- **W&B logged an incomplete config.** Only `n_layers`, `d_model`, `n_exits`,
  `exit_criterion` and `self_distill_weight` reached the run config. The brief
  requires the complete model and routing configuration in every experiment
  output. The local checkpoints carry it; W&B alone would not reproduce a run.
- **Training hardware was not recorded per run.** Throughput differs by 10%
  between the arms and the partitions differed, but the run record does not say
  which. Quality comparison is unaffected — the budget is matched in tokens —
  but any latency comparison must be on identical hardware.

---

## Next

```bash
sbatch --job-name=vr-tax jobs/sharing_tax.sh \
  checkpoints/vr-exits/final.pt checkpoints/vr-noexits/final.pt
```

Collects trajectories for both checkpoints under one seed, builds the horizontal
manifest, fits the controller over three seeds, and evaluates. It returns the
paired interval around the +0.074 above, and `learnable gain`.

Read `learnable gain`, not `oracle − best fixed`. If it is around zero on real
text, request-level routing has no case and no amount of controller work will
create one.

---

## Reproducing the numbers above

```bash
# Run history, including per-exit series (needs a wandb login).
python - <<'PY'
import wandb
api = wandb.Api()
for name in ("vr-noexits", "vr-exits"):
    r = api.run(f"iparramartin/effortless-vertical-routing/{name}")
    print(name, {k: v for k, v in r.summary.items() if not k.startswith("_")})
PY

# K/V audit, which runs as part of the latency benchmark.
python -m experiments.benchmark_latency --out results/latency --device=cpu

python -m unittest discover -s tests -t .   # 184 tests
```
