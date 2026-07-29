# 2026-07-29

State of the experiment after the first matched-budget training pair. Numbers
here were read from the W&B run history via the API, not from the rendered
report, and every claim below names where it came from.

**How to update:** replace the date above and rewrite the sections in place when
a new run lands. Move anything that a later run contradicts into
*Corrected/retracted* rather than deleting it — a result that was wrong once is
worth being able to find again.

> **Read this first.** An external review on 2026-07-29 retracted the two
> headline results below. They have been moved to *Retracted* and are no longer
> in *Established*. The numbers are unchanged and were not misread; what fails is
> their attribution. See `DESCRIPTION.md` for the full decision log.

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

This is the one result in this file that survives the review, and the review is
right that it is an implementation invariant rather than a research finding: it
shows the depth cap reaches the allocator. It says nothing about latency,
throughput or energy, and it is not novel on its own.

---

## Retracted

Neither number below was misread. Both are real properties of the two runs. What
fails is what they were taken to mean.

### The +0.075 nats is not a sharing tax

**Was:** "The sharing tax at full depth is real."

**Now:** an observation about one run pair, not a causal estimate. Two confounds,
each confirmed against the code rather than inferred:

1. **The arms did not share a backbone.** Constructing exit modules consumed
   random draws *before* the global initialization pass, so six exits and one exit
   under the same seed produced different embeddings and blocks — measured
   maximum differences around 0.0955 on the embedding and 0.1009 on block 0's
   query projection. Fixed in `5a8f3fb`; the two runs on disk predate the fix.
2. **The endpoint was down-weighted.** The legacy objective normalizes across all
   exits, so at depths 2/4/6/8/10/12 the final endpoint carried
   `12/42 = 0.2857` of the hard-target coefficient against the final-only arm's
   `1.0`, and each shallow exit also carried a distillation term at 0.5.

A degraded endpoint is the expected consequence of dividing its coefficient by
3.5. The pair cannot separate that from gradient interference, from the different
initialization, or from actual capacity sharing.

**What is still defensible:** under one seed and this specific six-exit,
linearly-normalized, self-distilled scratch recipe, the final endpoint has higher
CE than a final-only run at matched token exposure, and the multi-exit run costs
11.68% more wall clock. That motivates the anchored objective and the retrofit
ladder. It does not measure what sharing costs.

**What would establish it:** two arms from the same serialized parent
initialization, same data order, under `--objective_version=anchored_v1` with
`shallow_loss_weight` at 0 and at a nonzero value. The anchored objective's
`shallow_loss_weight=0.0` is exactly a final-only run, which is what makes the
control arm a control.

The numbers, retained:

| | depth-12 CE | perplexity |
|---|---|---|
| `vr-noexits` (final-only recipe) | 3.2024 eval / 3.1782 train | 24.59 |
| `vr-exits` (six-exit recipe) | 3.2768 eval / 3.2560 train | 26.49 |
| **difference** | **+0.0744 eval / +0.0778 train** | **+7.7%** |

Two estimates by different routes — held-out evaluation, and a 40-point trailing
mean of training CE — agreeing to within 0.003 nats against ±0.05 scatter. Both
are also subject to the reduction bug below.

### Depth 10 is not free

**Was:** "A fixed depth-10 endpoint is 17% cheaper for nothing measurable."

**Now:** retracted. Three reasons:

1. Depth 10 and depth 12 are close *inside the six-exit model*, which is the
   model whose endpoint is degraded. It is not evidence that two blocks are
   redundant in general.
2. Depth 10 was trained as an exit, so the flatness may reflect the objective
   rather than natural redundancy.
3. The comparison that matters goes the other way: the six-exit depth-10 CE is
   **0.0792 nats worse** than the final-only model's depth-12 CE
   (3.2574 − 3.1782), an 8.24% perplexity ratio.

And depth 10 was never scored on held-out data at all — see the aliasing bug
below.

The curve, retained (`vr-exits`, trailing-mean training CE):

| depth | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| CE | 4.0547 | 3.4855 | 3.3465 | 3.2997 | 3.2574 | 3.2560 |
| Δ from previous | — | −0.569 | −0.139 | −0.047 | −0.042 | **−0.001** |

The token-level threshold sweep pointed the same way: `vr-exits` routed at mean
depth 9.94 (threshold 0.2, 17.1% compute saved) reached accuracy 0.3470, against
0.3545 for `vr-noexits` at full depth.

**What would establish it:** a paired non-inferiority test on held-out data
between depth-10 and depth-12 readouts within one model, a post-hoc depth-10
readout on the final-only parent, and an independent model at comparable measured
cost.

### The held-out CEs describe a shard, not the split

**Was:** eval CE 3.2024 and 3.2768, described as held-out numbers.

**Now:** they are rank zero's validation shard. The validation sampler assigns
disjoint data per rank and the evaluator returned local losses without an
all-reduce, while only rank zero logged. If both runs used the same world size,
sampler and order, the comparison may still be paired over the same subset — but
it is a smaller and misdescribed sample. Fixed in `5a8f3fb`; both checkpoints need
re-scoring over the full split.

### Routed compute figures were optimistic by one probe pass

Any routed MAC or realized-compute number produced before `5a8f3fb` understates
cost. `generate_routed()` ran the shallow probe, discarded it, and recomputed
those layers from block zero inside the selected tier, while the counters charged
the probe as reused.

### The step count was an index

38,140 is the zero-based final logging index. `2,499,608,576 / 65,536 = 38,141`
completed updates. Both fields are now written separately.

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

Nothing below is blocked on missing code any more, except where it says so.

- **Nothing about sharing.** See *Retracted*. A causal estimate needs two arms
  from one serialized parent initialization under `anchored_v1`, with
  `shallow_loss_weight` at 0 and at a nonzero value. **Runnable now.**
- **No endpoint has been scored on real held-out text.** The collector can do it
  (`--corpus real_text`) and the pipeline runs end to end, but it has not been
  pointed at FineWeb-Edu with a real checkpoint. **Runnable now.**
- **`probe-policy gain` is unmeasured on real text.** This is the go/no-go
  number. On the toy workload the outcome oracle showed +0.051 of headroom while
  the cross-fitted probe policy attained +0.008, so 85% of the apparent gain
  required knowing the answer. Both figures are toy-workload numbers.
  **Runnable now.**
- **No retrofit has been trained.** `retrofit_parent.py` builds one and proves
  the parent is preserved to the bit; nothing has trained its exits yet.
  **Runnable now.**
- **Sharing tax at shallow tiers.** Needs independently trained models at
  matched cost. **Blocked on code** — no independent family and no adapter for
  one (Pythia or similar). This is the single largest remaining gap, and it
  blocks the paper's central claim.
- **Latency, throughput, memory and energy on serving hardware.**
  `benchmark_latency` measures TTFT and TPOT on a laptop toy model and the K/V
  audit is exact. **Blocked on code** for continuous batching, depth-homogeneous
  queues, SLO goodput, energy, and counter/profiler parity. TTFT also still comes
  from a separate invocation rather than instrumented inside one.
- **Whether token-level routing adds anything.** **Blocked on code**: the gating
  diagnostic — token-level outcome-oracle gain beyond the request-level cap — is
  not implemented, and by the roadmap nothing token-level should start until it
  is.
- **The K/V propagation strategy has never been trained.**
  `learned_kv_propagation` defaults to `False` and neither run overrode it.
  Gated behind request-level routing showing headroom.
- **Retrofit at scale.** The frozen modes preserve the parent bit-identically in
  tests on a tiny CPU model. The property is structural, so it should hold at
  scale, but it has not been run there.

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

The pipeline runs end to end on real text. Verified on a fixture corpus:
collection → controller → evaluation → no-regret test, with intervals resampling
documents. What is missing is real data through it.

```bash
# 1. What is decodable from a final-only parent's intermediate states, at zero
#    sharing cost. The parent's endpoint is preserved to the bit, so this is a
#    clean measurement and needs no new training.
python -m experiments.retrofit_parent \
    --checkpoint checkpoints/vr-noexits/final.pt \
    --run-dir runs/retrofit-adapter \
    --mode frozen_exit_adapter --exit_adapter_rank 32 --exit_every 2

# 2. Score every depth of both existing checkpoints on the same real held-out
#    requests. This replaces the shard-level eval CEs above, and gives depths
#    6, 8 and 10 their first held-out numbers.
python -m experiments.collect_depth_trajectories \
    --corpus real_text --data data/val.bin --eos_id 50256 \
    --checkpoint checkpoints/vr-exits/final.pt \
    --n_requests 4096 --out results/traj-exits

# 3. The go/no-go. Read probe-policy gain.
python -m experiments.train_depth_controller --trajectories results/traj-exits \
    --out results/controller --seeds 0 1 2
python -m experiments.evaluate_vertical_routing --trajectories results/traj-exits \
    --controller results/controller --controller_seed 0 --out results/evaluation
```

`--eos_id 50256` is GPT-2's end-of-text token. Omitting it makes every interval
unclustered and too narrow, and the corpus metadata will say so.

If `probe-policy gain` is near zero on real text, request-level routing has no
case and no amount of controller work will create one. That is the result worth
having early, and it now costs one job rather than a rewrite.

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

# The exit-weight arithmetic behind the retraction above.
python -c "from src.config import TransformerConfig as C; \
  print([round(w,6) for w in C(n_layers=12, exit_every=2).exit_weights])"

python -m unittest discover -s tests -t .   # 414 tests
```
