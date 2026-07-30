# Experiments

## The request-level pipeline

Four scripts, run in order. Each writes a `run.json` carrying the git commit and
dirty flag, platform, torch version, device, every seed by name, the full
configuration, checkpoint digests, and a schema version — in the same file as
the results, because two files drift apart.

```bash
python -m experiments.collect_depth_trajectories --out results/trajectories
python -m experiments.train_depth_controller --trajectories results/trajectories \
    --out results/controller
python -m experiments.evaluate_vertical_routing --trajectories results/trajectories \
    --controller results/controller --out results/evaluation
python -m experiments.benchmark_latency --out results/latency --device cpu
```

### 1. Trajectory collection

For every held-out request and every candidate depth: probe features, endpoint
NLL and accuracy, agreement with full depth, free-running reward, and every
cost component. Quality and cost are stored **separately** so that λ can change
at training and evaluation time without re-collecting.

Three label types are recorded under names that say which is which, because
substituting one for another is the most effective way to overstate early
exiting:

| Label | What it answers |
|---|---|
| `teacher_forced_*` | reference continuation scored in one parallel pass — cheap, low variance, never lets the endpoint's own mistakes into its context |
| `free_running_*` | generated at the endpoint, then scored — the one that supports a serving claim, and the one that degrades |
| `*_agreement` | relative to full depth, not to the truth |

The schema also records `cache_semantics: "exact"`. For request-level routing
the "exact cache" qualifier that matters so much elsewhere is *vacuous*: every
executed layer sees true keys and values. A later reader should not have to
reconstruct that argument.

### 2. Controller training

Backbone frozen. Splits are made **by source, never by request**, so two
requests derived from one block cannot land on opposite sides and let the
controller score well by recognizing the block. The held-out half is split
again into calibration and reporting, so the ordinal head's cutoff is not tuned
on the numbers it is then judged by.

Three targets are supported: earliest sufficient depth, per-tier utility, and
continuation gain. Every fit is repeated over seeds and every seed is reported.

### 3. Evaluation

Every system on the same requests, every difference computed per request before
averaging, and paired bootstrap intervals throughout.

The baseline that matters most and is easiest to omit: **the best static
randomized mixture at matched average cost**. A weighted coin flip between two
fixed depths already traces the straight line joining them, without looking at
the request at all. A router that merely lands on that line has demonstrated it
can hit a budget, not that it can read anything. The harness reports the paired
margin against it with a confidence interval.

The horizontal side comes from a manifest of independently trained checkpoints.
Without one, every horizontal estimand prints **"not evaluated"** — which is not
the same as zero.

### 4. Latency

Measured wall-clock and memory, in their own columns, never mixed with the
analytical multiply-accumulates. Reports P50 and P95 time to first token and
per output token, peak memory, cache bytes, route distribution, and the systems
realization gap. Benchmark order is randomized because a laptop's thermal state
drifts across a sweep and running full depth last would hand it the worst
clocks.

## Correctness of the routing path

These gate everything above.

| Check | Result |
|---|---|
| `forward_to_depth(d)` vs a reference run of the first `d` blocks | exact, every depth |
| `continue_from_depth(d, D)` composed with the prefix vs direct execution | exact |
| Routed full depth vs the pre-existing generation path | identical tokens |
| Batched mixed-depth routing vs routing each request alone | identical tokens |
| Depth-capped cache above the cap | raises |
| Cache bytes vs the analytical formula | exact |
| Cache bytes per unit of depth, across depths | identical |
| Vocabulary projections per generated token | exactly one |
| Escalation to full depth vs a full-depth reference | exact |
| An escalated cache decoding onward | matches a natively deep cache |
| GQA with 1, 2, and 4 key/value heads, all three paths | agree |
| Cost counters vs hand-calculated values | exact |

The mixed-depth batching checks **script** the depths rather than relying on a
controller. A uniformly confident controller sends every row to the same depth
and exercises only the path where bucketing is a no-op — which is precisely the
path that cannot fail.

## Instruments (token-level)

Three measurement tools, each isolating one thing.

### Threshold calibration — `utils/calibration.py`

One full-depth teacher-forced pass records, per exit and per position, the
uncertainty, whether the greedy prediction was correct, and the NLL of the
target (`Transformer.exit_statistics`). Everything is reduced over the
vocabulary as it is produced, so peak memory holds one exit's logits rather than
all of them, and an entire threshold sweep costs **one forward pass** regardless
of how many thresholds are tried.

> **Caveat, and it is not minor.** The sweep replays a full-depth pass, so every
> exit saw *exact* keys and values. Real early-exit generation feeds propagated
> states into those layers. The sweep is an **upper bound** on quality — use it
> to narrow the range, then confirm with generation.

### Readout redundancy — `teacher_forced_top1_agreement_oracle_exact_cache`

Cheap, informative, and previously named `oracle_frontier`, which was the
problem. The short name invited it to be read as "an early-exit policy could
save 47.8%", which is not what it measures. The long name is now the function
name, the printed label, and the name in prose; `oracle_frontier` survives only
as an alias so old call sites resolve.

For each position it finds the shallowest exit already producing the *deepest*
exit's greedy token. `ExitStatistics.agrees_with_final` records the per-exit
agreement it keys on.

Every qualifier in the name is load bearing:

| Qualifier | Without it, the reader would assume |
|---|---|
| *teacher forced* | this is what free generation would do |
| *top-1 agreement* | matching the current token implies the continuations stay together — it does not, because the shallow state also enters the cache later tokens read |
| *exact cache* | the exits saw the states early exiting really produces — they saw a full-depth replay |

What it *does* bound is readout redundancy: how much of the stack is already
carrying the final answer, token by token. The split it makes is still the
useful one — depth is a property of the architecture and the data, while the
gap between it and a threshold is a property of the exit policy. Treat the
absolute depth as optimistic and the ratio as the robust quantity.

### Drift — `utils/drift.py`

`measure_drift` runs `trace_decode` against the exact full-depth pass on
identical tokens. Because neither path can wander onto a different
continuation, whatever separates them is attributable to the cache. It returns
per-position KL and greedy agreement, plus per-layer relative error of the
cached keys.

`divergence_point` is the blunter, more relatable number: decoding greedily,
how many tokens pass before the early-exit model says something different.

### Compute-matched comparison — `refresh_vs_threshold`

Refreshing and lowering the threshold both spend compute and both improve
quality, so comparing them at their natural operating points proves nothing.
Each refresh setting is matched against the plain setting reaching the **same
average depth**, found by sweeping thresholds. A positive margin means the
compute is better spent on periodic exact anchors than on uniformly deeper
exits.

## The exposure study — `experiments/exposure.py`

```
python -m experiments.exposure
```

Three arms share **one frozen backbone**, so they cannot differ because one
happened to train the transformer better. Only adapter weights move, and only
the source of their input states differs.

| Arm | Adapters |
|---|---|
| `plain` | none (zero-init = exact identity) |
| `teacher` | fitted on full-depth states |
| `simulated` | fitted on states early exiting really produces |

Each arm is scored under **both** distributions. That cross-evaluation is the
point: adapters fitted on clean states can look excellent on clean states and no
better than nothing on the states they will actually receive.

### Corpus choice is load-bearing

The first version of this study used independent random tokens and returned a
flat null. The cause was the corpus, not the method: with random tokens
attention carries almost no information, hidden states barely depend on the
cache, and corrupting the cache changes nothing measurable. **Any experiment
about cache quality run on such a corpus reports a null by construction.**

The study now uses an **induction corpus** — a random block concatenated with
itself, so predicting the second half requires finding the earlier occurrence
and reading off what followed. Attention is load-bearing by construction. The
backbone reaches 0.512 next-token accuracy, which is near-ceiling given the
first half is unpredictable by design.

Switching corpora moved the adapters' effect on deep-layer cache error from
2–3% to **12–16%**.

### Measuring the premise

`exposure_gap` measures the thing the study assumes exists: the relative
distance between simulated and full-depth exit states. If the distributions
coincide there is no gap to close, and any downstream null says nothing about
the method. The script prints an explicit warning when the gap falls below 2%.

`gap_scan` traces the gap against the conditions that should widen it. Forcing
every position to the same depth measures **exactly zero** by construction —
with no disparity, no token ever attends to another token's approximation — so
the scan varies the *spread* of exit depths instead.

## Correctness checks

These gate everything above; a measurement tool that is subtly wrong is worse
than none.

| Check | Result |
|---|---|
| `threshold=0` reproduces full-depth generation | exact |
| Incremental cache vs one-shot forward | 1.8e-07 |
| Batched vs single-sequence decoding, **mixed exit depths** | identical tokens and exit layers |
| `simulate_early_exit` vs incremental decoding | 2.4e-07 |
| `refresh_every=1` reproduces full-depth generation | exact |
| Zero-init adapter reproduces plain propagation | exact |
| Untrained per-exit CE | ≈ `ln(vocab_size)` |

The mixed-depth batching check needed care. A uniformly confident model exits
everywhere at once and only exercises the "all rows done" fast path; the branch
that needs proving is a step where some rows have exited and others have not. So
exit decisions are **scripted** in that test, taking the confidence model out of
the picture — which produced rows at different depths in 5 of 5 decode steps.

## The diagnostic corpus — `experiments/workloads.py`

The instrument that decides whether any routing result means anything, and the
one that took three attempts. Its failures are recorded in FINDINGS.md §6; the
short version is that a depth gradient must come from a *rule* whose required
composition depth differs, must survive being held out, and must not be
substitutable by a positional shortcut.

Two properties are enforced by tests rather than by care:

- **Every hard request's answer is recoverable by induction** — there is an
  earlier occurrence of the final prompt token followed by the answer. A
  construction bug once truncated the repeat for a fraction of requests, making
  them unanswerable; the model's failure to learn looked like a modelling
  problem for an hour.
- **The repeat distance varies.** With it fixed, "copy from *k* positions back"
  solves the task in one layer with no lookup, and the depth gradient
  disappears.

Training **resamples** requests from the distribution rather than reusing a
fixed corpus. On a fixed 3072-request set the model memorized it: hard-request
accuracy climbed 0.50 → 0.94 across depths on the training split and *fell*
0.29 → 0.24 on held-out requests. Depth was buying memorization, and the
deepest endpoint was the worst one on unseen data.

## Reproducing

```bash
# Request-level pipeline, end to end, from nothing (a few minutes on CPU)
python -m experiments.collect_depth_trajectories --out results/trajectories
python -m experiments.train_depth_controller \
    --trajectories results/trajectories --out results/controller
python -m experiments.evaluate_vertical_routing \
    --trajectories results/trajectories --controller results/controller \
    --out results/evaluation
python -m experiments.benchmark_latency --out results/latency --device cpu

# Tests
python -m unittest discover -s tests -t .

# Token-level work, unchanged
python -m training.data --dataset_name Salesforce/wikitext \
    --dataset_config wikitext-103-raw-v1        # tokenize
python -m training.train --n_layers=12 --d_model=768 --exit_every=2 \
    --learned_kv_propagation=true --wandb_project=early-exit
python -m utils.calibration                     # threshold sweep demo
python -m experiments.exposure                  # the three-arm study
python -m utils.costs                           # cost model sanity check
```

Multi-GPU: `torchrun --standalone --nproc_per_node=4 -m training.train`.
On hosts whose loopback resolves to IPv6 first, `--standalone` stalls in
rendezvous; pin `--master_addr=127.0.0.1` instead.

Every script writes a `run.json` beside its results with the git commit and
dirty flag, hardware, torch version, every seed by name, the full
configuration, input digests, and a schema version.

## What has not been tested

- **The horizontal side.** No family of independently trained models exists, so
  the sharing tax, the substitution ratio, and horizontal regret are all
  unmeasured. The harness accepts a manifest and prints "not evaluated" until
  one is supplied.
- **Real corpora.** Every workload here is synthetic, and the request-level one
  has a hand-built depth structure. Whether real prompts separate by required
  depth at all is untested and is the load-bearing assumption.
- **Continuous batching.** Latency was measured with uniform-depth batches,
  which is the favourable case. Mixed-depth arrival under a scheduler is not
  simulated.
- **Multi-GPU.** All distributed runs used `gloo` on CPU. Code paths are
  exercised; NCCL and real scaling are not.
- **`torch.compile`.** Every run used `--compile_model=false`. `forward`
  returning a dataclass will likely graph-break at the return.
- **FlashAttention dispatch.** Not observable on this hardware. Note that the
  chunked-prefill path builds an explicit `attn_mask`, which is *not*
  flash-eligible; training and single-token decoding are.
