# Vertical routing through an elastic backbone

A decoder-only transformer that can be executed to a chosen depth, plus the
machinery to decide that depth per request and to measure whether doing so is
worth it.

The research question:

> **When can vertical routing through one elastic backbone replace horizontal
> routing across several independently trained models of the same family?**

Nothing here answers that yet. What exists is a correct implementation, an
accounting system that does not flatter it, and an evaluation harness that
refuses to call a difference a substitution unless a predeclared test passes.

**Read [START_HERE.md](START_HERE.md) first** — it holds the plan: three
experiments, the commands in order, and the condition under which each one kills
the next. This file is the code tour.

| file | what it is |
|---|---|
| [START_HERE.md](START_HERE.md) | the plan |
| [CURRENT.md](CURRENT.md) | state of the runs: established, retracted, not established |
| [DESCRIPTION.md](DESCRIPTION.md) | research diary and decision log |
| [MIGRATIONS.md](MIGRATIONS.md) | schema and default changes, and how to reproduce old behaviour |
| [jobs/README.md](jobs/README.md) | cluster specifics |

## The method

Take a model that is already trained. **Freeze it.** Attach a zero-initialized
adapter and a readout to intermediate layers, and train only those. The original
model's output is then unchanged *to the bit*, because none of its weights moved.
Then learn a controller that reads a one-block probe of the prompt and picks a
depth for the request.

This is **fine-tuning a frozen parent**, not pre-training. `src/retrofit.py`
holds the ladder of adaptation modes — `frozen_tied_head`, `frozen_untied_head`,
`frozen_exit_adapter`, `selective_unfreeze`, `lora`, `qlora`, `full_finetune` —
and reports which of them preserve the parent exactly rather than approximately.

## Two mechanisms, deliberately separable

|  | Request-level (primary) | Token-level (cut to future work) |
|---|---|---|
| Decision made | once, before generation | every token |
| Decided by | a controller reading a shallow prompt probe | an entropy threshold on the output distribution |
| Layers above the choice | never executed | executed for key/value projections |
| Cache | capped — upper layers never allocated | full depth, upper entries synthesized |
| Cache approximation | **none**; every executed layer sees exact keys and values | propagated states, which drift |
| K/V memory saved | exactly `1 − d/L`, verified to the byte | none |
| Latency saved at batch > 1 | yes, if requests are bucketed by depth | no |

The token-level path is the pre-existing work in this repository and still runs.
It is **cut to future work**: it saves arithmetic that a batched server cannot
turn into time, and its cache approximation is the source of most of the
complexity here. Its gating diagnostic — token-level oracle gain beyond the
request-level cap — was never built, so the branch is closed rather than pending.

The vocabulary head is why routing is priced on hidden states rather than logits:
one head call costs about **4.6 blocks** at `d_model=768`, `V=52000`. A router
that read logits to decide depth would spend more on the decision than it saves.

## Architecture

```
src/
  config.py     TransformerConfig, RoutingConfig, TrainConfig, CLI parsing
  modules.py    KVCache (depth-capped), RMSNorm, RoPE, ExitModule, ExitAdapter,
                KVPropagator, uncertainty measures
  model.py      Transformer: forward_to_depth / continue_from_depth /
                endpoint_logits / generate_routed / escalate, the anchored
                multi-exit objective, and gradient-conflict diagnostics
  retrofit.py   the no-regret adaptation ladder, LoRA, parent preservation
  routing.py    DepthController, prompt pooling, RoutingTrace
  tokenizer.py  Hugging Face tokenizer glue
training/
  data.py       corpus preparation, StatelessBlockSampler, packed memmap loading
  train.py      DDP training loop, exact resume, common-parent serialization
  distributed.py
utils/
  costs.py      analytical MAC model + measured counters + K/V audit
  statistics.py bootstrap (document-clustered), frontiers, VSR, non-inferiority
  calibration.py threshold sweep + the readout-agreement oracle
  drift.py      token-level cache drift instrumentation
  provenance.py RunRecord (single file) and RunArtifacts (run directory)
experiments/
  workloads.py                  real_text_corpus + the synthetic mechanism test
  collect_depth_trajectories.py per-request, per-depth labels and costs
  retrofit_parent.py            build an elastic model from a trained parent
  no_regret.py                  did the retrofit damage the parent?
  train_depth_controller.py     frozen-backbone controller fit
  evaluate_vertical_routing.py  the central comparison
  horizontal_family.py          score Pythia on the same requests
  benchmark_latency.py          measured wall-clock and memory
  exposure.py                   token-level adapter study (cut experiment)
tests/
```

**Depth means executed blocks**, always, and runs from 1 to `n_layers`. A layer
*index* is one less. The two conversions live in `Transformer.layer_of_depth` and
`depth_of_layer` and nowhere else; an earlier off-by-one here was a real bug.

## Commands

Everything runs from the repository root as a module.

```bash
# Tests (429, a few seconds, no GPU)
python -m unittest discover -s tests -t .

# 1. Retrofit a trained parent. Prints a bit-identical parent check.
python -m experiments.retrofit_parent \
    --checkpoint checkpoints/vr-noexits/final.pt \
    --run-dir runs/retrofit-adapter \
    --mode frozen_exit_adapter --exit_adapter_rank 32 --exit_every 2

# 2. Per-request, per-depth labels and costs, on real text.
python -m experiments.collect_depth_trajectories \
    --corpus real_text --data data/val.bin --eos_id 50256 \
    --checkpoint checkpoints/retrofit-adapter/final.pt \
    --n_requests 4096 --out results/traj

# 3. Fit the controller on the frozen backbone, over several seeds.
python -m experiments.train_depth_controller \
    --trajectories results/traj --out results/controller --seeds 0 1 2

# 4. Evaluate every system and emit JSON + Markdown.
python -m experiments.evaluate_vertical_routing \
    --trajectories results/traj --controller results/controller \
    --controller_seed 0 --out results/evaluation

# 5. Score an independent family, then pass --manifest to step 4.
python -m experiments.horizontal_family \
    --models EleutherAI/pythia-70m,EleutherAI/pythia-160m,EleutherAI/pythia-410m \
    --revisions step1000,step1000,step1000 \
    --data data/val.bin --eos_id 50256 --out results/pythia

# Did a non-frozen retrofit damage the parent? One-sided, document-clustered.
python -m experiments.no_regret \
    --parent checkpoints/vr-noexits/final.pt \
    --candidate runs/retrofit-lora/checkpoints/retrofit.pt \
    --data data/val.bin --eos_id 50256 --quality_margin 0.01

# Measured wall-clock and memory, separately from the MAC estimates.
python -m experiments.benchmark_latency --out results/latency --device cpu
```

Corpus preparation and training from scratch:

```bash
python -m training.data --dataset_name HuggingFaceFW/fineweb-edu \
    --tokenizer_name gpt2
python -m training.train --n_layers=12 --d_model=768 --exit_every=2 \
    --objective_version=anchored_v1 --shallow_loss_weight=0.5
torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 \
    -m training.train                    # --standalone stalls on IPv6 hosts
```

`--corpus synthetic` exists on the collector and is a **mechanism test only**.
Its depth structure is a rule installed by the experimenter, so a result on it
says whether the machinery works, never whether real prompts vary in the depth
they need. Its own metadata says so.

## Reading the evaluation

Read **`probe-policy gain`**, not `outcome oracle − best fixed`.

The outcome oracle picks per request by looking at how each depth turned out,
which no deployable policy can do. `cross_fitted_probe_policy` is one specified
learner restricted to the probe features, fitted out of fold. On the toy workload
the oracle showed +0.051 of headroom while the probe policy attained +0.008, so
judging a controller against the oracle would report a near-optimal policy as a
failure. The probe policy is **not a ceiling** — it is one model class, so a
better one can beat it and the regret column can legitimately go negative.

Also read the static-mixture column. A weighted coin flip between two fixed
depths reaches any average cost between them without reading the request at all,
so a router whose interval spans zero has shown only that it can hit a budget.

Intervals resample **documents**, not requests. Two requests cut from one document
are not independent evidence, and an unclustered interval is too narrow — which
for a one-sided non-inferiority test biases toward passing.

## Comparing against a different tokenizer

Pythia uses GPT-NeoX; this backbone uses GPT-2. The same string becomes different
token sequences, so **per-token loss is not comparable**: a tokenizer that splits
more finely earns a lower average loss per piece without predicting better.

Quality across families is therefore **bits per byte**:

```
bits_per_byte = nll_sum_nats / (ln 2 * utf8_bytes(continuation))
```

Bytes belong to the text, not to anyone's vocabulary.
`evaluate_vertical_routing` refuses a manifest whose unit does not match the
vertical side's rather than differencing two different quantities.

Budget must match too. `pythia-160m` at `main` saw 300B tokens against this
backbone's 2.5B; `--revisions step1000` selects the 2.097B-token checkpoint.
`experiments/horizontal_family.py` writes the manifest, with per-shape costs, a
stated quality unit, and a content digest per request so the pairing is checkable.

Without a manifest the harness prints *"not evaluated"* for every horizontal
estimand — which is not the same as zero, and is the difference between an
incomplete result and a wrong one.

## Reproducibility

`utils/provenance.RunArtifacts` owns a run directory holding the resolved config,
the command, the environment including the installed package set, the hardware,
the git commit and dirty diff, input digests, seeds by purpose, an append-only
`metrics.jsonl`, and a resume chain. Everything is written through `os.replace`,
so a killed process leaves the previous contents rather than a prefix that still
parses. A run that cannot state its commit, config, seeds and command refuses to
start.

Resume is exact rather than approximate. `training.data.StatelessBlockSampler`
makes block order a pure function of the seed and the global position, so there is
no cursor to serialize; checkpoints carry every random stream plus the
exit-rotation counter. `tests/test_resume.py` runs the real entry point in real
processes — 100 updates against 50 plus 50 with a process kill between — and
compares consumed blocks, scored exits, parameters, optimizer moments and the next
draw from every stream.

Causal arms branch from a **serialized** common parent (`--save_init_to`,
`--init_from`), not from a matching seed: two arms differ in construction by
definition, so any change in how many draws construction consumes moves the
initialization even under one seed.

## Known limitations

Stated here rather than discovered later.

- **Scale.** No number comes from a model larger than 124M parameters.
- **The two runs on disk are retracted as a causal comparison.** They differed in
  backbone initialization *and* objective weighting — the six-exit arm gave its
  final endpoint `12/42 = 0.2857` of the hard-target weight against the other's
  `1.0` — so their 0.075-nat gap is not a sharing tax. Their reported held-out CEs
  also describe rank zero's validation shard rather than the whole split. See
  [CURRENT.md](CURRENT.md) under *Retracted*.
- **No endpoint has been scored on real held-out text yet.** The machinery exists
  and the pipeline has been verified end to end on a fixture corpus.
- **No *learnable* adaptivity gain has been demonstrated.** On the toy workload
  the headroom was real but determined by the continuation, not the prompt.
  Whether real prompts separate by depth is unknown.
- **No systems claim.** K/V memory falls exactly `1 − d/L`, verified across 12
  configurations. Latency, throughput, goodput and energy are **unmeasured**: the
  serving benchmark is a cut experiment. Analytical MAC counts are never reported
  as latencies.
- **No downstream tasks.** At this scale task accuracy is near chance, so quality
  is held-out likelihood only.
- **Grouped routing is not a serving implementation.** `generate_routed` buckets
  requests in Python and runs the buckets in sequence. Correct and countable, not
  fast; continuous batching is not simulated.
- **Multi-GPU is untested.** Distributed paths have only run under `gloo` on CPU.
- **`torch.compile` is untested.** Every run used `--compile_model=false`;
  returning dataclasses will likely graph-break.
- **Absolute bits-per-byte depends on `batch_size`** in fp32 on CPU, by up to 2.6%
  of one request's total NLL — GEMM kernel selection changes the accumulation
  order with the batch dimension. Every tier in one manifest is scored at the same
  value, so a frontier is internally consistent; values from different batch sizes
  are not comparable.
- **Two training objectives exist.** `legacy_normalized` is the default and is
  retained only so the runs on disk reproduce. Use
  `--objective_version=anchored_v1` for anything new. See
  [MIGRATIONS.md](MIGRATIONS.md).
