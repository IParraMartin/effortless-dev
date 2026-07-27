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

## Two mechanisms, deliberately separable

|  | Request-level (primary) | Token-level (extension) |
|---|---|---|
| Decision made | once, before generation | every token |
| Decided by | a controller reading a shallow prompt probe | an entropy threshold on the output distribution |
| Layers above the choice | never executed | executed for key/value projections |
| Cache | capped — upper layers never allocated | full depth, upper entries synthesized |
| Cache approximation | **none**; every executed layer sees exact keys and values | propagated states, which drift |
| K/V memory saved | exactly `1 − d/L` | none |
| Latency saved at batch > 1 | yes, if requests are bucketed by depth | no |

The token-level path is the pre-existing work in this repository and still
runs. It is kept, and it is no longer the headline: it saves arithmetic that a
batched server cannot turn into time, and its cache approximation is the source
of most of the complexity here. Request-level routing gives up per-token
granularity and gets exactness and a real memory saving in exchange.

## Architecture

```
src/
  config.py     TransformerConfig, RoutingConfig, TrainConfig, CLI parsing
  modules.py    KVCache (depth-capped), RMSNorm, RoPE, ExitModule,
                KVPropagator, uncertainty measures
  model.py      Transformer: forward_to_depth / continue_from_depth /
                endpoint_logits / generate_routed / escalate, plus the
                original multi-exit training and early-exit generation
  routing.py    DepthController, prompt pooling, RoutingTrace
  tokenizer.py  Hugging Face tokenizer glue
training/
  data.py       corpus preparation, packed memmap loading
  train.py      DDP training loop
  distributed.py
utils/
  costs.py      analytical MAC model + measured counters
  statistics.py bootstrap, frontiers, VSR, non-inferiority tests
  calibration.py threshold sweep + the readout-agreement oracle
  drift.py      token-level cache drift instrumentation
  provenance.py run records: git state, hardware, seeds, digests
experiments/
  workloads.py                  diagnostic corpus + demo backbone
  collect_depth_trajectories.py per-request, per-depth labels and costs
  train_depth_controller.py     frozen-backbone controller fit
  evaluate_vertical_routing.py  the central comparison
  benchmark_latency.py          measured wall-clock and memory
  exposure.py                   token-level adapter study
tests/
```

**Depth means executed blocks**, always, and runs from 1 to `n_layers`. A layer
*index* is one less. The two conversions live in `Transformer.layer_of_depth`
and `depth_of_layer` and nowhere else; an earlier off-by-one here was a real
bug.

## Commands

Everything runs from the repository root as a module.

```bash
# Tests (~100, a few seconds, no GPU)
python -m unittest discover -s tests -t .

# 1. Collect per-request, per-depth labels and costs.
#    Trains a small demonstration backbone in-process if no checkpoint given.
python -m experiments.collect_depth_trajectories --out results/trajectories

#    ...or against a real checkpoint from training/train.py:
python -m experiments.collect_depth_trajectories \
    --checkpoint checkpoints/final.pt --out results/trajectories

# 2. Fit the controller on the frozen backbone, over several seeds.
python -m experiments.train_depth_controller \
    --trajectories results/trajectories --out results/controller

# 3. Evaluate every system and emit JSON + Markdown.
python -m experiments.evaluate_vertical_routing \
    --trajectories results/trajectories \
    --controller results/controller \
    --out results/evaluation
#    Add --manifest models.json to include independently trained models.

# 4. Measure wall-clock and memory, separately from the MAC estimates.
python -m experiments.benchmark_latency --out results/latency --device cpu
```

Pre-existing work, unchanged:

```bash
python -m training.data --dataset_name Salesforce/wikitext \
    --dataset_config wikitext-103-raw-v1
python -m training.train --n_layers=12 --d_model=768 --exit_every=2
torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 \
    -m training.train                    # --standalone stalls on IPv6 hosts
python -m utils.calibration              # threshold sweep + readout oracle
python -m experiments.exposure           # token-level adapter study
python -m utils.costs                    # cost model sanity check
```

### The horizontal manifest

The comparison this repository is built for needs independently trained models,
and training a family is out of scope here. Supply them through a manifest:

```json
[
  {"model_id": "small", "tokenizer_id": "gpt2", "tier": 2,
   "cost": 0.33, "results": "results/horizontal/small.json"},
  {"model_id": "large", "tokenizer_id": "gpt2", "tier": 6,
   "cost": 1.00, "results": "results/horizontal/large.json"}
]
```

Each `results` file is a JSON array of per-request quality on **the same
requests in the same order** as the trajectories. Mismatched lengths and mixed
tokenizers are rejected rather than silently compared. Without a manifest the
harness prints *"not evaluated"* for every horizontal estimand — which is not
the same as zero, and is the difference between an incomplete result and a
wrong one.

## Reproducibility

Every experiment writes a `run.json` containing the git commit and whether the
tree was dirty, the platform, Python and torch versions, the device, every seed
by name, the full configuration, dataset and checkpoint identifiers with
SHA-256 digests, and a schema version. Results live in the same file as the
provenance, because two files drift apart.

## Known limitations

Stated here rather than discovered later:

- **Scale.** Every number comes from models of at most 8 layers on synthetic
  corpora, on a laptop with no GPU. The mechanisms are verified; the magnitudes
  are not evidence about real models.
- **No *learnable* adaptivity gain has been demonstrated.** The plain oracle
  shows +0.051 of headroom, but it picks per request by knowing how each
  candidate turned out. The reachable ceiling — a strong cross-fitted predictor
  restricted to the probe features — does not beat the best fixed depth at all,
  and the trained controller sits at that ceiling. The headroom is real but
  determined by the continuation, not the prompt. See
  [PAPER/FINDINGS.md](PAPER/FINDINGS.md) §7.
- **Whether real prompts separate by depth is unknown.** Four diagnostic
  workloads were built; three failed, two of them by producing a convincing
  gradient that was memorization or a positional shortcut. See §6.
- **The horizontal side is unevaluated.** No family of independent models has
  been trained, so the central claim is untested. The harness is ready for one:
  the manifest path, sharing-tax alignment, cascade costing, and the
  substitution test are exercised end to end against a synthetic manifest and
  covered by `tests/test_evaluation.py`.
- **Grouped routing is not a serving implementation.** `generate_routed`
  buckets requests in Python and runs the buckets in sequence. It is correct
  and countable; it is not fast, and continuous batching is not simulated.
- **Latency was measured with uniform-depth batches.** That is the favourable
  case. Mixed-depth arrival under a real scheduler has not been benchmarked.
- **Multi-GPU is untested.** Distributed paths have only run under `gloo` on
  CPU.
- **`torch.compile` is untested.** Every run used `--compile_model=false`;
  returning dataclasses will likely graph-break.
- **Free-running labels are collected but the controller is fitted on
  teacher-forced ones by default.** Those answer different questions; the
  choice is recorded in each run and `--quality_metric free_running_reward`
  switches it.

## Write-up

See [`PAPER/`](PAPER/) — method, experiments, findings (including corrected and
retracted ones), and the roadmap.
