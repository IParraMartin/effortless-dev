# Start here

One question, three experiments, one GPU job. Everything else in this repository
is reference material or was cut — see *What was cut* below.

If you read nothing else, read the table in **The plan** and run step 1.

---

## The question

> Can a trained full-depth language model be given useful shallow endpoints and a
> request-level depth controller, **without degrading the original model**, such
> that one elastic backbone recovers part of the quality–cost frontier of routing
> among independently trained models of different sizes?

Three claims, one experiment each:

| claim | plain English |
|---|---|
| **no regret** | adding endpoints does not damage the original model |
| **useful tiers** | at least two shallow endpoints are worth using |
| **substitution** | a controller over those endpoints recovers part of what a portfolio of separate models would give you |

---

## The method, in one paragraph

Take a model that is already trained. **Freeze it.** Attach a small readout to
some intermediate layers and train only those. The original model's output is
then unchanged *to the bit*, because none of its weights moved. Then learn a
controller that reads the first layer or two of a request and decides how deep
that request needs to go. Compare the resulting quality–cost curve against a
family of separately trained models of increasing size.

This is **fine-tuning a frozen parent**, not pre-training. Nothing is trained
from scratch.

---

## The plan

Run in order. Each step can kill the next, which is the point.

| step | command | what it answers | cost | kill condition |
|---|---|---|---|---|
| **1** | `experiments/retrofit_parent.py` then train its exits | no regret, useful tiers | free | shallow tiers no better than chance → the parent's middle layers carry nothing, and the method has no basis |
| **2** | `collect_depth_trajectories` → `train_depth_controller` → `evaluate_vertical_routing` | does a controller beat a static mixture | ~1 GPU-hour | `probe-policy gain` ≈ 0 → requests do not differ in the depth they need, and there is nothing to route on |
| **3** | `jobs/pythia_family.sh` | substitution ratio | one GPU job | frontier too compressed to divide by → report the sharing tax alone, not a ratio |

### Step 1

```bash
python -m experiments.retrofit_parent \
    --checkpoint checkpoints/vr-noexits/final.pt \
    --run-dir runs/retrofit-adapter \
    --mode frozen_exit_adapter --exit_adapter_rank 32 --exit_every 2
```

Prints `preserved max logit difference 0.000e+00` — the parent is bit-identical,
which is the no-regret claim, verified rather than argued. Then train the exits
it created, which touches no backbone weight:

```bash
python -m training.train \
    --resume_from=runs/retrofit-adapter/checkpoints/retrofit.pt \
    --objective_version=anchored_v1 --shallow_loss_weight=0.5 --distill_weight=0.5 \
    --data_dir=data --out_dir=checkpoints/retrofit-adapter --max_steps=4000
```

### Step 2

```bash
python -m experiments.collect_depth_trajectories \
    --corpus real_text --data data/val.bin --eos_id 50256 \
    --checkpoint checkpoints/retrofit-adapter/final.pt \
    --n_requests 4096 --out results/traj

python -m experiments.train_depth_controller \
    --trajectories results/traj --out results/controller --seeds 0 1 2

python -m experiments.evaluate_vertical_routing \
    --trajectories results/traj --controller results/controller \
    --controller_seed 0 --out results/evaluation
```

Read **`probe-policy gain`**, not `outcome oracle − best fixed`. The oracle looks
at how each depth turned out, which no deployable policy can do.

### Step 3

```bash
sbatch --job-name=pythia jobs/pythia_family.sh step1000
```

`step1000` is Pythia at 2.097B tokens. Not `main` — that is 300B tokens, 120×
more data than your backbone saw, and the gap would be data, not sharing.

---

## What was cut, and why

| cut | reason |
|---|---|
| **4 Pile scratch arms** (64 GPU-h) | measured the cost of multi-exit *pre-training*, which is not the method being proposed. A frozen parent pays no sharing tax. Returns only if step 1's frozen tiers turn out useless. |
| Pythia 1b, 1.4b | far past the backbone's capacity; they do not inform substitution |
| Serving benchmark | a second paper. K/V memory is verified exactly; latency, throughput and energy are stated as unmeasured |
| Distribution shift, safety by endpoint | separate contribution |
| Token-level routing, learned K/V propagation | future work, gated behind step 2 showing headroom |
| RL / contextual bandit controller | future work, gated behind a working supervised controller |

This is a smaller paper than the original brief describes. It is a *complete*
smaller one: three claims, three experiments, each claim testable and each
failure informative.

---

## Things that will confuse you later

**"KV cache" means three different things here.**
1. *Depth-capped cache* — a request stopping at depth `d` allocates only `d/L` of
   the cache. Verified to the byte across 12 configurations. An implementation
   invariant, not a finding.
2. *Learned K/V propagation* — only relevant to token-level routing. Never
   trained; `learned_kv_propagation=False` in every run so far.
3. *K/V bytes as a cost metric* — a column the controller can price depth by.

**Two training objectives exist.**
- `legacy_normalized` (the default) normalizes across all exits, so at six exits
  the final endpoint receives only `12/42 = 0.2857` of the hard-target weight.
  Retained only so the two 2026-07-27 runs reproduce.
- `anchored_v1` (use this) fixes the full-depth coefficient at 1 and normalizes
  only the shallow weights. `--shallow_loss_weight=0` is *exactly* a final-only
  run.

**The two runs on disk are retracted as a causal comparison.** They differed in
backbone initialization *and* objective weighting, so their 0.075-nat gap is not
a sharing tax. Details in `CURRENT.md` under *Retracted*.

**Cross-tokenizer quality is bits per byte, never per-token loss.** Pythia uses
GPT-NeoX, this backbone uses GPT-2. A tokenizer that splits text more finely
earns a lower per-token loss without predicting better. The evaluation refuses
the comparison in the wrong unit rather than averaging it.

---

## Where everything lives

| file | what it is |
|---|---|
| `START_HERE.md` | this — the plan |
| `CURRENT.md` | state of the runs: established, retracted, not established |
| `DESCRIPTION.md` | research diary and decision log; history of what changed and why |
| `MIGRATIONS.md` | schema and default changes, with how to reproduce old behaviour |
| `jobs/README.md` | cluster specifics |

Run the tests with `python -m unittest discover -s tests -t .`
