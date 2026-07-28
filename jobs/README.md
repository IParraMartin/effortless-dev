# Savio jobs

Slurm scripts for `fc_bsclab`. Every one sources [`_env.sh`](_env.sh), which
redirects the caches off the 10 GB home quota and configures Weights & Biases.

```bash
# once, on a login node
bash jobs/setup_env.sh
uv run wandb login

# confirm W&B works *from a compute node* before spending a GPU allocation
sbatch --job-name=vr-wandb-check jobs/check_wandb.sh

# 1. tokenize a corpus                (CPU, ~hours)
sbatch --job-name=vr-data    jobs/prepare_data.sh

# 2. train the backbone               (1x A40; see sizing note below)
sbatch --job-name=vr-exits   jobs/train.sh exits
sbatch --job-name=vr-noexits jobs/train.sh noexits

# 3. route and evaluate               (1x A40, ~1h)
sbatch --job-name=vr-route   jobs/route.sh checkpoints/vr-exits/final.pt

# 4. only if submitted with WANDB_MODE=offline
bash jobs/sync_wandb.sh
```

## Weights & Biases

**Runs stream online.** Savio compute nodes reach the service on this account,
so there is nothing to sync afterwards. `wandb login` once on a login node and
the credential lands in `~/.netrc`, which the compute nodes read.

`WANDB_CONFIG_DIR` is deliberately left alone. Only `WANDB_DIR` and
`WANDB_CACHE_DIR` are redirected to scratch — those are the ones that grow, and
moving the settings directory risks disturbing a login that already works to
save kilobytes.

### Checking you are logged in

**Do not use `wandb status`.** It prints the settings file, so it reports
`"api_key": null` on a machine that is perfectly well logged in through
`~/.netrc` — a false negative. The check that means something resolves the
effective credential and round-trips to the server:

```bash
sbatch --job-name=vr-wandb-check jobs/check_wandb.sh   # from a compute node
bash jobs/check_wandb.sh                               # or on a login node
```

Submit it rather than running it locally. A login node proves nothing about
whether the node your 24-hour training job lands on can reach the service, and
that is the failure that costs something. The script resolves the key, calls
`viewer()` to confirm who you are, then creates and finishes a real run. It
exits non-zero on failure, so Slurm marks the job failed and the `FAIL` mail
fires.

The one-liner equivalent, if you would rather not submit anything:

```bash
uv run python -c "import wandb; print(wandb.api.viewer()['entity'])"
```

Every job also checks for a credential in `report_env` and warns if an online
run is about to go anonymous, so you find out in the first lines of the log
rather than at the end of training.

Offline stays available as a fallback:

```bash
sbatch --export=ALL,WANDB_MODE=offline --job-name=vr-exits jobs/train.sh exits
bash jobs/sync_wandb.sh                 # later, from a login node
```

Worth reaching for if a node turns out to be firewalled or the service is
unreachable mid-run. Offline recording cannot fail; it only defers.

What lands in W&B, verified end to end:

| Series | Contents |
|---|---|
| `train/loss`, `train/lr`, `train/tokens`, `train/tokens_per_sec` | reduced across ranks, logged on rank 0 |
| `train/exit_ce/layer_N`, `eval/exit_ce/layer_N` | one series per exit — a healthy run shows deep exits below shallow ones, converging as self-distillation pulls the shallow ones up |
| `eval/loss` | held-out |
| `sweep/*` plus a table and scatter plot | the accuracy-versus-depth tradeoff, every `sweep_every` steps |
| config | both dataclasses, with `n_layers`/`d_model`/`n_exits` promoted so they work as axes when comparing runs |

The run id is keyed on the **job name**, not the job id, with `WANDB_RESUME=allow`.
A requeued job therefore continues the same W&B run instead of starting a second
one with the step counter reset. Give each experiment its own `--job-name`.

## Resuming

`train.sh` looks for the newest `step-*.pt` in its output directory and passes
`--resume_from` automatically. Re-submitting the same job name continues where
it stopped, which matters against a 72-hour wall clock. The architecture comes
from the checkpoint on resume, so changing an architecture flag has no effect —
start a new job name instead.

## Sizing

The defaults are a 124M model (12 layers, 768 wide) at 1024 context, batch 8 ×
accum 4 on one A40 → 32k tokens/step, 20k steps → **2.6B tokens**.

Wall-clock is *unmeasured* — nothing in this repository has run on an A40. The
first job will print `train/tokens_per_sec` within a minute; use that to decide
whether 20k steps fits the 72-hour limit before committing to it, and lower
`MAX_STEPS` if not. Auto-resume covers you if it does not.

Memory is dominated by logits, not activations. Each scored exit holds
`batch × seq_len × vocab` and `cross_entropy` keeps its log-softmax for the
backward pass, so they cannot be freed as the loop advances — about **3.3 GB per
scored exit** at these settings. `--exits_per_step=2` scores three exits (two
sampled plus the final one) for ~10 GB, leaving room on a 48 GB A40. Scoring all
six would want ~20 GB of logits alone.

To scale:

```bash
# more GPUs — torchrun adapts via SLURM_GPUS_ON_NODE
sbatch --gres=gpu:A40:4 --cpus-per-task=32 --job-name=vr-exits-4g jobs/train.sh exits

# smaller card (A5000, 24 GB)
sbatch --partition=savio4_gpu --qos=a5k_gpu4_normal --gres=gpu:A5000:1 \
       --cpus-per-task=4 --export=ALL,BATCH_SIZE=4,GRAD_ACCUM=8 \
       --job-name=vr-exits-a5k jobs/train.sh exits

# shorter run
sbatch --export=ALL,MAX_STEPS=5000 --job-name=vr-exits-short jobs/train.sh exits
```

## What the two training variants are for

Together they are the **sharing tax**: the cheapest decisive experiment
available. `exits` trains the elastic backbone; `noexits` sets
`exit_every = n_layers`, leaving one exit on the final layer — an ordinary
language model, no separate code path. Compare their **final-layer** quality at
matched budget. Multi-exit training can degrade the top layer, because shallow
exits pull representations toward early linear decodability. If that tax is
large, the thesis loses to "train one good model and distill it", and it is
worth knowing before spending anything on a model family.

## Reading the routing result

`route.sh` prints the adaptivity table at the end. **Read `learnable gain`, not
`oracle − best fixed`.** The plain oracle chooses per request by knowing how
each candidate turned out, which no deployable policy can. On the toy workload
in this repository the plain oracle showed +0.051 while the reachable ceiling
was +0.008 — 85% of the apparent headroom was unreachable by construction, and
judging the controller against it reported a near-optimal policy as a failure.

If `learnable gain` is near zero on real text as well, request-level routing has
no case and no amount of controller work will create one. That is the go/no-go,
and it costs one job.

## Watching jobs

Two shell helpers, sourced rather than pasted:

```bash
echo "source $PWD/jobs/follow.sh" >> ~/.bashrc
echo "source $PWD/jobs/check.sh"  >> ~/.bashrc
```

| Command | Does |
|---|---|
| `follow [jobid]` | tails one job's stdout and stderr live; defaults to your newest job |
| `check [jobid\|name] [-v]` | one line per queued and running job: step, loss, rate, ETA |

`check` resolves each log path from `scontrol show job`, **per job**, and never
from a fixed directory. That is the whole point: a monitor pointed at one
hardcoded results directory reports whatever last wrote there under whichever
job it happens to be displaying, so a finished run from another project appears
as live progress on a job that has barely started. Deriving the path from Slurm
makes that unrepresentable, and makes the helper work for any job in any repo.

It flags three failures that look perfectly healthy in `squeue`:

- a running job whose log has gone quiet for ten minutes;
- one whose log holds a traceback, an OOM, or a kill;
- one whose **ETA exceeds the time left on its wall clock** — invisible until
  the moment Slurm kills it, and the cheapest of the three to act on.

Rate and ETA come from the difference between the last two step lines, not from
total steps over job elapsed. The latter charges queue-to-start, model
construction and the first data load against the training rate, which
understated a real run by 55%.

## Local testing

These scripts run off the cluster, which is how two bugs in them were found (a
resume path that killed the job on its first run, and a `mapfile` that is absent
from older shells). Point them at a scratch directory and fake the Slurm
variables:

```bash
export SCRATCH_ROOT=/tmp/savio-test REPO_DIR=$PWD
export SLURM_JOB_NAME=smoke SLURM_GPUS_ON_NODE=1
export BATCH_SIZE=2 GRAD_ACCUM=1 MAX_STEPS=12
bash jobs/train.sh exits --n_layers=4 --d_model=128 --n_heads=4 --seq_len=128 \
    --dtype=fp32 --num_workers=0
```

`uv` must be on `PATH`; substitute your interpreter if not.
