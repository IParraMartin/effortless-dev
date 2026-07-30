# Savio jobs

Slurm scripts for `fc_bsclab`. Every one sources [`_env.sh`](_env.sh), which
redirects the caches off the 10 GB home quota and configures Weights & Biases.

```bash
# once, on a login node
bash jobs/setup_env.sh
uv run wandb login

# 1. tokenize a corpus         (CPU, bigmem partition)
sbatch --job-name=data jobs/prepare_data.sh

# 2. train the model           (1x A40 by default)
sbatch --job-name=lm jobs/train.sh

# only if a run was submitted with WANDB_MODE=offline
bash jobs/sync_wandb.sh
```

| Script | Does |
|---|---|
| [`setup_env.sh`](setup_env.sh) | `uv sync` into `.venv`, print versions, run the tests |
| [`prepare_data.sh`](prepare_data.sh) | tokenize a corpus into `data/{train,val}.bin` |
| [`train.sh`](train.sh) | train the transformer; extra flags pass through to `training.train` |
| [`sync_wandb.sh`](sync_wandb.sh) | push offline W&B runs from a login node |

## Weights & Biases

Runs stream online. `wandb login` once on a login node writes the credential to
`~/.netrc`, which the compute nodes read; each job warns in its first lines if an
online run is about to go anonymous. The run id is keyed on the **job name** with
`WANDB_RESUME=allow`, so a requeue continues the same run rather than starting a
second one — give each experiment its own `--job-name`.

Logged series: `train/loss`, `train/lr`, `train/tokens`, `train/tokens_per_sec`
(reduced across ranks, on rank 0), and `eval/loss`.

Offline is available as a fallback for a firewalled node:

```bash
sbatch --export=ALL,WANDB_MODE=offline --job-name=lm jobs/train.sh
bash jobs/sync_wandb.sh    # later, from a login node
```

## Resuming

`train.sh` finds the newest `step-*.pt` in its output directory and passes
`--resume_from` automatically, so re-submitting the same job name continues where
it stopped. The architecture comes from the checkpoint on resume, so changing an
architecture flag then has no effect — start a new job name instead.

## Sizing

The defaults are a ~124M model (12 layers, 768 wide) at 1024 context, batch 8 ×
accum 4 on one A40 → 32k tokens/step, 20k steps → ~2.6B tokens. The first job
prints `train/tokens_per_sec` within a minute; use it to check the run fits the
72-hour wall clock, and lower `MAX_STEPS` if not (auto-resume covers you if it
does not). To scale:

```bash
# more GPUs — torchrun adapts via SLURM_GPUS_ON_NODE
sbatch --gres=gpu:A40:4 --cpus-per-task=32 --job-name=lm-4g jobs/train.sh

# smaller card (A5000, 24 GB): fewer sequences, more accumulation
sbatch --partition=savio4_gpu --qos=a5k_gpu4_normal --gres=gpu:A5000:1 \
       --cpus-per-task=4 --export=ALL,BATCH_SIZE=4,GRAD_ACCUM=8 \
       --job-name=lm-a5k jobs/train.sh

# shorter run
sbatch --export=ALL,MAX_STEPS=5000 --job-name=lm-short jobs/train.sh
```

## Local testing

The scripts run off the cluster. Point them at a scratch directory and fake the
Slurm variables (`uv` must be on `PATH`):

```bash
export SCRATCH_ROOT=/tmp/savio-test REPO_DIR=$PWD
export SLURM_JOB_NAME=smoke SLURM_GPUS_ON_NODE=1
export BATCH_SIZE=2 GRAD_ACCUM=1 MAX_STEPS=12
bash jobs/train.sh --n_layers=4 --d_model=128 --n_heads=4 --seq_len=128 \
    --dtype=fp32 --num_workers=0
```
