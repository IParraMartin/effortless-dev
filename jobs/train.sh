#!/bin/bash
# Train the elastic backbone, or its no-exit control, on real text.
#
# Usage:
#   sbatch --job-name=vr-exits    jobs/train.sh exits
#   sbatch --job-name=vr-noexits  jobs/train.sh noexits
#   sbatch --job-name=vr-exits-4g --gres=gpu:A40:4 --cpus-per-task=32 \
#          jobs/train.sh exits
#
# Arguments:
#   $1   variant       exits | noexits          (default: exits)
#   $2+  extra flags    passed through to training.train
#
# The two variants together are the **sharing tax** experiment, which is the
# cheapest decisive result available: train one model with exits and one without
# at a matched budget, then compare *final-layer* quality. Multi-exit training
# can degrade the top layer, because shallow exits pull representations toward
# early linear decodability. If that tax is large, the whole thesis loses to
# "train one good model and distill it", and it is worth knowing before spending
# anything on a model family.
#
# `noexits` is not a separate code path: exit_every = n_layers leaves exactly one
# exit, on the final layer, which is an ordinary language model.
#
#SBATCH --job-name=vr-train
#SBATCH --account=fc_bsclab
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=iparra@berkeley.edu
set -euo pipefail

VARIANT="${1:-exits}"
shift || true
EXTRA_FLAGS=("$@")

# Optional argument arrays below are expanded as `${ARR[@]+"${ARR[@]}"}` rather
# than `"${ARR[@]}"`. The plain form is an unbound-variable error on an empty
# array under `set -u` in bash before 4.4. Savio is newer than that, so this is
# not for Savio's benefit — it is what lets these scripts be run off the cluster
# before submitting, which is how the resume bug in _env.sh was found.

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$REPO_DIR"
mkdir -p logs

N_GPUS="$(detect_gpus)"
[ "$N_GPUS" -ge 1 ] || { echo "No GPU visible; submit with --gres=gpu:A40:N"; exit 1; }
report_env

# ------------------------------------------------------------------ sizing
# ~124M parameters: 12 layers, 768 wide, 12 heads, GQA off at this scale.
N_LAYERS=12
SEQ_LEN=1024
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_STEPS="${MAX_STEPS:-20000}"

case "$VARIANT" in
  exits)
    # Six exits, at depths 2/4/6/8/10/12 — the tiers request-level routing will
    # choose among.
    #
    # exits_per_step is the memory knob and it is not optional here. Logits are
    # batch x seq_len x vocab per exit, and cross_entropy holds its log-softmax
    # for the backward pass, so they cannot be freed as the loop advances. At
    # batch 8 x 1024 tokens x 50304 vocab that is ~3.3 GB *per scored exit*.
    # Scoring all six would want ~20 GB of logits alone; scoring three (two
    # sampled shallow exits plus the final one, which always participates) wants
    # ~10 GB and leaves room on a 48 GB A40.
    #
    # Sampling a subset leaves the unselected exit norms without gradient, which
    # DDP rejects unless told to expect it.
    ARCH_FLAGS=(
        --exit_every=2
        --exits_per_step=2
        --min_exit_layer=1
        --self_distill_weight=0.5
        --find_unused_parameters=true
    )
    ;;
  noexits)
    # One exit, on the final layer. The control arm for the sharing tax.
    ARCH_FLAGS=(
        --exit_every="$N_LAYERS"
        --min_exit_layer=0
        --self_distill_weight=0.0
        --find_unused_parameters=false
    )
    ;;
  *)
    echo "Unknown variant '$VARIANT' (expected: exits | noexits)"; exit 1 ;;
esac

OUT_DIR="${OUT_DIR:-$REPO_DIR/checkpoints/${SLURM_JOB_NAME:-$VARIANT}}"
mkdir -p "$OUT_DIR"

# Automatic resume. A 72-hour wall clock is shorter than some runs, and a
# requeue that silently restarted from step zero would waste the whole budget
# without any error to notice.
RESUME_FLAGS=()
RESUME_FROM="$(latest_checkpoint "$OUT_DIR")"
if [ -n "$RESUME_FROM" ]; then
    echo "Resuming from $RESUME_FROM"
    RESUME_FLAGS=(--resume_from="$RESUME_FROM")
fi

# torchrun rendezvous: pinned to IPv4 because --standalone stalls resolving
# ip6.arpa on hosts whose loopback resolves to IPv6 first. The port is derived
# from the job id so two jobs landing on one node do not collide.
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 20000))

echo "variant=$VARIANT gpus=$N_GPUS batch=$BATCH_SIZE x accum=$GRAD_ACCUM"
echo "global batch = $((BATCH_SIZE * GRAD_ACCUM * N_GPUS)) sequences"
echo "             = $((BATCH_SIZE * GRAD_ACCUM * N_GPUS * SEQ_LEN)) tokens/step"
echo "total budget = $((BATCH_SIZE * GRAD_ACCUM * N_GPUS * SEQ_LEN * MAX_STEPS)) tokens"

uv run torchrun \
    --nnodes=1 --nproc_per_node="$N_GPUS" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    -m training.train \
    --data_dir="$REPO_DIR/data" \
    --tokenizer_name=gpt2 \
    --seq_len="$SEQ_LEN" \
    --n_layers="$N_LAYERS" \
    --d_model=768 \
    --n_heads=12 \
    --batch_size="$BATCH_SIZE" \
    --grad_accum_steps="$GRAD_ACCUM" \
    --max_steps="$MAX_STEPS" \
    --learning_rate=3e-4 \
    --min_lr=3e-5 \
    --warmup_steps=500 \
    --dtype=bf16 \
    --compile_model=false \
    --num_workers=4 \
    --ddp_backend=nccl \
    --eval_every=500 \
    --eval_steps=50 \
    --sweep_every=2000 \
    --save_every=1000 \
    --log_every=20 \
    --out_dir="$OUT_DIR" \
    --wandb_project="$WANDB_PROJECT" \
    --wandb_run_name="${SLURM_JOB_NAME:-$VARIANT}" \
    --wandb_mode="$WANDB_MODE" \
    "${ARCH_FLAGS[@]}" \
    ${RESUME_FLAGS[@]+"${RESUME_FLAGS[@]}"} \
    ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}

echo "Finished $(timestamp). Checkpoints in $OUT_DIR"
if [ "$WANDB_MODE" = "offline" ]; then
    echo "Run recorded offline. Push it with:  bash jobs/sync_wandb.sh"
else
    echo "Run streamed to W&B project '$WANDB_PROJECT' as '$WANDB_RUN_ID'."
fi
