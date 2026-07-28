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

# Locate _env.sh. Slurm copies a batch script to /var/spool/slurmd/job<id>/,
# so BASH_SOURCE points there and not at the repo — the sibling-file assumption
# that works when running this directly is wrong under sbatch. SLURM_SUBMIT_DIR
# is where sbatch was invoked from; both it and its jobs/ subdirectory are
# checked, so submitting from the repo root or from inside jobs/ both work.
_find_env() {
    local candidate
    for candidate in "${SLURM_SUBMIT_DIR:-}/jobs" "${SLURM_SUBMIT_DIR:-}" \
                     "$(dirname "${BASH_SOURCE[0]}")" "$(pwd)/jobs" "$(pwd)"; do
        if [ -n "$candidate" ] && [ -f "$candidate/_env.sh" ]; then
            printf '%s' "$candidate/_env.sh"
            return 0
        fi
    done
    echo "Cannot find jobs/_env.sh (looked near SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}" >&2
    echo "and $(pwd)). Submit from the repository root: sbatch jobs/<script>.sh" >&2
    return 1
}
source "$(_find_env)"
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

TOKENS_PER_STEP=$(( BATCH_SIZE * GRAD_ACCUM * SEQ_LEN * N_GPUS ))

# TOKEN_BUDGET is the honest way to specify a run, because steps are not
# comparable across jobs: adding GPUs multiplies the tokens each step consumes,
# so "20,000 steps" means six times as much data on six cards as on one. The
# sharing-tax comparison needs both arms trained on the *same* number of tokens,
# and pinning the budget rather than the step count is what makes that hold even
# if the two land on differently sized allocations.
if [ -n "${TOKEN_BUDGET:-}" ]; then
    MAX_STEPS=$(( (TOKEN_BUDGET + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))
    echo "TOKEN_BUDGET=$TOKEN_BUDGET -> MAX_STEPS=$MAX_STEPS on $N_GPUS GPU(s)"
fi

TOTAL_TOKENS=$(( TOKENS_PER_STEP * MAX_STEPS ))

# Warn when the run would consume the corpus more than once. Repeating is not
# an error and the loader handles it, but a second epoch on a pretraining
# corpus is a different experiment from a single pass and should be deliberate.
CORPUS_META="$REPO_DIR/data/train.bin.meta.json"
if [ -f "$CORPUS_META" ]; then
    CORPUS_TOKENS="$(sed -n 's/.*"n_tokens": *\([0-9]*\).*/\1/p' "$CORPUS_META" | head -1)"
    if [ -n "$CORPUS_TOKENS" ] && [ "$TOTAL_TOKENS" -gt "$CORPUS_TOKENS" ]; then
        echo "NOTE: this run wants $TOTAL_TOKENS tokens but the corpus holds"
        echo "      $CORPUS_TOKENS, so it will wrap and repeat data "
        echo "      ($(( TOTAL_TOKENS / CORPUS_TOKENS ))x or more). Lower"
        echo "      MAX_STEPS/TOKEN_BUDGET, or prepare a larger corpus."
    fi
fi

case "$VARIANT" in
  exits)
    # Six exits, at depths 2/4/6/8/10/12 — the tiers request-level routing will
    # choose among.
    # exits_per_step is the memory knob. Logits are batch x seq_len x vocab
    # per scored exit, and cross_entropy holds its log-softmax for the backward
    # pass, so they cannot be freed as the loop advances: about 0.4 GB per exit
    # per sequence at 1024 tokens and a 50k vocabulary.
    #
    # Set it to "none" to score every exit. That costs memory but buys two
    # things worth having on several GPUs: full gradient coverage every step,
    # and no need for find_unused_parameters, whose overhead grows with the
    # number of ranks. On a 24 GB card, batch 4 scoring all six exits costs the
    # same as batch 8 scoring three.
    EXITS_PER_STEP="${EXITS_PER_STEP:-2}"
    ARCH_FLAGS=(
        --exit_every=2
        --exits_per_step="$EXITS_PER_STEP"
        --min_exit_layer=1
        --self_distill_weight=0.5
        # Sampling a subset leaves the unselected exit norms without gradient,
        # which DDP rejects unless told to expect it.
        --find_unused_parameters=$(
            [ "$EXITS_PER_STEP" = "none" ] && echo false || echo true
        )
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
echo "             = $TOKENS_PER_STEP tokens/step"
echo "total budget = $TOTAL_TOKENS tokens over $MAX_STEPS steps"

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
