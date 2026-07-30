#!/bin/bash
# The sharing-tax experiment, run as a causal comparison rather than a pair of
# runs that happen to differ.
#
# Usage:
#   # 1. mint one common parent per seed (fast, CPU-ish, needs a GPU only to build)
#   sbatch --job-name=vr-parent-s1 jobs/controlled_arms.sh parent 1
#   sbatch --job-name=vr-parent-s2 jobs/controlled_arms.sh parent 2
#
#   # 2. two arms per seed, branching from that parent
#   sbatch --job-name=vr-s1-final jobs/controlled_arms.sh final 1
#   sbatch --job-name=vr-s1-multi jobs/controlled_arms.sh multi 1
#   sbatch --job-name=vr-s2-final jobs/controlled_arms.sh final 2
#   sbatch --job-name=vr-s2-multi jobs/controlled_arms.sh multi 2
#
# Arguments:
#   $1   mode    parent | final | multi
#   $2   seed    integer, identifying the parent both arms share
#   $3+  extra   passed through to training.train
#
# What makes this a causal comparison, and what each piece is for
# ---------------------------------------------------------------
#
# **One serialized parent per seed.** The arms branch from a file, not from a
# matching seed. Two arms of a comparison differ in construction by definition,
# and any change in how many random draws construction consumes moves the
# initialization even under an identical seed. That is the defect that made the
# first two Savio runs non-comparable: measured differences of ~0.0955 on the
# embedding between the one-exit and six-exit builds. Branching from a file
# removes the argument -- the digest either matches or it does not.
#
# **The anchored objective.** The legacy objective normalizes across all exits,
# so at six exits the final endpoint carried 12/42 = 0.2857 of the hard-target
# coefficient against the control's 1.0. A degraded endpoint is then the expected
# consequence of dividing its weight by 3.5, not evidence about sharing. Under
# anchored_v1 the full coefficient is fixed and only the shallow weights
# normalize, so the two arms below differ in exactly one number:
# shallow_loss_weight. At 0.0 no shallow readout is computed at all, which makes
# the control an exact final-only run rather than an approximation to one.
#
# **32,000 steps.** 32,000 x 65,536 = 2.097B tokens, which is exactly
# pythia-160m at revision step1000. Comparing against final Pythia instead would
# measure 300B versus 2.5B tokens -- 120x more data -- and attribute it to
# sharing. Matching 300B is not an option: ~91 days at the measured throughput.
#
# **Two seeds.** One seed gives a point estimate with no variance, and the
# trailing-window scatter in the earlier runs was +/-0.05 nats against an effect
# of 0.075. A predeclared non-inferiority test needs an interval that includes
# seed variation, or it reports run-to-run noise as an effect.
#
# Afterwards
# ----------
#
#   # score the family at the *matched* budget, not at main
#   sbatch --job-name=pythia jobs/pythia_family.sh
#
#SBATCH --job-name=vr-arm
#SBATCH --account=fc_bsclab
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=iparra@berkeley.edu
set -euo pipefail

MODE="${1:-parent}"
SEED="${2:-1}"
shift 2 || true
EXTRA_FLAGS=("$@")

# Locate _env.sh. Slurm copies a batch script to /var/spool/slurmd/job<id>/, so
# BASH_SOURCE points there and not at the repo -- the sibling-file assumption that
# works when running this directly is wrong under sbatch. SLURM_SUBMIT_DIR is
# where sbatch was invoked from; both it and its jobs/ subdirectory are checked,
# so submitting from the repository root or from inside jobs/ both work.
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
trap report_failure EXIT
cd "$REPO_DIR"
report_env

# Matched to pythia-160m@step1000 = 1000 * 1024 * 2048 = 2.097B tokens.
# 32,000 * 65,536 = 2.097B, so the two budgets agree to the token.
STEPS="${STEPS:-32000}"
DATA_DIR="${DATA_DIR:-$SCRATCH_ROOT/data/pile-neox}"
PARENT_DIR="${PARENT_DIR:-$SCRATCH_ROOT/checkpoints/parents}"
PARENT="$PARENT_DIR/parent-seed${SEED}.pt"

# Architecture matched to pythia-160m: 12 layers, d_model 768. The exit placement
# is the only architectural difference between the arms, and under the anchored
# objective it costs the control nothing because alpha=0 skips shallow readouts.
ARCH=(
    --n_layers=12
    --d_model=768
    --n_heads=12
    --exit_every=2
    --min_exit_layer=2
    --objective_version=anchored_v1
)

COMMON=(
    --tokenizer_name=EleutherAI/pythia-160m
    --data_dir="$DATA_DIR"
    --seq_len=1024
    --batch_size=8
    --grad_accum_steps=8
    --max_steps="$STEPS"
    --seed="$SEED"
    --dtype=bf16
    --eval_every=500
    --save_every=2000
    --exits_per_step=2
    --find_unused_parameters=true
    --wandb_project="$WANDB_PROJECT"
)

case "$MODE" in
parent)
    # One step, purely to serialize the initialization. The arms then branch from
    # this file, so "same parent" is a checkable digest rather than a claim about
    # seeds.
    mkdir -p "$PARENT_DIR"
    echo "minting common parent for seed $SEED -> $PARENT"
    "${PY[@]}" -m training.train \
        "${ARCH[@]}" "${COMMON[@]}" \
        --shallow_loss_weight=0.0 \
        --max_steps=1 \
        --eval_every=0 --save_every=0 --sweep_every=0 \
        --save_init_to="$PARENT" \
        --out_dir="$SCRATCH_ROOT/checkpoints/parent-scratch-seed${SEED}" \
        --wandb_project=none \
        ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
    echo
    echo "parent digest:"
    sha256sum "$PARENT"
    echo
    echo "Both arms for seed $SEED must report this digest. If they do not, they"
    echo "did not share a parent and the comparison is not causal."
    ;;
final | multi)
    if [ ! -f "$PARENT" ]; then
        echo "error: $PARENT does not exist." >&2
        echo "Mint it first:  sbatch jobs/controlled_arms.sh parent $SEED" >&2
        exit 1
    fi

    # The single number that separates the arms. 0.0 is an exact final-only run:
    # no shallow readout is computed, so it matches a one-exit model in gradient,
    # in memory and in wall clock.
    if [ "$MODE" = "final" ]; then
        ALPHA=0.0
    else
        ALPHA="${ALPHA:-0.5}"
    fi

    OUT_DIR="${OUT_DIR:-$SCRATCH_ROOT/checkpoints/vr-${MODE}-seed${SEED}}"
    echo "arm:    $MODE (shallow_loss_weight=$ALPHA)"
    echo "parent: $PARENT"
    sha256sum "$PARENT"
    echo "budget: $STEPS steps x 65,536 tokens = $((STEPS * 65536)) tokens"
    echo "        matches pythia-160m@step1000 (2,097,152,000 tokens)"
    echo

    "${PY[@]}" -m training.train \
        "${ARCH[@]}" "${COMMON[@]}" \
        --shallow_loss_weight="$ALPHA" \
        --distill_weight=0.5 \
        --init_from="$PARENT" \
        --out_dir="$OUT_DIR" \
        --grad_diagnostics_every=4000 \
        ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
    ;;
*)
    echo "error: mode must be parent | final | multi, got '$MODE'" >&2
    exit 1
    ;;
esac
