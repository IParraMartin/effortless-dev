#!/bin/bash
# Score the Pythia suite on the same requests as the vertical side.
#
# Usage:
#   sbatch --job-name=pythia jobs/pythia_family.sh
#   sbatch --job-name=pythia-final jobs/pythia_family.sh main
#
# Arguments:
#   $1   revision   step1000 (default, budget-matched) | main | any Pythia revision
#   $2+  extra      passed through to experiments.horizontal_family
#
# Which revision, and why it is not "main"
# ----------------------------------------
#
# The default is step1000 = 1000 * 1024 * 2048 = 2.097B tokens, which is exactly
# the budget jobs/controlled_arms.sh trains to. Scoring `main` instead compares a
# 2.097B-token backbone against 300B-token Pythia -- 120x more data -- and any
# gap would be attributed to capacity sharing.
#
# The cost of matching is that the horizontal frontier compresses. Measured on a
# small fixture, the 70m-to-160m gap in bits per byte is 0.016 at step1000
# against 0.111 at main. The vertical substitution ratio divides by that
# frontier improvement, so a compressed denominator makes the ratio noisy -- the
# evaluation flags unstable ratios rather than reporting them. Run `main` as a
# secondary, deliberately-confounded comparison if the matched frontier turns out
# too narrow to divide by, and label it as such.
#
#SBATCH --job-name=pythia
#SBATCH --account=fc_bsclab
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=iparra@berkeley.edu
set -euo pipefail

REVISION="${1:-step1000}"
shift || true
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

MODELS="${MODELS:-EleutherAI/pythia-70m,EleutherAI/pythia-160m,EleutherAI/pythia-410m,EleutherAI/pythia-1b,EleutherAI/pythia-1.4b}"
DATA="${DATA:-$SCRATCH_ROOT/data/pile-neox/val.bin}"
TRAJECTORIES="${TRAJECTORIES:-$SCRATCH_ROOT/results/traj-multi-seed1}"
OUT="${OUT:-$SCRATCH_ROOT/results/pythia-${REVISION}}"
N_REQUESTS="${N_REQUESTS:-2048}"

# One revision per model, since --revisions requires a full list rather than a
# partial one: a partial list would score some tiers at a different budget from
# others without saying so.
COUNT=$(awk -F, '{print NF}' <<<"$MODELS")
REVISIONS=$(printf "%s," $(for _ in $(seq "$COUNT"); do echo "$REVISION"; done))
REVISIONS="${REVISIONS%,}"

echo "models:      $MODELS"
echo "revision:    $REVISION"
echo "corpus:      $DATA"
echo "vertical:    $TRAJECTORIES"
echo

# --trajectories makes the pairing checkable: it verifies that the requests drawn
# here carry the same content digests as the vertical side's records, so a
# mismatch is an error rather than a silently unpaired comparison.
#
# --batch_size is recorded in the manifest. Absolute bits-per-byte shifts with it
# (fp32 GEMM kernel selection changes accumulation order with the batch
# dimension), so every tier is scored at one value and numbers from different
# values must not be mixed.
"${PY[@]}" -m experiments.horizontal_family \
    --models="$MODELS" \
    --revisions="$REVISIONS" \
    --data="$DATA" \
    --eos_id=0 \
    --shapes=64:32,128:64,256:128 \
    --n_requests="$N_REQUESTS" \
    --trajectories="$TRAJECTORIES" \
    --out="$OUT" \
    --run-dir="$SCRATCH_ROOT/runs/pythia-${REVISION}" \
    --device=cuda \
    --dtype=bf16 \
    --batch_size=8 \
    ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}

echo
echo "Then compare, in the only unit that crosses tokenizers:"
echo "  ${PY[*]} -m experiments.evaluate_vertical_routing \\"
echo "      --trajectories=$TRAJECTORIES \\"
echo "      --controller=\$SCRATCH_ROOT/results/controller --controller_seed=0 \\"
echo "      --manifest=$OUT/manifest.json \\"
echo "      --quality_metric=bits_per_byte \\"
echo "      --out=\$SCRATCH_ROOT/results/evaluation"
