#!/bin/bash
# The routing pipeline against a trained checkpoint: collect trajectories, fit
# the controller, evaluate every system, benchmark latency.
#
# Usage:
#   sbatch --job-name=vr-route jobs/route.sh checkpoints/vr-exits/final.pt
#   sbatch --job-name=vr-route jobs/route.sh checkpoints/vr-exits/final.pt \
#          results/horizontal/manifest.json
#
# Arguments:
#   $1  checkpoint            (required)
#   $2  horizontal manifest   (optional)
#
# **This job answers the go/no-go question**, and it is the one to read first.
# In the evaluation output, look at `probe-policy gain`, not `outcome oracle - best fixed`.
# The plain oracle chooses per request by knowing how each candidate turned out,
# which no deployable policy can do. On the toy workload in this repository the
# outcome oracle showed +0.051 of headroom while the cross-fitted probe policy — one
# cross-fitted predictor restricted to the probe features — showed **+0.008**,
# so 85% of the apparent gain was unreachable by construction, and judging the
# controller against it reported a near-optimal policy as a failure.
#
# If `probe-policy gain` is around zero on real text too, request-level routing has
# no case and no amount of controller work will create one. That result is worth
# having early and it costs one job.
#
#SBATCH --job-name=vr-route
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

CHECKPOINT="${1:?Usage: sbatch jobs/route.sh <checkpoint.pt> [manifest.json]}"
MANIFEST="${2:-}"

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
report_env
echo "checkpoint=$CHECKPOINT manifest=${MANIFEST:-<none, horizontal side skipped>}"

[ -f "$CHECKPOINT" ] || { echo "No such checkpoint: $CHECKPOINT"; exit 1; }

TAG="${SLURM_JOB_NAME:-route}"
RESULTS="$REPO_DIR/results/$TAG"
mkdir -p "$RESULTS"

# 1. Per-request, per-depth labels and costs. Quality and cost are stored
#    separately so the price of compute can change later without recollecting.
#    Free-running labels cost generation but are the only ones that support a
#    serving claim, so they are collected rather than skipped.
"${PY[@]}" -m experiments.collect_depth_trajectories \
    --checkpoint="$CHECKPOINT" \
    --out="$RESULTS/trajectories" \
    --probe_depth="${PROBE_DEPTH:-2}" \
    --n_requests="${N_REQUESTS:-4096}" \
    --max_new_tokens="${MAX_NEW_TOKENS:-16}" \
    --cache_dtype=bf16

# 2. Fit the controller on the frozen backbone, over several seeds. Every seed
#    is reported; the split is by source, and the held-out half is split again
#    so calibration never touches the reported numbers.
"${PY[@]}" -m experiments.train_depth_controller \
    --trajectories="$RESULTS/trajectories" \
    --out="$RESULTS/controller" \
    --quality_metric="${QUALITY_METRIC:-teacher_forced_accuracy}" \
    --routing_lambda="${ROUTING_LAMBDA:-0.2}" \
    --seeds 0 1 2

# 3. Evaluate every system. Emits JSON and a Markdown summary.
MANIFEST_FLAGS=()
[ -n "$MANIFEST" ] && MANIFEST_FLAGS=(--manifest="$MANIFEST")

"${PY[@]}" -m experiments.evaluate_vertical_routing \
    --trajectories="$RESULTS/trajectories" \
    --controller="$RESULTS/controller" \
    --out="$RESULTS/evaluation" \
    ${MANIFEST_FLAGS[@]+"${MANIFEST_FLAGS[@]}"}

# 4. Measured wall-clock and memory, kept in their own columns. A routed system
#    can save arithmetic and still be slower; the two are never mixed.
# Benchmarks the *trained* architecture, not the demonstration one. Without the
# checkpoint this reports latency for a 64-wide toy, which shares no kernel
# regime with a 768-wide model and whose vocabulary head is three orders of
# magnitude cheaper.
"${PY[@]}" -m experiments.benchmark_latency \
    --checkpoint="$CHECKPOINT" \
    --out="$RESULTS/latency" \
    --device="${DEVICE:-cuda}" \
    --batch_sizes 1 8 32 \
    --prompt_lens 128 512 \
    --new_tokens 32 128

echo
echo "=================================================================="
sed -n '/## Adaptivity/,/endpoint or feature/p' "$RESULTS/evaluation/evaluation.md" || true
echo "=================================================================="
echo "Full results in $RESULTS"
