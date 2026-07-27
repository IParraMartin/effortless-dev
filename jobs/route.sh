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
# In the evaluation output, look at `learnable gain`, not `oracle - best fixed`.
# The plain oracle chooses per request by knowing how each candidate turned out,
# which no deployable policy can do. On the toy workload in this repository the
# plain oracle showed +0.051 of headroom while the reachable ceiling — a strong
# cross-fitted predictor restricted to the probe features — showed **+0.008**,
# so 85% of the apparent gain was unreachable by construction, and judging the
# controller against it reported a near-optimal policy as a failure.
#
# If `learnable gain` is around zero on real text too, request-level routing has
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

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
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
uv run python -m experiments.collect_depth_trajectories \
    --checkpoint="$CHECKPOINT" \
    --out="$RESULTS/trajectories" \
    --probe_depth="${PROBE_DEPTH:-2}" \
    --n_requests="${N_REQUESTS:-4096}" \
    --max_new_tokens="${MAX_NEW_TOKENS:-16}" \
    --cache_dtype=bf16

# 2. Fit the controller on the frozen backbone, over several seeds. Every seed
#    is reported; the split is by source, and the held-out half is split again
#    so calibration never touches the reported numbers.
uv run python -m experiments.train_depth_controller \
    --trajectories="$RESULTS/trajectories" \
    --out="$RESULTS/controller" \
    --quality_metric="${QUALITY_METRIC:-teacher_forced_accuracy}" \
    --routing_lambda="${ROUTING_LAMBDA:-0.2}" \
    --seeds 0 1 2

# 3. Evaluate every system. Emits JSON and a Markdown summary.
MANIFEST_FLAGS=()
[ -n "$MANIFEST" ] && MANIFEST_FLAGS=(--manifest="$MANIFEST")

uv run python -m experiments.evaluate_vertical_routing \
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
uv run python -m experiments.benchmark_latency \
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
