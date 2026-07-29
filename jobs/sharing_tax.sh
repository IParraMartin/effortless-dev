#!/bin/bash
# The sharing tax: what one backbone serving every size costs at full depth.
#
# Usage:
#   sbatch --job-name=vr-tax jobs/sharing_tax.sh \
#          checkpoints/vr-exits/final.pt checkpoints/vr-noexits/final.pt
#
# Arguments:
#   $1  shared backbone checkpoint, trained with exits      (required)
#   $2  independently trained checkpoint, no exits          (required)
#
# This is the job the two training arms were run for. `vr-exits` and
# `vr-noexits` saw the same tokens in the same order at the same budget, and
# differ only in whether shallow exits were trained alongside the final layer.
# The question is whether carrying those exits degrades the top of the model:
# if it does, "train one good model and distill it" beats an elastic backbone
# before routing is even considered, and the rest of the programme is moot.
#
# Both checkpoints are scored on **the same requests in the same order**,
# because the comparison is paired -- it bootstraps a per-request difference.
# That is why both collections run inside one job with one seed rather than
# being submitted separately: two jobs with drifting flags produce two files
# that still subtract, still bootstrap, and still report a confident interval
# around nothing. build_manifest verifies the alignment and refuses a mismatch,
# but not creating the mismatch is better than catching it.
#
#SBATCH --job-name=vr-tax
#SBATCH --account=fc_bsclab
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=iparra@berkeley.edu
set -euo pipefail

SHARED="${1:?Usage: sbatch jobs/sharing_tax.sh <exits.pt> <noexits.pt>}"
INDEPENDENT="${2:?Usage: sbatch jobs/sharing_tax.sh <exits.pt> <noexits.pt>}"

# Locate _env.sh. Slurm copies a batch script to /var/spool/slurmd/job<id>/,
# so BASH_SOURCE points there and not at the repo.
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

_report_exit() {
    local status=$?
    [ "$status" -eq 0 ] && return 0
    echo
    echo "FAILED with status $status at $(timestamp)"
    sacct -j "${SLURM_JOB_ID:-0}" \
        -o JobID%20,JobName%12,State%22,ExitCode,MaxRSS,Elapsed 2>/dev/null || true
}
trap _report_exit EXIT

N_GPUS="$(detect_gpus)"
report_env

for path in "$SHARED" "$INDEPENDENT"; do
    [ -f "$path" ] || { echo "No such checkpoint: $path"; exit 1; }
done

TAG="${SLURM_JOB_NAME:-tax}"
RESULTS="$REPO_DIR/results/$TAG"
mkdir -p "$RESULTS"

# Settings shared by both collections. Held in variables rather than repeated,
# so the two calls cannot drift apart: the seed and request count are what make
# the two sides describe the same requests, and a comparison built on different
# ones is not repairable afterwards.
SEED="${SEED:-0}"
N_REQUESTS="${N_REQUESTS:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
PROBE_DEPTH="${PROBE_DEPTH:-2}"
QUALITY_METRIC="${QUALITY_METRIC:-teacher_forced_accuracy}"

echo "shared      = $SHARED"
echo "independent = $INDEPENDENT"
echo "requests    = $N_REQUESTS at seed $SEED, $MAX_NEW_TOKENS new tokens"

collect() {
    local checkpoint="$1" out="$2"
    "${PY[@]}" -m experiments.collect_depth_trajectories \
        --checkpoint="$checkpoint" \
        --out="$out" \
        --probe_depth="$PROBE_DEPTH" \
        --n_requests="$N_REQUESTS" \
        --max_new_tokens="$MAX_NEW_TOKENS" \
        --seed="$SEED" \
        --cache_dtype=bf16
}

echo; echo "--- trajectories: shared backbone"
collect "$SHARED" "$RESULTS/vertical"

echo; echo "--- trajectories: independently trained model"
collect "$INDEPENDENT" "$RESULTS/independent"

echo; echo "--- manifest"
"${PY[@]}" -m experiments.build_manifest \
    --vertical="$RESULTS/vertical" \
    --independent="noexits=$RESULTS/independent" \
    --quality_metric="$QUALITY_METRIC" \
    --out="$RESULTS/horizontal"

# The controller is fitted here too, because the evaluation reports the routing
# estimands and the sharing tax from one run and one held-out split. Splitting
# them across jobs would mean two different validation subsets.
echo; echo "--- controller"
"${PY[@]}" -m experiments.train_depth_controller \
    --trajectories="$RESULTS/vertical" \
    --out="$RESULTS/controller" \
    --quality_metric="$QUALITY_METRIC" \
    --routing_lambda="${ROUTING_LAMBDA:-0.2}" \
    --seeds 0 1 2

echo; echo "--- evaluation"
"${PY[@]}" -m experiments.evaluate_vertical_routing \
    --trajectories="$RESULTS/vertical" \
    --controller="$RESULTS/controller" \
    --manifest="$RESULTS/horizontal/manifest.json" \
    --quality_metric="$QUALITY_METRIC" \
    --out="$RESULTS/evaluation"

# The headline number, read straight out of the JSON rather than scraped from
# the Markdown, so a reformatted report cannot silently empty this section.
echo
echo "=================================================================="
"${PY[@]}" - "$RESULTS/evaluation/evaluation.json" <<'PY'
import json
import sys

# provenance.write nests the payload under "results"; unwrap if present so this
# reads the same file whether or not it carries a run record.
document = json.loads(open(sys.argv[1]).read())
results = document.get("results", document)
taxes = results.get("estimands", {}).get("sharing_tax")
if not taxes:
    print("No sharing tax reported. Check evaluation.json for")
    print("sharing_tax_unmatched_tiers -- the independent model's tier has to")
    print("be one of the backbone's exit depths for the two to be paired.")
    raise SystemExit(0)

print("Sharing tax  (independent - shared endpoint, per request)")
print("positive means training with exits cost the endpoint quality\n")
for row in taxes:
    interval = row["sharing_tax"]
    print(
        f"  tier {row['tier']:>3}   independent {row['independent_quality']:.4f}"
        f"   endpoint {row['endpoint_quality']:.4f}"
        f"   tax {interval['estimate']:+.4f}"
        f"  [{interval['low']:+.4f}, {interval['high']:+.4f}]"
    )
print(
    "\nAn interval straddling zero means the tax is not distinguishable from"
    "\nnothing at this sample size -- which is the result the elastic backbone"
    "\nneeds, and is not the same as having shown it is zero."
)
PY
echo "=================================================================="
echo "Full results in $RESULTS"
