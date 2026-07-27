#!/bin/bash
# Push offline Weights & Biases runs to the server. Run on a LOGIN node.
#
#   bash jobs/sync_wandb.sh              # sync everything not yet sent
#   bash jobs/sync_wandb.sh --clean      # and delete runs that synced cleanly
#
# Jobs log online by default, so most runs never need this. It exists for the
# ones submitted with WANDB_MODE=offline — a node that turns out to be
# firewalled, or a run started while the service was unreachable. Offline
# recording cannot fail; it only defers, and this is the deferred half.
set -euo pipefail

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

CLEAN="${1:-}"

if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$WANDB_CONFIG_DIR/settings" ]; then
    echo "Not logged in to Weights & Biases."
    echo "Either export WANDB_API_KEY=... or run: uv run wandb login"
    exit 1
fi

# WANDB_DIR is the parent; wandb creates its own 'wandb/' beneath it.
RUN_ROOT="$WANDB_DIR/wandb"
[ -d "$RUN_ROOT" ] || { echo "No runs under $RUN_ROOT"; exit 0; }

# A plain glob rather than `mapfile`, which is bash 4+ and simply is not
# present on older shells — where it fails as "command not found" and takes the
# whole script with it under `set -e`.
RUNS=()
shopt -s nullglob
for run in "$RUN_ROOT"/offline-run-*; do
    [ -d "$run" ] && RUNS+=("$run")
done
shopt -u nullglob

if [ "${#RUNS[@]}" -eq 0 ]; then
    echo "No offline runs to sync in $RUN_ROOT"
    exit 0
fi

echo "Syncing ${#RUNS[@]} offline run(s) to project '$WANDB_PROJECT'"

failed=0
for run in ${RUNS[@]+"${RUNS[@]}"}; do
    echo "--- $(basename "$run")"
    # Do not abort the loop on one bad run; a single corrupted transaction log
    # should not block every other run from being uploaded.
    if uv run wandb sync --project "$WANDB_PROJECT" "$run"; then
        if [ "$CLEAN" = "--clean" ]; then
            rm -rf "$run"
            echo "    removed after successful sync"
        fi
    else
        echo "    FAILED — left in place for a retry"
        failed=$((failed + 1))
    fi
done

echo
echo "Done. $(( ${#RUNS[@]} - failed )) synced, $failed failed."
[ "$failed" -eq 0 ]
