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

_find_env() {
    local candidate
    for candidate in "${SLURM_SUBMIT_DIR:-}/jobs" "${SLURM_SUBMIT_DIR:-}" \
                     "$(dirname "${BASH_SOURCE[0]}")" "$(pwd)/jobs" "$(pwd)"; do
        [ -n "$candidate" ] && [ -f "$candidate/_env.sh" ] && {
            printf '%s' "$candidate/_env.sh"; return 0; }
    done
    echo "Cannot find jobs/_env.sh. Run from the repo root." >&2
    return 1
}
source "$(_find_env)"
cd "$REPO_DIR"

CLEAN="${1:-}"

# Uses the shared credential probe rather than testing $WANDB_CONFIG_DIR, which
# _env.sh deliberately no longer sets -- referencing it here was an unbound
# variable under `set -u`, so this check aborted the script it was meant to
# guard. `wandb login` writes to ~/.netrc, which is what the probe reads.
if ! wandb_credentials_present; then
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
    if "${PY[@]}" -m wandb sync --project "$WANDB_PROJECT" "$run"; then
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
