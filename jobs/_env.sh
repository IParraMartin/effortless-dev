#!/bin/bash
# Shared environment for the Savio jobs in this directory. Sourced, not run.
#
# Home is a 10 GB quota, so uv's caches, the Hugging Face cache and Weights &
# Biases run data are all redirected to scratch. Not `set -e`: this file is
# sourced, and a non-zero probe should not kill the caller's shell.
set -uo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/global/scratch/users/iparra}"
REPO_DIR="${REPO_DIR:-$SCRATCH_ROOT/effortless-dev}"

# uv installs to ~/.local/bin, which a non-interactive batch shell lacks on PATH.
export PATH="$HOME/.local/bin:$PATH"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH_ROOT/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SCRATCH_ROOT/.local/uv-python}"
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

# Unbuffered stdout: block buffering to a file loses everything a killed process
# had printed, making a crash indistinguishable from a hang.
export PYTHONUNBUFFERED=1

# Call the venv interpreter directly. `uv run` re-resolves the project and locks
# .venv on every invocation, which on scratch costs minutes per job. The fallback
# keeps this working before setup_env.sh has built the environment.
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
if [ -x "$VENV_DIR/bin/python" ]; then
    PY=("$VENV_DIR/bin/python")
else
    PY=(uv run python)
fi
export VENV_DIR

# Weights & Biases. Runs stream online by default; WANDB_MODE=offline defers to
# jobs/sync_wandb.sh. The run id is keyed on the job name so a requeue continues
# the same run instead of splitting it in two.
export WANDB_PROJECT="${WANDB_PROJECT:-effortless-transformer}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$SCRATCH_ROOT/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$SCRATCH_ROOT/.cache/wandb}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${SLURM_JOB_NAME:-local}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$HF_HOME" "$REPO_DIR/logs"

# Whether a W&B credential is reachable (WANDB_API_KEY, or a ~/.netrc entry from
# `wandb login`). Lets an online run warn up front if it is about to go anonymous.
wandb_credentials_present() {
    [ -n "${WANDB_API_KEY:-}" ] && return 0
    local netrc="${NETRC:-$HOME/.netrc}"
    [ -f "$netrc" ] && grep -q "api.wandb.ai" "$netrc" 2>/dev/null
}

report_env() {
    echo "=================================================================="
    echo "Job        ${SLURM_JOB_NAME:-<none>} (${SLURM_JOB_ID:-<none>})"
    echo "Node       ${SLURMD_NODENAME:-<none>}"
    echo "GPUs       ${CUDA_VISIBLE_DEVICES:-<none>}  (count ${N_GPUS:-?})"
    echo "Repo       $REPO_DIR"
    echo "Python     ${PY[*]}"
    echo "Started    $(timestamp)"
    echo "wandb      project=$WANDB_PROJECT mode=$WANDB_MODE id=$WANDB_RUN_ID"
    if [ "$WANDB_MODE" = "online" ] && ! wandb_credentials_present; then
        echo "wandb      WARNING: online mode but no credential; run 'wandb login'"
        echo "           or submit with WANDB_MODE=offline and sync afterwards."
    fi
    echo "=================================================================="
}

# GPUs visible to this job: Slurm reports it under --gres, nvidia-smi is the
# fallback for interactive testing.
detect_gpus() {
    if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
        echo "$SLURM_GPUS_ON_NODE"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
    else
        echo 0
    fi
}

# Newest step-*.pt in a directory, or empty. A glob (not `ls | sort -V`) so it
# does not fail under `set -e` when a first run's directory is still empty.
latest_checkpoint() {
    local dir="$1" newest="" best=-1 file base step
    shopt -s nullglob
    for file in "$dir"/step-*.pt; do
        base="$(basename "$file")"
        step="${base#step-}"; step="${step%.pt}"
        step=$((10#$step))  # base ten: step-000020.pt is not octal
        if [ "$step" -gt "$best" ]; then best="$step"; newest="$file"; fi
    done
    shopt -u nullglob
    printf '%s' "$newest"
}

# ISO-8601 timestamp. `date +%...` rather than `date -Is`, which is GNU-only.
timestamp() {
    date +%Y-%m-%dT%H:%M:%S%z
}

# Report a nonzero exit loudly, including from Slurm's accounting, since a job
# that dies before its first flush leaves a log that simply stops.
# Install with:  trap report_failure EXIT
report_failure() {
    local status=$?
    [ "$status" -eq 0 ] && return 0
    echo
    echo "=================================================================="
    echo "FAILED with status $status"
    case "$status" in
        127) echo "  command-not-found; usually a stale checkout. Try 'git pull'." ;;
        137) echo "  SIGKILL, almost always the cgroup OOM killer (host RAM)." ;;
        139) echo "  segmentation fault, usually a native library." ;;
        143) echo "  SIGTERM: the wall clock ran out, or scancel." ;;
    esac
    sacct -j "${SLURM_JOB_ID:-0}" \
        -o JobID%20,JobName%14,State%22,ExitCode,MaxRSS,Elapsed 2>/dev/null \
        || echo "  (sacct unavailable)"
    echo "=================================================================="
}
