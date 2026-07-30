#!/bin/bash
# Shared environment for every Savio job in this directory. Sourced, not run.
#
# Home is a 10 GB quota. uv's package cache, uv's managed interpreters, the
# Hugging Face cache and Weights & Biases run data all default to home and all
# of them will fill it, so each is redirected to scratch.
#
# Weights & Biases logs **online**: Savio compute nodes do reach the service,
# confirmed on this account. WANDB_CONFIG_DIR is deliberately *not* redirected —
# it is a small settings file, and moving it risks disturbing a `wandb login`
# that already works, which is not a trade worth making to save kilobytes. The
# credentials themselves live in ~/.netrc and are untouched either way.
#
# WANDB_MODE=offline remains available and jobs/sync_wandb.sh pushes afterwards.
# Worth reaching for if a node turns out to be firewalled or the service is
# unreachable mid-run; offline recording cannot fail, it just defers.

# Not `set -e` here: this file is sourced, and killing the caller's shell on a
# non-zero probe would be surprising.
set -uo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/global/scratch/users/iparra}"
REPO_DIR="${REPO_DIR:-$SCRATCH_ROOT/effortless-dev}"

# uv installs to ~/.local/bin, which a non-interactive batch shell will not have
# on its PATH.
export PATH="$HOME/.local/bin:$PATH"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH_ROOT/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SCRATCH_ROOT/.local/uv-python}"
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

# Unbuffered Python. Not a preference — it is what makes a failed job
# diagnosable. Writing to a file rather than a terminal, Python block-buffers
# stdout in 8 KB chunks, so a process killed before it fills the buffer loses
# everything it printed. A run that died during model construction and one that
# never started produce byte-identical empty logs, and there is nothing left to
# tell them apart. Costs a syscall per line, against a job that logs every
# twentieth step.
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------- interpreter
# Call the virtual environment's interpreter directly rather than going through
# `uv run`, which is not free here: it re-resolves the project and takes a lock
# on .venv before it will execute anything, and .venv lives on scratch. Measured
# on a login node, `uv run python -c "import torch"` took 94 seconds wall for 7
# seconds of CPU -- almost all of it blocked on the filesystem, and that was one
# command. A job pays it at every invocation, and two jobs sharing one .venv
# contend for the same lock; that is how vr-exits spent 24 minutes at 63 MB of
# resident memory and exited before Python ever started.
#
# Nothing is lost by skipping it. The environment is already built by
# jobs/setup_env.sh, so there is nothing to resolve at job start, and naming the
# interpreter is what `uv run` would have arrived at anyway. `uv` remains the
# right tool for *changing* the environment -- just not for entering it.
#
# The fallback keeps this working before setup_env.sh has ever run.
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
if [ -x "$VENV_DIR/bin/python" ]; then
    PY=("$VENV_DIR/bin/python")
else
    PY=(uv run python)
fi
export VENV_DIR

# ---------------------------------------------------------------- Weights & Biases
export WANDB_PROJECT="${WANDB_PROJECT:-effortless-vertical-routing}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$SCRATCH_ROOT/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$SCRATCH_ROOT/.cache/wandb}"

# A stable run id keyed on the *experiment*, not on the Slurm job id. A requeued
# or resumed job gets a new job id, and keying on that would split one training
# run across several W&B runs with the step counter restarting inside each.
export WANDB_RUN_ID="${WANDB_RUN_ID:-${SLURM_JOB_NAME:-local}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$HF_HOME"
# Slurm writes --output=logs/%x_%j.out before the job body runs, so a missing
# logs/ makes the job fail with no log at all -- the least diagnosable failure
# there is. setup_env.sh creates it, but a fresh clone or a moved REPO_DIR would
# not have run that.
mkdir -p "$REPO_DIR/logs"

# ---------------------------------------------------------------- reporting
report_env() {
    echo "=================================================================="
    echo "Job        ${SLURM_JOB_NAME:-<none>} (${SLURM_JOB_ID:-<none>})"
    echo "Node       ${SLURMD_NODENAME:-<none>}"
    echo "GPUs       ${CUDA_VISIBLE_DEVICES:-<none>}  (count ${N_GPUS:-?})"
    echo "Repo       $REPO_DIR"
    # Which interpreter actually ran. Worth a line: these jobs use the project's
    # .venv and ignore an active conda environment, so a log that does not say
    # which one it used leaves "wrong environment" and "real bug" looking
    # identical when something imports differently than expected.
    #
    # Printed, not executed. This used to run `uv run python -c` to ask the
    # interpreter for its own path, which meant the banner itself blocked on the
    # scratch filesystem for a minute and a half before the job could start.
    echo "Python     ${PY[*]}"
    [ -n "${CONDA_PREFIX:-}" ] && echo "           (conda env $CONDA_PREFIX is active but unused)"
    echo "Started    $(timestamp)"
    echo "wandb      project=$WANDB_PROJECT mode=$WANDB_MODE id=$WANDB_RUN_ID"
    if [ "$WANDB_MODE" = "online" ] && ! wandb_credentials_present; then
        echo "wandb      WARNING: online mode, but no WANDB_API_KEY and no"
        echo "           api.wandb.ai entry in ${NETRC:-$HOME/.netrc}. The run"
        echo "           will go anonymous or fail. Fix with 'wandb login', or"
        echo "           submit with WANDB_MODE=offline and sync afterwards."
    fi
    echo "=================================================================="
}

# Whether a Weights & Biases credential is reachable. `wandb login` writes an
# api.wandb.ai entry to ~/.netrc; WANDB_API_KEY overrides it. Checked so an
# online run that is about to go anonymous says so at the top of the log rather
# than at the end of training.
wandb_credentials_present() {
    [ -n "${WANDB_API_KEY:-}" ] && return 0
    local netrc="${NETRC:-$HOME/.netrc}"
    [ -f "$netrc" ] && grep -q "api.wandb.ai" "$netrc" 2>/dev/null
}

# Number of GPUs visible to this job. Slurm reports it when --gres is used;
# nvidia-smi is the fallback for interactive testing.
detect_gpus() {
    if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
        echo "$SLURM_GPUS_ON_NODE"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
    else
        echo 0
    fi
}

# Most recent checkpoint in a directory, or empty. Used for automatic resume, so
# a requeued job continues instead of silently restarting from step zero.
#
# Written with a glob rather than `ls | sort -V | tail -1` for two reasons, one
# of which is a bug rather than a preference. Under `set -euo pipefail` the `ls`
# fails when the directory holds no checkpoints yet — which is the state of
# *every* first run — the failure propagates through pipefail, and the job dies
# before training starts. The second reason is that `sort -V` is GNU-only, so
# the pipeline was not testable outside Linux either.
latest_checkpoint() {
    local dir="$1" newest="" best=-1 file base step
    shopt -s nullglob
    for file in "$dir"/step-*.pt; do
        base="$(basename "$file")"
        step="${base#step-}"
        step="${step%.pt}"
        # 10# forces base ten: step-000020.pt would otherwise be read as octal.
        step=$((10#$step))
        if [ "$step" -gt "$best" ]; then
            best="$step"
            newest="$file"
        fi
    done
    shopt -u nullglob
    printf '%s' "$newest"
}

# ISO-8601 timestamp. `date -Is` is GNU-only and errors on BSD/macOS, which
# makes these scripts untestable off the cluster for no benefit.
timestamp() {
    date +%Y-%m-%dT%H:%M:%S%z
}


# Report a nonzero exit loudly, and from Slurm's accounting as well as the log.
#
# A job that dies before its first flush leaves a log that simply stops, which is
# indistinguishable from a hung job. train.sh grew this after a run failed with an
# empty .err file; every job script should have it, so it lives here rather than
# being copied per script.
#
# Install with:  trap report_failure EXIT
report_failure() {
    local status=$?
    [ "$status" -eq 0 ] && return 0
    echo
    echo "=================================================================="
    echo "FAILED with status $status"
    case "$status" in
        1)   echo "  1 is a plain error. Read upward for the first message; with"
             echo "  'set -e' the job stops at the first failing command." ;;
        127) echo "  127 is command-not-found. Usually a stale checkout: the"
             echo "  script references a module or flag that this commit lacks."
             echo "  Try 'git pull' in $REPO_DIR." ;;
        137) echo "  137 is SIGKILL. On this cluster that is almost always the"
             echo "  cgroup OOM killer, meaning host RAM, not GPU memory."
             echo "  A CUDA OOM raises a Python exception and leaves a traceback." ;;
        139) echo "  139 is a segmentation fault, usually a native library." ;;
        143) echo "  143 is SIGTERM: the wall clock ran out, or scancel." ;;
    esac
    echo "Slurm's own accounting, which survives when the log does not:"
    sacct -j "${SLURM_JOB_ID:-0}" \
        -o JobID%20,JobName%14,State%22,ExitCode,MaxRSS,Elapsed 2>/dev/null \
        || echo "  (sacct unavailable)"
    echo "=================================================================="
}
