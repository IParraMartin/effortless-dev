#!/bin/bash
# One-shot environment setup on Savio. Run on a LOGIN node, not under sbatch.
#
#   bash jobs/setup_env.sh
#
# Installs the project into .venv with uv, keeping caches on scratch so the
# 10 GB home quota is not consumed, and runs the test suite as a smoke check.
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

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

cd "$REPO_DIR"
mkdir -p logs data checkpoints

echo "Syncing environment with uv (caches on scratch)..."
uv sync

echo
echo "Versions:"
uv run python - <<'PY'
import importlib, torch
print(f"  torch          {torch.__version__}")
print(f"  cuda available {torch.cuda.is_available()}")
for name in ("transformers", "datasets", "numpy", "wandb"):
    print(f"  {name:<14} {importlib.import_module(name).__version__}")
PY

echo
echo "Running the test suite (a few seconds, CPU only):"
uv run python -m unittest discover -s tests -t . 2>&1 | tail -4

cat <<EOF

Setup complete.

  1. Tokenize a corpus   sbatch --job-name=data jobs/prepare_data.sh
  2. Train the model     sbatch --job-name=lm   jobs/train.sh

Runs stream to Weights & Biases online. Log in once on this login node so the
compute nodes find the credential in ~/.netrc:

  uv run wandb login

If you submit with WANDB_MODE=offline, push the runs later with:

  bash jobs/sync_wandb.sh
EOF
