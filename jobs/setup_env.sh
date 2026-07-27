#!/bin/bash
# One-shot environment setup on Savio. Run on a LOGIN node, not under sbatch.
#
#   bash jobs/setup_env.sh
#
# Installs the project into .venv with uv, keeping every cache on scratch so the
# 10 GB home quota is not consumed by wheels and interpreters, and checks that
# the pieces the jobs depend on are actually present.
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

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

cd "$REPO_DIR"
mkdir -p logs data checkpoints results

echo "Syncing environment with uv (caches on scratch)..."
uv sync

echo
echo "Checking the pieces the jobs rely on:"
uv run python - <<'PY'
import importlib, torch
print(f"  torch          {torch.__version__}")
print(f"  cuda available {torch.cuda.is_available()}"
      f"{' (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else ''}")
print(f"  bf16 supported {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'n/a'}")
for name in ("wandb", "transformers", "datasets", "numpy"):
    print(f"  {name:<14} {importlib.import_module(name).__version__}")
PY

echo
echo "Running the test suite (a few seconds, CPU only):"
uv run python -m unittest discover -s tests -t . 2>&1 | tail -4

cat <<EOF

Setup complete.

  1. Tokenize a corpus     sbatch --job-name=vr-data    jobs/prepare_data.sh
  2. Train the backbone    sbatch --job-name=vr-exits   jobs/train.sh exits
     ...and its control    sbatch --job-name=vr-noexits jobs/train.sh noexits
  3. Route and evaluate    sbatch --job-name=vr-route   jobs/route.sh \\
                               checkpoints/vr-exits/final.pt
  4. Push W&B runs         bash jobs/sync_wandb.sh

Runs stream to Weights & Biases online. Log in once on this login node so the
compute nodes find the credential in ~/.netrc:

  uv run wandb login

Step 4 is only needed for runs submitted with WANDB_MODE=offline.

EOF
