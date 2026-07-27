#!/bin/bash
# Verify Weights & Biases will actually log, before spending a GPU allocation.
#
#   bash   jobs/check_wandb.sh      # on a login node
#   sbatch jobs/check_wandb.sh      # from a compute node -- the real question
#
# Submit it. A login node proves nothing about whether the node your 24-hour
# training job lands on can reach the service, and that is the failure that
# costs something.
#
# Note that `wandb status` does NOT answer this. It prints the contents of the
# settings file, so it reports `"api_key": null` on a machine that is perfectly
# well logged in through ~/.netrc. Checking it would give a false negative. This
# script resolves the effective credential and round-trips to the server
# instead, then creates and finishes a real run end to end.
#
#SBATCH --job-name=vr-wandb-check
#SBATCH --account=fc_bsclab
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$REPO_DIR"
mkdir -p logs

N_GPUS="$(detect_gpus)"
report_env

WANDB_RUN_ID="wandb-check-${SLURM_JOB_ID:-local}" \
uv run python - <<'PY'
import os
import sys

import wandb

print(f"mode            {os.environ.get('WANDB_MODE', 'online')}")
print(f"project         {os.environ.get('WANDB_PROJECT')}")
print(f"netrc           {os.environ.get('NETRC', os.path.expanduser('~/.netrc'))}")

key = wandb.api.api_key
if not key:
    print("\nFAIL: no API key. Run 'uv run wandb login' on a login node, or")
    print("      export WANDB_API_KEY. Credentials live in ~/.netrc, which is")
    print("      shared with compute nodes.")
    sys.exit(1)
print(f"api key         resolved (...{key[-4:]})")

# Round trip to the server. This is the part a login node cannot answer for a
# compute node, and the part that actually decides whether training will log.
try:
    viewer = wandb.api.viewer()
    print(f"authenticated   yes, as '{viewer.get('entity')}'")
except Exception as error:
    print(f"\nFAIL: reached for the server and could not: "
          f"{type(error).__name__}: {error}")
    print("      This node cannot talk to W&B. Submit training with")
    print("      --export=ALL,WANDB_MODE=offline and run jobs/sync_wandb.sh")
    print("      from a login node afterwards.")
    sys.exit(1)

# End to end: a real run, a real metric, a clean finish.
run = wandb.init(job_type="connectivity-check")
wandb.log({"check/ok": 1.0})
url = run.url
run.finish()

print(f"\nPASS: created and finished a run.")
print(f"      {url or '(offline, nothing uploaded)'}")
PY

echo
echo "If this passed from a compute node, jobs/train.sh will log online."
