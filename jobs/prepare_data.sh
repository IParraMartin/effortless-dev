#!/bin/bash
# Tokenize a corpus into the packed uint16/uint32 memmaps training reads.
#
# Usage:
#   sbatch --job-name=vr-data jobs/prepare_data.sh
#   sbatch --job-name=vr-data jobs/prepare_data.sh HuggingFaceFW/fineweb-edu sample-10BT
#
# Arguments:
#   $1  dataset name    (default: HuggingFaceFW/fineweb-edu)
#   $2  dataset config  (default: sample-10BT)
#   $3  text column     (default: text)
#
# No GPU: this is tokenization, which is CPU and I/O bound.
#
# **Internet.** Compute nodes do have outbound access — a W&B connectivity check
# uploaded from one — so the hub download normally just works. Different host,
# though, so if this job dies resolving huggingface.co, pre-download on a login
# node; the cache is on scratch and shared, so the compute node finds it
# locally afterwards:
#
#   source jobs/_env.sh
#   uv run python -c "import datasets; datasets.load_dataset( \
#       'HuggingFaceFW/fineweb-edu', 'sample-10BT', split='train')"
#
# Local files work too and need no network at all; training/data.py accepts a
# path to .jsonl/.parquet/.csv/.txt (and globs) in place of a repository id.
#
#SBATCH --job-name=vr-data
#SBATCH --account=fc_bsclab
#SBATCH --partition=savio3_bigmem
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=iparra@berkeley.edu
set -euo pipefail

DATASET="${1:-HuggingFaceFW/fineweb-edu}"
CONFIG="${2:-sample-10BT}"
COLUMN="${3:-text}"

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
trap report_failure EXIT
cd "$REPO_DIR"
mkdir -p logs data

N_GPUS=0
report_env
echo "dataset=$DATASET config=$CONFIG column=$COLUMN"

# Streaming keeps peak memory flat: tokens are appended to disk as they are
# produced rather than assembled in RAM, so a corpus far larger than the node
# can be prepared.
"${PY[@]}" -m training.data \
    --dataset_name="$DATASET" \
    --dataset_config="$CONFIG" \
    --text_column="$COLUMN" \
    --tokenizer_name=gpt2 \
    --streaming=true \
    --max_train_docs="${MAX_TRAIN_DOCS:-4000000}" \
    --data_dir="$REPO_DIR/data"

echo
echo "Wrote:"
ls -lh data/*.bin data/*.meta.json
echo
echo "The sidecar records the element type, so a vocabulary beyond 65536 costs"
echo "disk rather than failing, and a reader never has to guess the width."
