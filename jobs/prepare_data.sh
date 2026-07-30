#!/bin/bash
# Tokenize a corpus into the packed uint16/uint32 memmaps training reads.
#
# Usage:
#   sbatch --job-name=data jobs/prepare_data.sh
#   sbatch --job-name=data jobs/prepare_data.sh HuggingFaceFW/fineweb-edu sample-10BT
#
# Arguments:  $1 dataset  $2 config  $3 text column   (defaults below)
#
# No GPU: tokenization is CPU and I/O bound. Compute nodes have outbound access,
# so the hub download normally just works; a path to a local .jsonl/.parquet/
# .csv/.txt file works with no network at all.
#
#SBATCH --job-name=data
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

_find_env() {
    local candidate
    for candidate in "${SLURM_SUBMIT_DIR:-}/jobs" "${SLURM_SUBMIT_DIR:-}" \
                     "$(dirname "${BASH_SOURCE[0]}")" "$(pwd)/jobs" "$(pwd)"; do
        [ -n "$candidate" ] && [ -f "$candidate/_env.sh" ] && {
            printf '%s' "$candidate/_env.sh"; return 0; }
    done
    echo "Cannot find jobs/_env.sh. Submit from the repo root." >&2
    return 1
}
source "$(_find_env)"
trap report_failure EXIT
cd "$REPO_DIR"
mkdir -p data

N_GPUS=0
report_env
echo "dataset=$DATASET config=$CONFIG column=$COLUMN"

# Streaming keeps peak memory flat: tokens are appended to disk as produced
# rather than assembled in RAM, so a corpus larger than the node can be prepared.
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
