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
# **Internet.** Savio compute nodes cannot reliably reach the Hugging Face hub.
# If this job dies resolving huggingface.co, pre-download on a login node — the
# cache is on scratch and shared, so the compute node then finds it locally:
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

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$REPO_DIR"
mkdir -p logs data

N_GPUS=0
report_env
echo "dataset=$DATASET config=$CONFIG column=$COLUMN"

# Streaming keeps peak memory flat: tokens are appended to disk as they are
# produced rather than assembled in RAM, so a corpus far larger than the node
# can be prepared.
uv run python -m training.data \
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
