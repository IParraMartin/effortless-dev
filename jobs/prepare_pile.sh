#!/bin/bash
# Tokenize the Pile with the GPT-NeoX tokenizer, matching Pythia's inputs.
#
# Usage:
#   sbatch --job-name=pile-prep jobs/prepare_pile.sh
#   sbatch --job-name=pile-prep jobs/prepare_pile.sh 400000   # cap documents
#
# Arguments:
#   $1   max_train_docs   document cap, or "none" for the whole split
#
# Why this corpus and this tokenizer, rather than the FineWeb-Edu data already
# on disk: the horizontal comparison is against Pythia, and a difference between
# a FineWeb-Edu-trained backbone and a Pile-trained Pythia confounds capacity
# sharing with training corpus. Bits per byte removes the *tokenizer* from the
# comparison; it cannot remove the corpus. Matching both leaves capacity as the
# only thing that differs.
#
# Budget is matched separately, on the Pythia side, by scoring an intermediate
# revision rather than the final checkpoint. See jobs/controlled_arms.sh.
#
#SBATCH --job-name=pile-prep
#SBATCH --account=fc_bsclab
#SBATCH --partition=savio3
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=iparra@berkeley.edu
set -euo pipefail

MAX_DOCS="${1:-none}"

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
trap report_failure EXIT
cd "$REPO_DIR"
report_env

# monology/pile-uncopyrighted is the maintained mirror; EleutherAI/pile itself is
# no longer served. The tokenizer is the one Pythia trained with -- taken from a
# Pythia checkpoint rather than from gpt-neox-20b, so it is the same artefact the
# comparison models used and not merely the same family.
DATA_DIR="${DATA_DIR:-$SCRATCH_ROOT/data/pile-neox}"

echo "corpus:    monology/pile-uncopyrighted"
echo "tokenizer: EleutherAI/pythia-160m (GPT-NeoX BPE)"
echo "out:       $DATA_DIR"
echo "doc cap:   $MAX_DOCS"
echo

"${PY[@]}" -m training.data \
    --dataset_name=monology/pile-uncopyrighted \
    --dataset_config=none \
    --text_column=text \
    --streaming=true \
    --tokenizer_name=EleutherAI/pythia-160m \
    --data_dir="$DATA_DIR" \
    --seq_len=1024 \
    --max_train_docs="$MAX_DOCS"

echo
echo "Wrote:"
ls -la "$DATA_DIR"
echo
echo "The end-of-text id for this tokenizer is what --eos_id needs later:"
"${PY[@]}" - <<'PY'
from src.tokenizer import load_tokenizer

tokenizer = load_tokenizer("EleutherAI/pythia-160m")
print(f"  eos_id = {tokenizer.eos_token_id}  (vocab {len(tokenizer)})")
PY
