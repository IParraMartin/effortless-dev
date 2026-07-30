#!/bin/bash
# Train the decoder-only Transformer on a tokenized corpus.
#
# Usage:
#   sbatch --job-name=lm jobs/train.sh
#   sbatch --job-name=lm --gres=gpu:A40:4 --cpus-per-task=32 jobs/train.sh
#   sbatch --export=ALL,MAX_STEPS=5000 --job-name=lm-short jobs/train.sh
#
# Any extra arguments are passed straight through to training.train, e.g.
#   sbatch --job-name=lm jobs/train.sh --d_model=1024 --n_layers=24
#
#SBATCH --job-name=lm
#SBATCH --account=fc_bsclab
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=iparra@berkeley.edu
set -euo pipefail

EXTRA_FLAGS=("$@")

# Slurm copies the batch script off the repo, so find _env.sh via SLURM_SUBMIT_DIR
# (where sbatch was invoked) as well as the usual sibling locations.
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

N_GPUS="$(detect_gpus)"
[ "$N_GPUS" -ge 1 ] || { echo "No GPU visible; submit with --gres=gpu:A40:N"; exit 1; }
report_env

# ~124M parameters: 12 layers, 768 wide, 12 heads.
N_LAYERS=12
D_MODEL=768
N_HEADS=12
SEQ_LEN=1024
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_STEPS="${MAX_STEPS:-20000}"

OUT_DIR="${OUT_DIR:-$REPO_DIR/checkpoints/${SLURM_JOB_NAME:-lm}}"
mkdir -p "$OUT_DIR"

# Automatic resume: a 72-hour wall clock is shorter than some runs, and a
# requeue that restarted from step zero would waste the whole budget.
RESUME_FLAGS=()
RESUME_FROM="$(latest_checkpoint "$OUT_DIR")"
if [ -n "$RESUME_FROM" ]; then
    echo "Resuming from $RESUME_FROM"
    RESUME_FLAGS=(--resume_from="$RESUME_FROM")
fi

# torchrun rendezvous pinned to IPv4 (--standalone stalls on IPv6-first hosts);
# the port is derived from the job id so two jobs on one node do not collide.
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 20000))

echo "gpus=$N_GPUS batch=$BATCH_SIZE x accum=$GRAD_ACCUM over $MAX_STEPS steps"
echo "tokens/step = $((BATCH_SIZE * GRAD_ACCUM * SEQ_LEN * N_GPUS))"

"${PY[@]}" -m torch.distributed.run \
    --nnodes=1 --nproc_per_node="$N_GPUS" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    -m training.train \
    --data_dir="$REPO_DIR/data" \
    --tokenizer_name=gpt2 \
    --seq_len="$SEQ_LEN" \
    --n_layers="$N_LAYERS" \
    --d_model="$D_MODEL" \
    --n_heads="$N_HEADS" \
    --batch_size="$BATCH_SIZE" \
    --grad_accum_steps="$GRAD_ACCUM" \
    --max_steps="$MAX_STEPS" \
    --learning_rate=3e-4 \
    --min_lr=3e-5 \
    --warmup_steps=500 \
    --dtype=bf16 \
    --compile_model=false \
    --num_workers=4 \
    --ddp_backend=nccl \
    --eval_every=500 \
    --eval_steps=50 \
    --save_every=1000 \
    --log_every=20 \
    --out_dir="$OUT_DIR" \
    --wandb_project="$WANDB_PROJECT" \
    --wandb_run_name="${SLURM_JOB_NAME:-lm}" \
    --wandb_mode="$WANDB_MODE" \
    ${RESUME_FLAGS[@]+"${RESUME_FLAGS[@]}"} \
    ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}

echo "Finished $(timestamp). Checkpoints in $OUT_DIR"
