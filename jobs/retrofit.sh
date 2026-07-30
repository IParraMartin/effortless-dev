#!/bin/bash
# Experiments A1 and A2 from ../START_HERE.md, in one job.
#
# Usage:
#   sbatch --job-name=vr-a1 jobs/retrofit.sh            # A1 then A2
#   sbatch --job-name=vr-a1 jobs/retrofit.sh build      # A1 build only, ~seconds
#   sbatch --job-name=vr-a1 jobs/retrofit.sh train      # A1 exit training
#   sbatch --job-name=vr-a2 jobs/retrofit.sh route      # A2 only
#
# Arguments:
#   $1   stage    all (default) | build | train | route
#   $2+  extra    passed through to the stage's command
#
# What this answers
# -----------------
#
# **A1 — no regret, and useful tiers.** Freeze the final-only checkpoint, attach
# a zero-initialized adapter and readout to layers 2/4/6/8/10, train only those.
# The build step prints `max logit difference 0.000e+00`: the parent is
# bit-identical, so the no-regret claim is verified rather than argued. Training
# touches no backbone weight, so it stays verified.
#
# What A1 measures is whether a *normal* model's intermediate states are
# decodable at all. If they are, the sharing question that the two 2026-07-27
# runs were built to answer stops mattering, because a frozen parent pays no
# sharing tax.
#
# **A2 — learnable adaptivity.** Collect per-request, per-depth labels on real
# held-out text, fit the controller over three seeds, evaluate. Read
# `probe-policy gain`, not `outcome oracle - best fixed`.
#
# Kill conditions, from START_HERE.md
# -----------------------------------
#
#   A1  shallow tiers no better than chance  -> the parent's intermediate states
#       carry nothing, and the retrofit framing fails
#   A2  probe-policy gain around zero        -> requests do not differ in the
#       depth they need, and there is nothing to route on
#
# Either sends the project back to trained exits, which reinstates
# jobs/controlled_arms.sh and jobs/prepare_pile.sh.
#
#SBATCH --job-name=vr-retrofit
#SBATCH --account=fc_bsclab
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=iparra@berkeley.edu
set -euo pipefail

STAGE="${1:-all}"
shift || true
EXTRA_FLAGS=("$@")

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cd "$REPO_DIR"
report_env

PARENT="${PARENT:-$SCRATCH_ROOT/checkpoints/vr-noexits/final.pt}"
DATA_DIR="${DATA_DIR:-$SCRATCH_ROOT/data}"
RUN_DIR="${RUN_DIR:-$SCRATCH_ROOT/runs/retrofit-adapter}"
OUT_DIR="${OUT_DIR:-$SCRATCH_ROOT/checkpoints/retrofit-adapter}"
RESULTS="${RESULTS:-$SCRATCH_ROOT/results/a2}"
RANK="${RANK:-32}"
EXIT_EVERY="${EXIT_EVERY:-2}"
STEPS="${STEPS:-4000}"

# GPT-2's end-of-text token. Document boundaries come from it, and without it
# every interval the evaluation reports is unclustered and too narrow.
EOS_ID="${EOS_ID:-50256}"

if [ ! -f "$PARENT" ]; then
    echo "error: parent checkpoint $PARENT not found." >&2
    echo "Set PARENT=/path/to/final.pt, or check $SCRATCH_ROOT/checkpoints/" >&2
    exit 1
fi

echo "parent:  $PARENT"
echo "corpus:  $DATA_DIR/val.bin  (eos $EOS_ID)"
echo "stage:   $STAGE"
echo

# ---------------------------------------------------------------- A1: build
if [ "$STAGE" = "all" ] || [ "$STAGE" = "build" ]; then
    echo "== A1 build: attach exits to a frozen parent =="
    "${PY[@]}" -m experiments.retrofit_parent \
        --checkpoint="$PARENT" \
        --run-dir="$RUN_DIR" \
        --mode=frozen_exit_adapter \
        --exit_adapter_rank="$RANK" \
        --exit_every="$EXIT_EVERY" \
        --device=cuda \
        "${EXTRA_FLAGS[@]}"
    echo
fi

# ---------------------------------------------------------------- A1: train
if [ "$STAGE" = "all" ] || [ "$STAGE" = "train" ]; then
    RETROFIT="$RUN_DIR/checkpoints/retrofit.pt"
    if [ ! -f "$RETROFIT" ]; then
        echo "error: $RETROFIT not found; run the build stage first." >&2
        exit 1
    fi

    echo "== A1 train: exits only, no backbone weight moves =="
    # anchored_v1 explicitly. The default is still legacy_normalized, which
    # would give the final endpoint 12/42 of the hard-target weight -- and here
    # the final endpoint is the frozen parent's, so the weighting would be
    # applied to something that cannot move anyway. Stating it keeps the run
    # record honest about which objective produced the exits.
    "${PY[@]}" -m training.train \
        --resume_from="$RETROFIT" \
        --objective_version=anchored_v1 \
        --shallow_loss_weight="${ALPHA:-0.5}" \
        --distill_weight=0.5 \
        --data_dir="$DATA_DIR" \
        --out_dir="$OUT_DIR" \
        --max_steps="$STEPS" \
        --seq_len=1024 --batch_size=8 --grad_accum_steps=8 \
        --dtype=bf16 --eval_every=250 --save_every=1000 \
        --exits_per_step=2 --find_unused_parameters=true \
        --grad_diagnostics_every=1000 \
        --wandb_project="$WANDB_PROJECT" \
        "${EXTRA_FLAGS[@]}"
    echo

    echo "== A1 check: is the parent still bit-identical after training? =="
    # Expected to pass trivially -- no backbone parameter was trainable -- which
    # is exactly why it is worth asserting rather than assuming.
    "${PY[@]}" -m experiments.no_regret \
        --parent="$PARENT" \
        --candidate="$OUT_DIR/final.pt" \
        --data="$DATA_DIR/val.bin" \
        --eos_id="$EOS_ID" \
        --n_requests=1024 \
        --quality_margin=0.01 \
        --run-dir="$SCRATCH_ROOT/runs/no-regret-adapter"
    echo
fi

# ---------------------------------------------------------------- A2: route
if [ "$STAGE" = "all" ] || [ "$STAGE" = "route" ]; then
    CANDIDATE="${CANDIDATE:-$OUT_DIR/final.pt}"
    if [ ! -f "$CANDIDATE" ]; then
        echo "error: $CANDIDATE not found; run the train stage first." >&2
        exit 1
    fi

    echo "== A2: trajectories on real text, controller, evaluation =="
    # --corpus real_text is not optional here. The default is synthetic, whose
    # depth structure is a rule the experimenter installed; a controller result
    # on it says the machinery works and nothing about language.
    "${PY[@]}" -m experiments.collect_depth_trajectories \
        --corpus=real_text \
        --data="$DATA_DIR/val.bin" \
        --eos_id="$EOS_ID" \
        --checkpoint="$CANDIDATE" \
        --out="$RESULTS/trajectories" \
        --probe_depth="${PROBE_DEPTH:-2}" \
        --n_requests="${N_REQUESTS:-4096}" \
        --shapes="${SHAPES:-64:32,128:64,256:128}" \
        --max_new_tokens="${MAX_NEW_TOKENS:-16}" \
        --cache_dtype=bf16

    "${PY[@]}" -m experiments.train_depth_controller \
        --trajectories="$RESULTS/trajectories" \
        --out="$RESULTS/controller" \
        --quality_metric="${QUALITY_METRIC:-bits_per_byte}" \
        --routing_lambda="${ROUTING_LAMBDA:-0.2}" \
        --seeds 0 1 2

    "${PY[@]}" -m experiments.evaluate_vertical_routing \
        --trajectories="$RESULTS/trajectories" \
        --controller="$RESULTS/controller" \
        --controller_seed=0 \
        --quality_metric="${QUALITY_METRIC:-bits_per_byte}" \
        --out="$RESULTS/evaluation" \
        ${MANIFEST:+--manifest="$MANIFEST"}

    echo
    echo "Read the evaluation:"
    echo "  $RESULTS/evaluation/evaluation.md"
    echo
    echo "The go/no-go is the **probe-policy gain** column, not"
    echo "'outcome oracle - best fixed'. If it is near zero, request-level"
    echo "routing has no case and no amount of controller work will create one."
fi
