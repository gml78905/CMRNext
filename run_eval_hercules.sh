#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="${DATA_ROOT:-/workspace/data/LG_Innotek/PublicDataset/hercules}"
SAVE_ROOT="${SAVE_ROOT:-/workspace/data/checkpoints/CMRNext/hercules}"
RUN_NAME="${RUN_NAME:-hercules_radar}"
VAL_SCENE="${VAL_SCENE:-library_1}"

MAX_T="${MAX_T:-1.5}"
MAX_R="${MAX_R:-20.0}"
NUM_WORKER="${NUM_WORKER:-2}"
SEED="${SEED:-42}"
GPU_IDS="${GPU_IDS:-0}"
SAVE_FILE="${SAVE_FILE:-${RUN_NAME}_val_${VAL_SCENE}_eval}"
DRY_RUN="${DRY_RUN:-false}"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1

if [ -z "${WEIGHTS:-}" ]; then
    CHECKPOINT_DIR="$SAVE_ROOT/$RUN_NAME/remove"
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "Checkpoint directory not found: $CHECKPOINT_DIR" >&2
        echo "Set WEIGHTS=/path/to/checkpoint.tar or check SAVE_ROOT/RUN_NAME." >&2
        exit 1
    fi

    WEIGHTS="$(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'checkpoint_*.tar' | sort -V | tail -n 1)"
    if [ -z "$WEIGHTS" ]; then
        echo "No checkpoint_*.tar found in $CHECKPOINT_DIR" >&2
        echo "Set WEIGHTS=/path/to/checkpoint.tar explicitly." >&2
        exit 1
    fi
fi

echo "============================================================"
echo "Evaluating $RUN_NAME"
echo "  data root : $DATA_ROOT"
echo "  weights   : $WEIGHTS"
echo "  val scene : $VAL_SCENE"
echo "  max_t/max_r : $MAX_T / $MAX_R"
echo "  gpu ids   : $GPU_IDS"
echo "  workers   : $NUM_WORKER"
echo "  save file : ${SAVE_FILE}_errors_[t|r].torch"
echo "============================================================"

cmd=(
    python3 evaluate_flow_calibration.py
    --dataset hercules
    --data_folder "$DATA_ROOT"
    --weights "$WEIGHTS"
    --max_t "$MAX_T"
    --max_r "$MAX_R"
    --num_worker "$NUM_WORKER"
    --val_scene "$VAL_SCENE"
    --seed "$SEED"
    --save_file "$SAVE_FILE"
)

if [ "$DRY_RUN" = "true" ]; then
    printf 'DRY RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

"${cmd[@]}"
