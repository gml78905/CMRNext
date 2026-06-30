#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="${DATA_ROOT:-/workspace/data/LG_Innotek/PublicDataset/hercules}"
SAVE_ROOT="${SAVE_ROOT:-/workspace/data/checkpoints/CMRNext/hercules}"
RESULTS_DIR="${RESULTS_DIR:-$SAVE_ROOT/results}"

STAGE1_RUN="${STAGE1_RUN:-hercules_radar}"
STAGE2_RUN="${STAGE2_RUN:-hercules_radar2}"
STAGE3_RUN="${STAGE3_RUN:-hercules_radar3}"

VAL_SCENES_DEFAULT="library_3 parking_lot_1 parking_lot_2 parking_lot_4"
VAL_SCENES_RAW="${VAL_SCENES:-${VAL_SCENE:-$VAL_SCENES_DEFAULT}}"
MAX_T="${MAX_T:-1.5}"
MAX_R="${MAX_R:-20.0}"
NUM_WORKER="${NUM_WORKER:-2}"
GPU_IDS="${GPU_IDS:-0}"

NUM_TRIALS="${NUM_TRIALS:-100}"
BASE_SEED="${BASE_SEED:-42}"
TAG="${TAG:-CMRNext}"
SCENE_TAG="${SCENE_TAG:-multi_scene}"
RESULT_BASENAME="${RESULT_BASENAME:-cmrnext_3stage_val_${SCENE_TAG}_rot${MAX_R}_trans${MAX_T}_${NUM_TRIALS}seeds}"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1

find_latest_checkpoint() {
    local run_name="$1"
    local checkpoint_dir="$SAVE_ROOT/$run_name/remove"

    if [ ! -d "$checkpoint_dir" ]; then
        echo "Checkpoint directory not found: $checkpoint_dir" >&2
        return 1
    fi

    local latest
    latest="$(find "$checkpoint_dir" -maxdepth 1 -type f -name 'checkpoint_*.tar' | sort -V | tail -n 1)"
    if [ -z "$latest" ]; then
        echo "No checkpoint_*.tar found in $checkpoint_dir" >&2
        return 1
    fi

    printf '%s\n' "$latest"
}

STAGE1_CKPT="${STAGE1_CKPT:-$(find_latest_checkpoint "$STAGE1_RUN")}"
STAGE2_CKPT="${STAGE2_CKPT:-$(find_latest_checkpoint "$STAGE2_RUN")}"
STAGE3_CKPT="${STAGE3_CKPT:-$(find_latest_checkpoint "$STAGE3_RUN")}"

RESULTS_JSON="${RESULTS_JSON:-$RESULTS_DIR/${RESULT_BASENAME}.json}"
BOXPLOT_OUT="${BOXPLOT_OUT:-$RESULTS_DIR/${RESULT_BASENAME}.pdf}"
SAVE_FILE="${SAVE_FILE:-}"

mkdir -p "$RESULTS_DIR"

read -r -a VAL_SCENE_LIST <<< "$VAL_SCENES_RAW"

echo "============================================================"
echo "Evaluating 3-stage CMRNext on Hercules"
echo "  data root   : $DATA_ROOT"
echo "  stage1 run  : $STAGE1_RUN"
echo "  stage1 ckpt : $STAGE1_CKPT"
echo "  stage2 run  : $STAGE2_RUN"
echo "  stage2 ckpt : $STAGE2_CKPT"
echo "  stage3 run  : $STAGE3_RUN"
echo "  stage3 ckpt : $STAGE3_CKPT"
echo "  val scenes  : ${VAL_SCENE_LIST[*]}"
echo "  max_t/max_r : $MAX_T / $MAX_R"
echo "  gpu ids     : $GPU_IDS"
echo "  workers     : $NUM_WORKER"
echo "  trials      : $NUM_TRIALS"
echo "  base seed   : $BASE_SEED"
echo "  results json: $RESULTS_JSON"
echo "  box plot    : $BOXPLOT_OUT"
echo "============================================================"

cmd=(
    python3 evaluate_flow_calibration.py
    --dataset hercules
    --data_folder "$DATA_ROOT"
    --weights "$STAGE1_CKPT" "$STAGE2_CKPT" "$STAGE3_CKPT"
    --max_t "$MAX_T"
    --max_r "$MAX_R"
    --num_worker "$NUM_WORKER"
    --val_scene "${VAL_SCENE_LIST[@]}"
    --num_trials "$NUM_TRIALS"
    --base_seed "$BASE_SEED"
    --tag "$TAG"
    --save_errors "$RESULTS_JSON"
)

if [ -n "$SAVE_FILE" ]; then
    cmd+=(--save_file "$SAVE_FILE")
fi

"${cmd[@]}"

python3 plot_boxplot.py \
    "$RESULTS_JSON" \
    --out "$BOXPLOT_OUT"

echo ""
echo "============================================================"
echo "Done!"
echo "  results json: $RESULTS_JSON"
echo "  box plot    : $BOXPLOT_OUT"
echo "============================================================"
