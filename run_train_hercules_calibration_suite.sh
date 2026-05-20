#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="${DATA_ROOT:-/workspace/data/LG_Innotek/PublicDataset/hercules}"
SAVE_ROOT="${SAVE_ROOT:-/workspace/data/checkpoints/CMRNext/hercules}"
VAL_SCENE="${VAL_SCENE:-library_1}"

EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKER="${NUM_WORKER:-8}"
BASE_LR="${BASE_LR:-3e-4}"
SEED="${SEED:-42}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_COUNT="$(awk -F',' '{print NF}' <<< "$GPU_IDS")"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export MASTER_ADDR
export MASTER_PORT
export NCCL_DEBUG
export NCCL_ASYNC_ERROR_HANDLING
export NCCL_IB_DISABLE
export PYTHONUNBUFFERED=1

if [ -z "$MASTER_PORT" ]; then
    MASTER_PORT="$(python3 - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
)"
fi

VISIBLE_GPU_COUNT="$(python3 - <<'PY'
import os

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if visible:
    print(len([gpu for gpu in visible.split(",") if gpu.strip()]))
else:
    try:
        import torch
        print(torch.cuda.device_count())
    except Exception:
        print(0)
PY
)"

if [ "$VISIBLE_GPU_COUNT" -lt "$GPU_COUNT" ]; then
    echo "Requested $GPU_COUNT GPUs via GPU_IDS=$GPU_IDS, but only $VISIBLE_GPU_COUNT are visible." >&2
    echo "Check docker GPU exposure or CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi

run_train() {
    local run_name="$1"
    local max_t="$2"
    local max_r="$3"

    local save_dir="$SAVE_ROOT/$run_name"
    mkdir -p "$save_dir"

    echo "============================================================"
    echo "Training $run_name"
    echo "  data root : $DATA_ROOT"
    echo "  save root : $save_dir"
    echo "  val scene : $VAL_SCENE"
    echo "  max_t/max_r : $max_t / $max_r"
    echo "  gpus : $GPU_IDS"
    echo "  master : $MASTER_ADDR:$MASTER_PORT"
    echo "============================================================"

    python3 train_calibration.py \
        --hercules_only true \
        --data_folder_hercules "$DATA_ROOT" \
        --savemodel "$save_dir" \
        --val_scene "$VAL_SCENE" \
        --max_t "$max_t" \
        --max_r "$max_r" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --num_worker "$NUM_WORKER" \
        --BASE_LEARNING_RATE "$BASE_LR" \
        --gpu_count "$GPU_COUNT" \
        --master_port "$MASTER_PORT" \
        --seed "$SEED"
}

run_train "hercules_radar" 1.5 20.0
run_train "hercules_radar2" 1.0 10.0
