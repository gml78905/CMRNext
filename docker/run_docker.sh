#!/bin/bash

# CMRNext Docker 컨테이너 실행 스크립트

# 컨테이너 및 이미지 이름 설정
CONTAINER_BASE_NAME="cmrnext_container"
IMAGE_NAME="${IMAGE_NAME:-wanheekim/cmrnext:latest}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
CONTAINER_WORKDIR="/workspace/CMRNext"

# ------------------------------------------------------------------
# [GPU 설정 로직]
# 첫 번째 인자($1)가 없으면 "all", 있으면 "device=$1" 형식으로 설정
if [ -z "$1" ]; then
    GPU_OPTION="all"
    GPU_NAME_SUFFIX="all"
    echo "▶ GPU 모드: 모든 GPU 사용 (Default)"
else
    GPU_OPTION="device=$1"
    GPU_NAME_SUFFIX="$(echo "$1" | tr ',:' '__')"
    echo "▶ GPU 모드: 지정된 GPU 사용 ($1)"
fi

CONTAINER_NAME="${CONTAINER_BASE_NAME}_${GPU_NAME_SUFFIX}"
# ------------------------------------------------------------------

# 현재 디렉토리 경로 (프로젝트 루트)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 데이터 디렉토리 경로
DATA_DIR="${DATA_DIR:-/media/TrainDataset/}"

if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "Error: Docker 이미지를 찾을 수 없습니다: $IMAGE_NAME"
    echo "힌트: IMAGE_NAME 환경변수로 다른 이미지를 지정할 수 있습니다."
    exit 1
fi

DOCKER_ENV_ARGS=()
if [ -n "$WANDB_API_KEY" ]; then
    DOCKER_ENV_ARGS+=(-e "WANDB_API_KEY=$WANDB_API_KEY")
fi

# 같은 GPU suffix를 가진 기존 컨테이너가 실행 중이면 중지 및 제거
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "기존 컨테이너를 중지하고 제거합니다..."
    docker stop $CONTAINER_NAME > /dev/null
    docker rm $CONTAINER_NAME > /dev/null
fi

# Docker 컨테이너 실행
echo "Docker 컨테이너를 실행합니다..."
echo " - 프로젝트 경로: $PROJECT_ROOT"
echo " - 컨테이너 이름: $CONTAINER_NAME"
echo " - Docker 이미지: $IMAGE_NAME"

# 데이터 디렉토리 존재 확인 및 마운트 옵션 설정
if [ -d "$DATA_DIR" ]; then
    echo " - 작업 디렉토리: $CONTAINER_WORKDIR"
    echo " - 데이터 마운트: $DATA_DIR -> /workspace/data"
    docker run -it --rm \
        --name $CONTAINER_NAME \
        --gpus "$GPU_OPTION" \
        --shm-size=32g \
        "${DOCKER_ENV_ARGS[@]}" \
        -v "$PROJECT_ROOT:$CONTAINER_WORKDIR" \
        -v "$DATA_DIR:/workspace/data" \
        -w "$CONTAINER_WORKDIR" \
        $IMAGE_NAME \
        /bin/bash
else
    echo "Warning: 데이터 디렉토리를 찾을 수 없습니다 ($DATA_DIR)"
    echo " - 데이터 마운트 없이 실행합니다."
    echo " - 작업 디렉토리: $CONTAINER_WORKDIR"
    docker run -it --rm \
        --name $CONTAINER_NAME \
        --gpus "$GPU_OPTION" \
        --shm-size=32g \
        "${DOCKER_ENV_ARGS[@]}" \
        -v "$PROJECT_ROOT:$CONTAINER_WORKDIR" \
        -w "$CONTAINER_WORKDIR" \
        $IMAGE_NAME \
        /bin/bash
fi
