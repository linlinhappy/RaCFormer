#!/usr/bin/env bash
# start_docker.sh — Start the RaCFormer inference container.
#
# Mounts /dev/shm/racformer_input as /workspace/ros_input so the collector
# node can write .npz files directly to tmpfs (RAM), reducing latency from
# ~96 ms to ~83 ms per frame.  The host's /dev/shm is on tmpfs by default;
# no sudo required to create subdirectories there.
#
# Usage:
#   ./start_docker.sh                        # start container named 'racformer_shm'
#   ./start_docker.sh --name my_name         # custom container name
#   ./start_docker.sh --no-shm              # fall back to disk-backed ros_input/

set -euo pipefail

CONTAINER_NAME="racformer_shm"
WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"

USE_SHM=true
args=("$@")
for i in "${!args[@]}"; do
    [[ "${args[$i]}" == "--no-shm" ]] && USE_SHM=false
    if [[ "${args[$i]}" == "--name" ]]; then
        CONTAINER_NAME="${args[$((i+1))]}"
    fi
done

# ---- Prepare SHM input directory ------------------------------------------
SHM_INPUT="/dev/shm/racformer_input"
if $USE_SHM; then
    mkdir -p "$SHM_INPUT"
    echo "[start_docker] Using tmpfs input: $SHM_INPUT -> /workspace/ros_input"
else
    echo "[start_docker] Using disk input: $WORKSPACE_DIR/ros_input -> /workspace/ros_input"
fi

# ---- Stop existing container if running ------------------------------------
if docker ps -q --filter "name=^${CONTAINER_NAME}$" | grep -q .; then
    echo "[start_docker] Stopping existing container '${CONTAINER_NAME}'..."
    docker stop "$CONTAINER_NAME" >/dev/null
fi
if docker ps -aq --filter "name=^${CONTAINER_NAME}$" | grep -q .; then
    docker rm "$CONTAINER_NAME" >/dev/null
fi

# ---- Build volume args -------------------------------------------------------
VOLUME_ARGS=(
    -v "${WORKSPACE_DIR}:/workspace"
    -v "/media/d300/T9_weilin2:/data"
)
if $USE_SHM; then
    # More-specific bind: overrides /workspace/ros_input from the parent mount
    VOLUME_ARGS+=(-v "${SHM_INPUT}:/workspace/ros_input")
fi

# ---- Start container --------------------------------------------------------
docker run -dit \
    --name "$CONTAINER_NAME" \
    --runtime nvidia \
    --gpus all \
    --shm-size=32g \
    "${VOLUME_ARGS[@]}" \
    racformer_ready \
    /bin/bash

echo "[start_docker] Container '${CONTAINER_NAME}' started."
if $USE_SHM; then
    echo "[start_docker] Inference daemon: python /workspace/ros_inference_daemon.py"
    echo "               (reads from /workspace/ros_input → host $SHM_INPUT)"
fi
