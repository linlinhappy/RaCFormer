# Docker Container Transmission Guide

This guide explains how to safely migrate from the original `racformer` container to the new `racformer_shm` container (with tmpfs-backed input for lower latency), while keeping the original container intact as a fallback.

---

## Overview

| Container | Image | Input Mount | Purpose |
|---|---|---|---|
| `racformer` | `racformer_ready` | `ros_input/` (disk) | Original — kept as fallback |
| `racformer_shm` | `racformer_ready` | `/dev/shm/racformer_input` (tmpfs) | New — lower latency (~83 ms vs ~96 ms) |

---

## Step 1: Commit the Existing Container (one-time only)

Snapshot the current `racformer` container into an image so both containers share the same base:

```bash
docker commit racformer racformer_ready
```

This produces the `racformer_ready` image that `start_docker.sh` launches from.

---

## Step 2: Start the New Container

Run `start_docker.sh` to create the `racformer_shm` container. The original `racformer` container is **not touched**.

```bash
/home/d300/Desktop/RaCFormer/start_docker.sh
```

**What the script does:**
- Creates `/dev/shm/racformer_input` on the host (tmpfs, in RAM)
- Mounts it into the container as `/workspace/ros_input`
- Binds `/media/d300/T9_weilin2` as `/data`
- Starts the container with `--runtime nvidia --gpus all --shm-size=32g`

**Optional flags:**

```bash
# Use a custom container name
./start_docker.sh --name my_name

# Fall back to disk-backed ros_input/ instead of tmpfs
./start_docker.sh --no-shm
```

---

## Step 3: Verify Both Containers Exist

```bash
docker ps -a | grep racformer
```

Expected output — both containers should appear:

```
<id>   racformer_ready   ...   Up   racformer_shm
<id>   racformer_ready   ...   ...  racformer
```

---

## Step 4: Test the New Container

Enter the new container and verify the inference daemon runs:

```bash
docker exec -it racformer_shm bash
conda activate racformer
python /workspace/ros_inference_daemon.py
```

---

## Rollback: Switch Back to the Original Container

If the new container has any issues, the original `racformer` is completely untouched. Switch back immediately:

```bash
docker start racformer
docker exec -it racformer bash
```

---

## Notes

- The tmpfs mount at `/dev/shm/racformer_input` is lost on host reboot. Re-run `start_docker.sh` after a reboot to recreate it.
- The `racformer` container (original) and `racformer_shm` container (new) can run simultaneously without conflict — they mount different input directories.
- To stop the new container without removing it: `docker stop racformer_shm`
- To completely remove the new container: `docker rm racformer_shm`
