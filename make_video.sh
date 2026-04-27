#!/bin/bash
# Convert grid overlay images to video at 6 fps

INPUT_DIR="/home/d300/Desktop/RaCFormer/ros_output/overlays"
OUTPUT="/home/d300/Desktop/RaCFormer/ros_output/grid_video.mp4"

# Sort files by name (timestamp order) and feed to ffmpeg
ls -1 "$INPUT_DIR"/frame_*_grid.jpg | sort | \
  ffmpeg -f concat -safe 0 \
    -i <(for f in $(ls -1 "$INPUT_DIR"/frame_*_grid.jpg | sort); do echo "file '$f'"; echo "duration 0.1667"; done) \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    -c:v libx264 -pix_fmt yuv420p \
    -y "$OUTPUT"

echo "Done: $OUTPUT"
