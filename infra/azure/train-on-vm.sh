#!/bin/bash
# Run on the GPU VM: pretrain the foundation on all 8 GPUs. Resumes automatically after interruptions.
set -euo pipefail
SIZE=${SIZE:-100m}; STEPS=${STEPS:-20000}
cd ~/artemis-ai
pip install -q -r requirements.txt
torchrun --standalone --nproc_per_node 8 -m artemis.train --size "$SIZE" --data runs/data --out "runs/$SIZE" \
  --steps "$STEPS" --batch 16 --accum 4 --lr 3e-4 --warmup $((STEPS / 50)) --eval-every 500 --ckpt-every 500
