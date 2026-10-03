#!/bin/bash
# Create one ND96isr H100 v5 VM (8x H100) for Artemis training. Costs roughly $98/hour while running.
# Requires approved NDSH100v5 quota. Run deliberately; stop-gpu.sh deallocates it.
set -euo pipefail
RG=${RG:-artm-rg}
LOCATION=${LOCATION:-eastus}
NAME=${NAME:-artemis-train-01}
SIZE=Standard_ND96isr_H100_v5
IMAGE=${IMAGE:-microsoft-dsvm:ubuntu-hpc:2204:latest}   # NVIDIA drivers, CUDA and InfiniBand preinstalled

# Refuse to start if the decision engine says the monthly cap would be exceeded (24h estimate).
python3 - <<'PY'
from datetime import date
from artemis.engine import DecisionEngine
DecisionEngine("runs/experiments.jsonl").check_budget(date.today().strftime("%Y-%m"), vms=1, hours=24)
PY

az vm create -g "$RG" -n "$NAME" -l "$LOCATION" --size "$SIZE" --image "$IMAGE" \
  --admin-username artemis --generate-ssh-keys --os-disk-size-gb 1024 \
  --public-ip-sku Standard --nsg-rule NONE --tags project=artemis role=train -o table
az vm auto-shutdown -g "$RG" -n "$NAME" --time 0600 -o none   # daily 06:00 UTC safety shutdown
echo "Created $NAME. No inbound ports are open; reach it with: az ssh vm -g $RG -n $NAME"
