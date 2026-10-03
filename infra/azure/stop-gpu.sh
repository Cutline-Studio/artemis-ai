#!/bin/bash
# Deallocate the GPU VM so billing for compute stops (disk storage still bills).
set -euo pipefail
az vm deallocate -g "${RG:-artm-rg}" -n "${NAME:-artemis-train-01}"
