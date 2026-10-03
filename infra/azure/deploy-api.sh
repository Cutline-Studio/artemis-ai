#!/bin/bash
# Deploy the website API to Azure Container Apps (scales to zero when idle).
# Until ARTEMIS_INFERENCE_URL is set, every chat answers with status "training".
set -euo pipefail
RG=${RG:-artm-rg}
az containerapp up -g "$RG" -n artemis-api -l "${LOCATION:-eastus}" --source . \
  --dockerfile infra/azure/Dockerfile.api --ingress external --target-port 8080 \
  --env-vars ARTEMIS_ALLOWED_ORIGINS="https://cutline-industries.studio,https://cutline-industries.vercel.app"
