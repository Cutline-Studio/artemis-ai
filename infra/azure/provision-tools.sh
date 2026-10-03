#!/bin/bash
# Code sandbox for chat tools: an Azure Container Apps dynamic session pool (PythonLTS, Hyper-V isolated,
# internet access disabled inside sessions), with the API's managed identity allowed to run code in it.
# Web search works without setup (Wikipedia); for full web search store a Brave Search API key in Key Vault and
# expose it to the app as ARTEMIS_BRAVE_API_KEY (a Key Vault secret reference, like the database URL).
set -euo pipefail
RG=${RG:-artm-rg}; POOL=${POOL:-artemis-code}; APP=${APP:-artemis-api}
az containerapp sessionpool create -g "$RG" -n "$POOL" -l eastus --container-type PythonLTS \
  --max-sessions 20 --cooldown-period 300 --network-status EgressDisabled
ENDPOINT=$(az containerapp sessionpool show -g "$RG" -n "$POOL" --query properties.poolManagementEndpoint -o tsv)
az role assignment create --role "Azure ContainerApps Session Executor" \
  --assignee "$(az containerapp show -g "$RG" -n "$APP" --query identity.principalId -o tsv)" \
  --scope "$(az containerapp sessionpool show -g "$RG" -n "$POOL" --query id -o tsv)"
az containerapp update -g "$RG" -n "$APP" --set-env-vars ARTEMIS_CODE_SESSIONS_ENDPOINT="$ENDPOINT"
