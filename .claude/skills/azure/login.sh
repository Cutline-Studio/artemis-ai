#!/bin/bash
# Sign the Azure CLI in as the artemis-claude service principal. Device-code login is blocked by the
# tenant's security defaults (AADSTS530035), so cloud sessions must use the service principal.
set -euo pipefail
if ! command -v az >/dev/null; then
  python3 -m venv /opt/azcli && /opt/azcli/bin/pip install -q azure-cli && ln -sf /opt/azcli/bin/az /usr/local/bin/az
fi
: "${AZURE_CLIENT_ID:?set AZURE_CLIENT_ID in the environment settings}"
: "${AZURE_CLIENT_SECRET:?set AZURE_CLIENT_SECRET in the environment settings}"
: "${AZURE_TENANT_ID:?set AZURE_TENANT_ID in the environment settings}"
az login --service-principal -u "$AZURE_CLIENT_ID" -p "$AZURE_CLIENT_SECRET" --tenant "$AZURE_TENANT_ID" -o none
az account set --subscription "${AZURE_SUBSCRIPTION_ID:-65c3f7b2-285e-47e3-84fe-2c77c16ad301}"
az account show --query "{subscription:name, id:id, user:user.name}" -o table
