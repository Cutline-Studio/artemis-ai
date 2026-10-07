#!/bin/bash
# Prepare Claude Code cloud sessions: install test dependencies and, when a scoped
# service principal is configured, sign in to Azure. Local sessions skip this.
set -euo pipefail
[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

# CPU-only torch keeps the install small; no test needs a GPU.
pip install -q --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Azure is opt-in: set AZURE_CLIENT_ID, AZURE_CLIENT_SECRET and AZURE_TENANT_ID as environment
# secrets for a least-privilege service principal. Never use a personal or Global Admin account.
if [ -n "${AZURE_CLIENT_ID:-}" ] && [ -n "${AZURE_CLIENT_SECRET:-}" ] && [ -n "${AZURE_TENANT_ID:-}" ]; then
  command -v az >/dev/null || pip install -q azure-cli
  az login --service-principal -u "$AZURE_CLIENT_ID" -p "$AZURE_CLIENT_SECRET" \
    --tenant "$AZURE_TENANT_ID" --output none
fi
