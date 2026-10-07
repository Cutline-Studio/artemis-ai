#!/bin/bash
# Claude Code cloud sessions: install the Python test dependencies and the Azure CLI so tests and az work at once.
# Idempotent: skips anything already installed. Does nothing on local machines.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi
cd "$CLAUDE_PROJECT_DIR"

if ! python3 -c "import torch, numpy, yaml, tokenizers, safetensors, transformers, pytest, psycopg" 2>/dev/null; then
  # CPU-only torch: the CUDA build is several GB and nothing here needs a GPU.
  python3 -c "import torch" 2>/dev/null || \
    python3 -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch
  python3 -m pip install -q -r requirements.txt
fi

# Azure CLI in its own virtualenv (pip refuses to replace Debian's argcomplete otherwise).
if ! command -v az >/dev/null 2>&1; then
  python3 -m venv /opt/azcli
  /opt/azcli/bin/pip install -q azure-cli
  ln -sf /opt/azcli/bin/az /usr/local/bin/az
fi
