# Artemis AI

Python 3.11+. Artemis's own model (Prime Core transformer + ten routed experts) and the application around it:
orchestrator, planet-named brains, memory, plans and limits, and the website API. Start with
[docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md); the repository map is in its section 4.

## Commands

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
python -m pytest tests                      # ~1 minute on CPU; must be green before any push
python -m artemis.smoke_chat --url <api>   # live deployed-chat check; ARTEMIS_SMOKE_API_KEY from env, never argv
```

## Rules

- Only Artemis's own model answers customers. Never wire an outside LLM into the product or its decisions.
- No secrets in code, commits, logs, PRs or chat. Secrets live in Key Vault `artemis-kv-6b22c1`.
- Permissions, limits, tool execution and deploy controls belong in `business.py`, `tools.py`, `server.py` and `infra/`,
  never in the model. Paid access requires a verified `Authorization: Bearer` API key; `X-Artemis-Account` grants nothing.
- Report results honestly: a claim that something trained, passed or deployed needs the log, eval file or test output.
- New API modules must be added to the `COPY` line in `infra/azure/Dockerfile.api` (the API image has no torch).

## Azure (resource group `artm-rg`)

- Ask the owner before anything that costs money or changes production: starting GPU VMs (~$98/hour each),
  `az containerapp update`, provisioning scripts in `infra/azure/`, or deleting resources.
- Use the scoped service principal (`AZURE_CLIENT_ID` / `AZURE_CLIENT_SECRET` / `AZURE_TENANT_ID` from environment
  secrets), never a personal or Global Admin account.
- Shipping steps are in DEVELOPER_GUIDE section 10.
