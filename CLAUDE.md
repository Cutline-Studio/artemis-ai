# Artemis AI

Artemis trains its own model from scratch and serves it through its own API. The model is the Prime Core
transformer with ten neural experts picked per token by a learned router (`artemis/model.py`). Around it sit the
orchestrator, eleven persona "brains" (`brains/*.yaml`), plans and billing, and a stdlib HTTP API. The brains are
product personas (system prompts plus optional LoRA adapters). They are not the ten neural experts.
No trained weights exist yet, so every chat answers with status `training` until `ARTEMIS_INFERENCE_URL` points at
Artemis's own vLLM server. Read `docs/ARCHITECTURE.md`, `docs/NEURAL_CORE.md` and `docs/DEVELOPER_GUIDE.md` for depth.

## Commands

```bash
# Install. CPU torch first: the default CUDA wheel is several GB and nothing here needs a GPU.
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch && python -m pip install -r requirements.txt

python -m pytest -q tests                      # full suite, about 40 s on an idle 4-core machine
python -m pytest -q --ignore=tests/test_neural_core.py --ignore=tests/test_training.py tests   # app layer only, about 8 s
OMP_NUM_THREADS=1 python -m pytest -q tests    # use this when another torch process shares the CPUs

ARTEMIS_DB=/tmp/artemis.db PORT=8765 python -m artemis.server   # local API; with no model configured it answers "training"
```

Pipeline modules are CLIs run as `python -m artemis.<module>` (corpus, pipeline, train, evaluate, export, infer,
specialize, build_brains, knowledge, checkpoint). Each has a `main(argv=None)` that tests call directly, and its
module docstring holds the canonical usage line. Update that docstring when flags change.

Cloud sessions run `.claude/hooks/session-start.sh`, which installs the requirements and the Azure CLI.
CI (`.github/workflows/tests.yml`) runs only pytest, on Python 3.11 and 3.13. There is no automated deploy.

## Layout

| Path | What it is |
|---|---|
| `artemis/model.py`, `train.py`, `data.py`, `tokenizer.py`, `checkpoint.py` | Prime Core, trainer, data shards, tokenizer, checkpoint format 2 |
| `artemis/evaluate.py`, `export.py`, `infer.py`, `specialize.py`, `build_brains.py`, `knowledge.py` | Evaluation, vLLM export, local inference, brain fine-tuning, self-knowledge SFT data |
| `artemis/engine.py` | Experiment picker and monthly GPU budget guard (`configs/engine.yaml`) |
| `artemis/server.py`, `orchestrator.py`, `backends.py`, `tools.py`, `business.py`, `store.py`, `memory.py`, `brains.py` | The API and everything it uses at runtime |
| `brains/*.yaml`, `configs/business.yaml` | Brain personas and plans/prices. Both are also training data |
| `configs/models.yaml` | Every model size. Sizes come only from here |
| `infra/azure/`, `infra/local/` | Hand-run deploy and training scripts, CPU prototype |
| `.claude/skills/azure/`, `.mcp.json` | Azure sign-in skill and Microsoft's Azure MCP Server |

`runs/` and `serve/` hold checkpoints and exports and are gitignored. Committed results go in `docs/results/`.

## Rules that must hold

- Only Artemis's own model answers customers. Never add an outside model backend (OpenAI, Claude, Gemini and so on).
- Tool output reaches the model only through `result_block` in `artemis/tools.py`, labeled as data, never as
  instructions. Retrieved memory reaches it only through `context_block` in `artemis/memory.py`. Keep both.
- Do not weaken the `web_fetch` SSRF guard (`check_public_url` and the pinned-IP connections in `tools.py`). It must run
  on every redirect hop. Never swap `fetch_page` for plain `urllib.urlopen`.
- `ARTEMIS_LOCAL_CODE=1` runs `run_python` in a local subprocess. It is for development only and is not a sandbox.
  Production uses `ARTEMIS_CODE_SESSIONS_ENDPOINT` (Azure dynamic sessions).
- Order-sensitive names: `EXPERT_NAMES` in `model.py` indexes the `.dom` label shards and checkpoints, and the key order of
  `TIERS` in `orchestrator.py` defines plan entitlements. Do not reorder or rename either.
- `load_brains` requires all ten specialists plus `artemis`. Removing or renaming a brain yaml breaks the server.
- `brains/*.yaml` and `configs/business.yaml` feed the weights through `knowledge.py`. A training test asserts that
  the artemis data contains "$20 a month", "$200 a month" and "I'm Artemis", and does not contain "Cutline".
- Plan limits and prices live in `configs/business.yaml`. Do not hard-code them in Python.
- The API image (`infra/azure/Dockerfile.api`) has no torch and copies an explicit list of modules. A new module the
  server imports must be added to its `COPY` line, and server-path code must not import torch at module level.
- SQL uses `?` placeholders (`store.py` rewrites them for Postgres). Never put a literal `?` in SQL text. The schema
  is `CREATE ... IF NOT EXISTS` with no migrations, so a column change does not reach existing databases.
- Config files resolve relative to the package (`Path(__file__).resolve().parent.parent`), not the working directory.

## Traps

- `python -m artemis.train` defaults to `--size smoke`, the 2-layer test model. Always pass `--size`.
- Training resumes automatically from `<out>/latest.pt`. Reusing `--out` with another size exits; with the same size
  it silently resumes. To pause, use `--stop-at` and keep `--steps` unchanged, since `--steps` sets the LR schedule.
- `ARTEMIS_ALLOWED_ORIGINS` and `ARTEMIS_TRUST_PROXY` are read once at import. In tests, patch `server.TRUST_PROXY`
  or `server.ALLOWED_ORIGINS` instead of setting the env var.
- Unset `ARTEMIS_DATABASE_URL` before running tests. If it is set it wins over `ARTEMIS_DB=':memory:'` and a test
  would connect to that real database.
- Any `Authorization` header that is not a valid `Bearer art-...` key gets 401. It does not fall back to guest.
- Only `/v1/chat/completions` retrieves stored memory, and it runs without tools. `/v1/chat` and `/v1/chat/stream`
  use tools but never memory.
- `/v1/chat/stream` sends its 200 and SSE headers before the generator runs. Validate everything before streaming.
- Tool quota is counted before the tool runs, so failed tool calls still use the daily allowance.
- `GET /v1/status` with a real inference server makes a real generation call (cached for 15 s).
- `torch.load(weights_only=False)` is used for checkpoints and adapters. Load only trusted `.pt` files.
- `python -m artemis.smoke_chat` hits the deployed API and uses real message and tool allowances. Ask first.

## Tests and code style

- Tests need no network, GPU or Postgres. They use `Business(':memory:')`, `tmp_path`, hand-written fakes
  (FakeBackend, ScriptedModel, FakeSearch) rather than mocks, and HTTP servers on `127.0.0.1` port 0.
- Test names are full behavior sentences, as plain functions. Put regression tests next to the feature's tests and
  cover failure and abuse cases (SSRF, auth spoofing, limits, interrupted streams).
- Keep pytest's default import mode and do not add `tests/__init__.py`: `test_chat_failures.py` imports helpers from
  `test_tools`.
- No formatter, linter or line limit is configured, and lines up to about 180 characters are normal. Do not reformat
  code you are not otherwise changing, and do not run ruff, black or mypy with fixes across the repo.
- Every module starts with a docstring stating its purpose and `from __future__ import annotations`. Comments are
  sparse and explain why, not what. Import torch and psycopg lazily where the module can run without them.
- Commit subjects are imperative and sentence case with no prefix or trailing period, for example
  "Add CI test workflow and cloud-session startup hook".

## Azure

- Everything is in resource group `artm-rg`, subscription `65c3f7b2-285e-47e3-84fe-2c77c16ad301`: Container App
  `artemis-api` (East US), ACR `artemisapi6b22c1`, Postgres `artemis-db-6b22c1` (East US 2, because East US blocks new
  Postgres servers), Key Vault `artemis-kv-6b22c1` and session pool `artemis-code`.
- Sign in with `.claude/skills/azure/login.sh`, which uses the service principal in `AZURE_CLIENT_ID`,
  `AZURE_CLIENT_SECRET` and `AZURE_TENANT_ID`. Device-code login is blocked by the tenant's security defaults
  (AADSTS530035). Never ask for secrets in chat; they belong in the environment settings.
- The service principal has Contributor and cannot assign roles. The role-assignment lines in `provision-db.sh` and
  `provision-tools.sh` must be run by the owner in Azure Cloud Shell.
- Deploy order: `register-providers.sh` once, then `deploy-api.sh`, then `provision-db.sh`, then `provision-tools.sh`.
  `provision-db.sh` is not idempotent: a rerun makes a new admin password and tries to recreate the server.
- Confirm with the owner and state the price before anything that costs money. `provision-gpu.sh` costs about
  $98/hour; always run `stop-gpu.sh` afterwards. GPU quota is currently 0.
- Verify an API deploy with `curl https://artemis-api.whitemeadow-751c0637.eastus.azurecontainerapps.io/v1/status`.
