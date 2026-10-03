# Artemis AI: developer starter guide

Welcome. This guide gets you from a fresh checkout to running tests, training a small model, running the API,
and shipping a change. Read the ground rules first; they are not optional.

## 1. What you are working on

Artemis AI builds its own AI, trained from scratch. Two codebases:

| Codebase | What it is | Hosted on |
|---|---|---|
| `artemis-ai` (Python) | The model (Prime Core + ten neural experts), training, evaluation, inference, the orchestrator, chat tools, memory, plans and limits, and the API server | Azure Container Apps (`artemis-api`) |
| Website (static HTML/CSS/JS in `src/`) | Company site and the customer chat UI; talks to the API | Vercel (project `cutline-industries`), domain cutline-industries.studio |

Read next: `docs/NEURAL_CORE.md` (the model) and `docs/ARCHITECTURE.md` (the application around it).

## 2. Ground rules

1. **Only Artemis's own model answers customers.** No OpenAI, Claude, Grok, Gemini or any outside model in the
   product or in Artemis's decisions (Blueprint Decisions 5 and 12). Search engines and code sandboxes are tools, not models.
2. **No secrets in code, chat, tickets or logs.** Secrets live in Azure Key Vault (`artemis-kv-6b22c1`) and reach
   the app as Key Vault references. Never paste a key or password anywhere to "share" it.
3. **Spending:** the system has a hard cap of $2,000,000 per month. Never start GPU VMs (about $98/hour each)
   without the owner's go-ahead.
4. **Tools are the application's job.** Permissions, limits, tool execution and deployment controls belong in
   `business.py`, `tools.py`, `server.py` and `infra/`, never inside the model.
5. **Report results honestly.** A claim that something trained, passed or improved needs the log or eval file behind it.
6. **Tests pass before you ship.** `python -m pytest tests` must be green.

## 3. Set up (10 minutes)

Requirements: Python 3.11+, git, and for the website tests Node 18+ with Playwright. A GPU is not needed for anything below.

```bash
cd artemis-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch, numpy, pyyaml, tokenizers, safetensors, pytest, psycopg
pip install transformers                 # optional: enables the export-equivalence tests
python -m pytest tests                   # about 45 s on a laptop CPU; expect all tests to pass
```

## 4. Repository map

| Path | What lives there |
|---|---|
| `artemis/model.py` | Prime Core transformer, `ExpertMixture` (router + ten experts), losses, generation |
| `artemis/train.py` | Training loop: resume, `--stop-at`, optimizer audit, routing diagnostics |
| `artemis/checkpoint.py` | Checkpoint format v2, strict loading, v1 → v2 migration, `inspect` |
| `artemis/evaluate.py` | Validation loss, per-expert routing, expert ablation, dense-baseline comparison |
| `artemis/data.py`, `tokenizer.py`, `corpus.py`, `pipeline.py` | Licensed data → tokenizer → token shards with domain labels and provenance |
| `artemis/infer.py` | Load a checkpoint and stream text, with expert routing traces |
| `artemis/specialize.py`, `knowledge.py`, `build_brains.py` | Fine-tuning (full or LoRA) and the self-knowledge / tool-use training sets |
| `artemis/export.py` | Export to Llama (dense) or Qwen2-MoE (experts) layout for vLLM |
| `artemis/orchestrator.py` | Plans each request, runs the tool loop, memory, audit, streaming events |
| `artemis/tools.py` | `web_search`, `web_fetch`, `run_python` |
| `artemis/memory.py` | Per-account notes (external memory, not weights) |
| `artemis/business.py`, `store.py` | Plans, limits, API keys, metering, waitlist; SQLite locally, PostgreSQL in Azure |
| `artemis/server.py`, `backends.py` | HTTP API; where answers come from (local checkpoint, vLLM server, or "still training") |
| `artemis/engine.py` | Decision engine: next experiment (UCB1), compute ramp, spending cap |
| `configs/models.yaml` | Model sizes (`smoke`, `proto`, `proto-dense`, `100m`, `1b`, `3b`, `8b`) |
| `configs/business.yaml` | Plans, prices, limits, which tools each plan gets |
| `brains/*.yaml` | Artemis and the ten planet personas |
| `infra/azure/`, `infra/local/` | Deploy, provisioning, GPU and prototype scripts |
| `tests/` | Everything above, tested on CPU |

## 5. Quick starts

### Train and evaluate the CPU prototype (about 90 minutes on 4 cores)

```bash
infra/local/run-prototype.sh            # corpus → tokenizer → shards → 3 runs → evaluations
cat runs/proto/eval.json                # per-expert routing, ablation, dense baseline
```

### Talk to a checkpoint from the command line

```bash
python -m artemis.infer --checkpoint runs/proto/latest.pt --tokenizer runs/tok-proto.json \
    --prompt "Security review of billing:" --raw --trace
```

`--trace` reports which experts the router picked for the generated tokens.

### Run the API locally on a checkpoint

```bash
ARTEMIS_CHECKPOINT=runs/proto/latest.pt ARTEMIS_TOKENIZER=runs/tok-proto.json ARTEMIS_DB=dev.db PORT=8099 \
    python -m artemis.server
curl localhost:8099/v1/status
curl -N -X POST localhost:8099/v1/chat/stream -H "Content-Type: application/json" \
    -d '{"message": "Deploy checkout to eastus", "tier": "gpt-1-base"}'
```

`/v1/status` shows the loaded checkpoint, its step, and the experts. The stream returns `plan`, then `token`
events, then `done`. (The prototype was never trained on chat, so its replies are not meaningful.) Add
`ARTEMIS_LOCAL_CODE=1` to let `run_python` use a local subprocess; that is for development only and is not a sandbox.

### Run the website locally

```bash
npx serve src                           # from the website folder; open the printed URL, add #chat for the chat
```

The API address is the `API` constant near the top of `src/polish.js`. To use a local API, change it and add your
local origin to `ARTEMIS_ALLOWED_ORIGINS` on the API.

## 6. The API

Base URL: `https://artemis-api.whitemeadow-751c0637.eastus.azurecontainerapps.io`

| Method and path | Auth | What it does |
|---|---|---|
| `GET /v1/status` | none | Whether a model is serving, which tools are configured, model info |
| `GET /v1/plans`, `GET /v1/brains` | none | Plans and prices; the personas |
| `POST /v1/chat` | optional `X-Artemis-Account` | One answer as JSON |
| `POST /v1/chat/stream` | optional `X-Artemis-Account` | Server-Sent Events: `plan`, `tool_call`, `tool_result`, `token`, `replace`, then `done` or `training` |
| `POST /v1/chat/completions` | `Authorization: Bearer art-...` | OpenAI-compatible request and response, billed per token |
| `GET/POST /v1/memory`, `DELETE /v1/memory/<id>` | API key | The account's memory notes |
| `POST /v1/waitlist` | none | `{"email": ...}` |

Errors: 400 bad input, 401 missing or unknown key, 402 the plan doesn't include it, 429 limit reached.
The `X-Artemis-Account` header is not proof of identity, so memory is only available with an API key.

Stream example:

```
event: tool_call
data: {"type": "tool_call", "name": "web_search", "arguments": {"query": "latest python release"}}

event: tool_result
data: {"type": "tool_result", "name": "web_search", "ok": true, "summary": "Searched the web for ...", "sources": [...]}

event: token
data: {"type": "token", "text": "Python 3.14 is "}

event: done
data: {"type": "done", "answer": "...", "tools_used": ["web_search"], "plan": "free", "tier": "gpt-2-sft", ...}
```

## 7. The model in one page

- **Prime Core**: decoder-only transformer with RMSNorm, rotary embeddings, grouped-query attention, SwiGLU.
- **Experts**: at the layers in `moe_layers`, a learned router picks the top 2 of 10 SwiGLU experts per token
  and renormalizes their weights; their outputs add into the residual stream next to the core's own FFN.
- **Training loss** = next-token loss + 0.01 × load balance + 0.001 × router z-loss + `--route-coef` × routing
  supervision on domain-labeled tokens.
- **Checkpoints**: format 2, strict loading. Old dense checkpoints must be migrated
  (`python -m artemis.checkpoint migrate --src old.pt --dst new.pt --moe-every 2`); the original is never modified.
- **Evaluate** every run: `python -m artemis.evaluate --checkpoint ... --data ... --out eval.json [--baseline dense.pt]`.
  An expert counts as specialized only if removing it hurts its own domain clearly more than the others.

Train on GPUs (owner approval first):

```bash
infra/azure/provision-gpu.sh             # one ND H100 v5 VM (8 GPUs)
SIZE=100m STEPS=20000 infra/azure/train-on-vm.sh
python -m artemis.export --checkpoint runs/100m/latest.pt --tokenizer runs/tok.json --out serve/artemis
infra/azure/serve-on-vm.sh               # vLLM; then set ARTEMIS_INFERENCE_URL on the API
infra/azure/stop-gpu.sh                  # always stop the VM when done
```

## 8. Chat tools

The model asks for a tool by writing `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`. The orchestrator
checks the plan (`business.authorize_tool`), runs it (`tools.Toolbox.execute`), and feeds the result back in a
`<tool_result>` block labeled as untrusted data. Up to 4 calls per answer. Customers see each step in the chat.

To add a tool:
1. Add its spec to `TOOL_SPECS` and a branch in `Toolbox.execute` (`artemis/tools.py`). Validate every argument; return `ToolResult`.
2. Add its name to `ALL_TOOLS` (`artemis/business.py`) and to the plans that get it (`configs/business.yaml`).
3. Add training examples to `tools_set()` (`artemis/knowledge.py`), using real outputs.
4. Add tests next to `tests/test_tools.py`, including the failure and abuse cases.
5. If it reaches the network or runs code, document its safety limits in `docs/ARCHITECTURE.md`.

## 9. Configuration (API environment variables)

| Variable | Meaning |
|---|---|
| `ARTEMIS_DATABASE_URL` | PostgreSQL URL (Key Vault reference in Azure). Otherwise `ARTEMIS_DB` (SQLite path, default `artemis.db`) |
| `ARTEMIS_INFERENCE_URL`, `ARTEMIS_INFERENCE_KEY` | Artemis's own vLLM server |
| `ARTEMIS_CHECKPOINT`, `ARTEMIS_TOKENIZER` | Load a checkpoint into the API process instead (needs torch) |
| `ARTEMIS_ALLOWED_ORIGINS` | Comma-separated website origins allowed to call the API |
| `ARTEMIS_TRUST_PROXY=1` | Use the client IP from Azure's ingress (needed for fair per-visitor limits) |
| `ARTEMIS_CODE_SESSIONS_ENDPOINT` | Azure dynamic sessions pool for `run_python` |
| `ARTEMIS_BRAVE_API_KEY` | Full web search (otherwise Wikipedia); store in Key Vault |
| `ARTEMIS_LOCAL_CODE=1` | Development only: run code in a local subprocess |
| `ARTEMIS_MODEL_ROUTER=1` | Let the trained orchestrator choose personas instead of keywords |

## 10. Shipping a change

API (from `artemis-ai/`):

```bash
python -m pytest tests
az acr build -r artemisapi6b22c1 -t artemis-api:vN -f infra/azure/Dockerfile.api .
az containerapp update -g artm-rg -n artemis-api --image artemisapi6b22c1.azurecr.io/artemis-api:vN
curl https://<api>/v1/status
```

If a module is new, add it to the `COPY` line in `infra/azure/Dockerfile.api`; the API image has no torch.

Website: deploy `src/` to the Vercel project `cutline-industries` (output directory `src`, no build step), then check
that the live files match what you deployed and test the chat on a phone width and a desktop width.

## 11. Azure resources (resource group `artm-rg`)

| Resource | Name | Region | Purpose |
|---|---|---|---|
| Container App + environment | `artemis-api`, `artemis-api-env` | East US | The API |
| Container registry | `artemisapi6b22c1` | East US | API images |
| PostgreSQL flexible server | `artemis-db-6b22c1`, database `artemis` | East US 2 | Accounts, usage, waitlist, memory |
| Key Vault (RBAC) | `artemis-kv-6b22c1` | East US | Database URL and other secrets |
| Dynamic session pool | `artemis-code` | East US | Isolated Python sandboxes for `run_python` (no internet) |

New PostgreSQL servers are blocked in East US on this subscription, so the database is in East US 2.

## 12. Where things stand

Built and tested on CPU: the neural core and its evaluations, checkpoints and migration, inference, export,
orchestrator, chat tools, memory, plans and limits, streaming API, website chat. Running in Azure: API, database,
Key Vault, code sandbox, website. Not done yet: any GPU training (H100 quota is 0), a production tokenizer and
licensed training data, vLLM serving of the expert model. Until a model passes evaluation, chat answers "still training".
