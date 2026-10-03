# Artemis AI architecture

Artemis AI runs on one model it trains itself (Decisions 3, 5, 12). No outside AI model answers customers or makes decisions.

**Inside the model** ([NEURAL_CORE.md](NEURAL_CORE.md)): the Prime Core, a shared transformer, plus ten trainable
neural experts (intent, architecture, planning, build, test, deploy, memory, security, observability, cost) that a
learned router picks per token, top two at a time.

**Around the model** (this page): the application. Artemis the orchestrator and the ten planet brains are product
personas (system prompts, optionally LoRA adapters on the Prime Core) that the website and plans expose. They are
not the neural experts.

```
                         ┌─────────────────────────────────┐
  website / voice  ───►  │  ARTEMIS  (orchestrator, big     │ ───► one answer
                         │  brain): plan → route → merge    │
                         └───────┬────────────────┬────────┘
              memory first ──►  Mercury        Venus  ◄── audits every draft (PASS / FAIL → fix)
                                 │
     ┌────────┬────────┬─────────┼─────────┬─────────┬────────┬────────┐
   Apollo   Earth     Mars    Jupiter    Saturn    Uranus   Neptune   Pluto
  planning  voice   shipping multimodal   code    math/logic research architecture
```

## The eleven brains

| Brain | Role | Used for |
|---|---|---|
| **Artemis** | Orchestrator, the big brain | Plans each request, picks brains, merges answers |
| **Apollo** | Navigator | Roadmaps, planning, strategy, business |
| **Mercury** | Memory | Persona for memory questions; retrieval itself comes from the memory store (`artemis/memory.py`) |
| **Venus** | Auditor | Checks every draft for errors and safety before it is sent |
| **Earth** | Voice | Spoken answers for the voice box, conversation, tone, translation |
| **Mars** | Shipping | Release readiness, go/no-go, rollback plans |
| **Jupiter** | Multimodal | Images, screenshots, charts, documents |
| **Saturn** | Code | Writing, fixing and testing code |
| **Uranus** | Math and logic | Calculation, statistics, step-by-step reasoning |
| **Neptune** | Deep research | Cited research and source comparison |
| **Pluto** | Systems architecture | Infrastructure, scaling, cloud design |

Definitions live in `brains/*.yaml` (role, instructions, rules, routing keywords, training focus, evaluations).

## Chat tools

Customers' chat can search the web, read pages, and run Python, all on Artemis's own model: the model asks for a
tool in its reply, and the application (`artemis/tools.py`, `artemis/orchestrator.py`) checks the plan, runs the tool,
and hands the result back to the model marked as data, never instructions. Up to 4 tool calls per answer.

| Tool | Runs on | Safety |
|---|---|---|
| `web_search` | Wikipedia search (no key), or Brave Search when `ARTEMIS_BRAVE_API_KEY` is set | Results are labeled untrusted |
| `web_fetch` | The API server | Public http(s) only; private, loopback, link-local and metadata addresses refused, redirects re-checked, 2 MB cap |
| `run_python` | Azure Container Apps dynamic sessions (`artemis-code` pool) | Hyper-V isolated sandbox per customer, no internet access, 60 s per run |

| Plan | Tools | Tool calls per day |
|---|---|---|
| Free | web search, web pages | 20 |
| Plus | all, including running code | 300 |
| Pro / Enterprise | all | unlimited |
| Team | all | 1,000 |

The website shows each step ("Searching the web for …", sources, "Ran Python" with the code and output) and renders
code blocks with a Copy button. Tool use is in the training data (`artemis/knowledge.py`, `tools` set; code outputs
are real), but the model only uses tools well once it is trained; until then chat answers "still training".

## Product tiers (the ladder)

| Tier | Brains used | Memory | Venus audit |
|---|---|---|---|
| GPT-1 Base | Artemis alone | no | no |
| GPT-2 SFT | best 1 specialist | no | no |
| GPT-3 Aligned RAG | up to 3 specialists | yes | yes |
| GPT-4 ASI Orchestrator | up to all 10 | yes | yes |

Customers can also open any single planet brain directly (for example "Saturn" for code).

## How the weights are built

1. **Foundation** (`artemis/train.py`): the Prime Core and its ten experts, trained together from random
   initialization on licensed data (`artemis/data.py` refuses sources without a recorded license). Sizes in
   `configs/models.yaml`: 100M → 1B → 3B → 8B (experts at every second layer). See [NEURAL_CORE.md](NEURAL_CORE.md).
2. **Specialists** (`artemis/specialize.py`): each planet is a LoRA adapter (a small set of extra weights) trained
   on that brain's conversations on top of the shared foundation. One server holds all ten.
3. **Orchestrator**: Artemis itself is fine-tuned on routing and merging examples collected from real traces.
4. **Serving** (`artemis/export.py`, `infra/azure/serve-on-vm.sh`): weights are exported to the Qwen2-MoE layout
   (Llama for dense models) and served by vLLM; the export is tested to give identical outputs. Without a GPU server,
   `ARTEMIS_CHECKPOINT` loads a checkpoint straight into the API process (`artemis/infer.py`).

**Memory** is not in the weights: per-account notes live in the database (`artemis/memory.py`), are retrieved by
relevance for tiers with memory, and are placed in the prompt in a block labeled as retrieved memory.
Only API-key accounts have memory. The website chat streams answers over `POST /v1/chat/stream`.

## Business model (same structure as ChatGPT, Claude and Grok)

Defined once in `configs/business.yaml`, enforced by `artemis/business.py` and `artemis/server.py`:

| Plan | Price | Ladder tier | Brains | Messages |
|---|---|---|---|---|
| Free | $0 | GPT-2 SFT | Earth, Neptune, Saturn | 30 / day |
| Plus | $20 / month | GPT-3 Aligned RAG | all ten | 500 / day |
| Pro | $200 / month | GPT-4 ASI Orchestrator | all ten | unlimited |
| Team | $30 / seat / month | GPT-3 Aligned RAG | all ten | 1,000 / day, admin console |
| Enterprise | contact sales | GPT-4 ASI Orchestrator | all ten | unlimited, SSO, API |

Developer API: `POST /v1/chat/completions` (same request shape as OpenAI's), API keys (`art-...`, stored hashed),
billed per million tokens per model (`artemis-base`, `artemis-sft`, `artemis-aligned`, `artemis-orchestrator`),
60 requests/minute, $5 starting credit. Over-limit requests get HTTP 402 (upgrade) or 429 (limit reached).

## Self-knowledge in the weights

`artemis/knowledge.py` turns `brains/*.yaml` and `configs/business.yaml` into training conversations
(who Artemis is, the eleven brains, plans, prices, API, policies, routing). Each brain's `training_sets`
says which sets its weights learn:

| Weights | Learns |
|---|---|
| Artemis (full fine-tune) | identity, business, brains, routing |
| Each planet (adapter) | identity, business, brains |

`python -m artemis.build_brains ...` regenerates the data and trains all eleven. Change a price in
`business.yaml`, rebuild, and the weights match.

## How the system improves itself

`artemis/engine.py` is the decision engine ("own brain", stage 1):
- picks the next experiment with UCB1 (mostly build on what scored best, sometimes explore),
- unlocks larger model sizes only after the smaller size has a stable result,
- ramps compute 1 → 4 → 16 → 27 VMs, advancing only after a stable run that improved,
- refuses any run that would exceed the $2,000,000 monthly cap,
- promotes a new model only if it beats the current one on held-out data by a margin.

In stage 2, once Artemis's own model passes its routing evaluation, it proposes experiments and routing
plans itself (`ARTEMIS_MODEL_ROUTER=1`); the engine stays as the referee.

## Running it

| Step | Command |
|---|---|
| Tests (CPU) | `python -m pytest tests` |
| CPU prototype (neural core vs dense baseline) | `infra/local/run-prototype.sh` |
| Evaluate experts | `python -m artemis.evaluate --checkpoint ... --data ... --out eval.json` |
| Inspect / migrate a checkpoint | `python -m artemis.checkpoint inspect ...` / `migrate ...` |
| Run a checkpoint locally | `python -m artemis.infer --checkpoint ... --tokenizer ... --prompt "..." --trace` |
| Tokenizer | `python -m artemis.pipeline tokenizer --files corpus/*.txt --out runs/tok.json` |
| Data shards | `python -m artemis.pipeline prepare --sources sources.yaml --tokenizer runs/tok.json --out runs/data` |
| Start GPU VM | `infra/azure/provision-gpu.sh` (needs H100 quota; ~$98/hour) |
| Pretrain | `infra/azure/train-on-vm.sh` |
| Specialize a brain | `python -m artemis.specialize --brain saturn ...` |
| Export + serve | `python -m artemis.export ...`, then `infra/azure/serve-on-vm.sh` |
| Website API | `infra/azure/deploy-api.sh`, then set `ARTEMIS_INFERENCE_URL` |
| Stop GPU VM | `infra/azure/stop-gpu.sh` |

## Status

Built and tested on CPU: the neural core (Prime Core + ten experts, router, losses, diagnostics, checkpoints and
migration, evaluations, CPU prototype against a dense baseline), brain definitions, specialization, export,
orchestrator, memory store, streaming website API, business engine, self-knowledge training, decision engine.
Not yet run: anything on GPUs. Blocked on Azure H100 quota.
