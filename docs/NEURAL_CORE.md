# Artemis neural core

One trainable **Prime Core** and **ten trainable neural experts**, inside one model (`artemis/model.py`). This is
the model used by training (`artemis/train.py`), evaluation (`artemis/evaluate.py`), fine-tuning
(`artemis/specialize.py`), export for serving (`artemis/export.py`), local inference (`artemis/infer.py`) and the
API server (`artemis/server.py` → `backends.LocalBackend`).

```
tokens ─► embedding ─► ┌──────────── layer i (Prime Core) ──────────────┐ ─► … ─► RMSNorm ─► tied head ─► next token
                       │ RMSNorm → grouped-query attention + RoPE        │
                       │ RMSNorm → h                                     │
                       │ x + FFN(h)                       (every layer)  │
                       │   + Σ_top2 w_e · Expert_e(h)     (expert layers)│
                       │     router(h) → softmax → top-2 → renormalize   │
                       └─────────────────────────────────────────────────┘
experts: intent · architecture · planning · build · test · deploy · memory · security · observability · cost
```

| Piece | Where | What it is |
|---|---|---|
| Prime Core | `ArtemisLM` (embedding, `Attention`, `FeedForward`, `RMSNorm`, head) | Shared decoder-only transformer. Runs on every token and generates every token. |
| Router | `ExpertMixture.router` | `Linear(d_model → 10)` per expert layer, learned. Softmax, keep top 2, renormalize the two weights to sum to 1. |
| Experts | `ExpertMixture.experts[0..9]` | Ten SwiGLU feed-forward modules, each with its own weights. Their weighted outputs are added to the Prime Core's residual stream. |
| Expert layers | `ModelConfig.moe_layers` / `moe_every` | Selected layers only (prototype: 1, 3, 5 of 6; GPU sizes: every second layer). |
| Load balancing | `balance_coef` × Switch-style loss | `10 · Σ_e (share of routing slots_e × mean router probability_e)`; 1.0 = perfectly even. |
| Router z-loss | `router_z_coef` × mean(logsumexp(logits)²) | Keeps router logits from growing without bound. |
| Routing supervision | `--route-coef` × NLL of the labeled domain's expert | Only on tokens whose domain is labeled. Teaches the router which expert a domain belongs to; at inference no labels exist and the router decides alone. |
| Diagnostics | `metrics.jsonl` | Per step: each loss term, per-layer expert load, router entropy; at eval steps: router and per-expert gradient norms, validation loss. |

Attention, normalization and positional encoding: RMSNorm (pre-norm), rotary embeddings, grouped-query
attention (`n_kv_heads < n_heads`), SwiGLU in the Prime Core FFN and in every expert.

## What is separate from the weights

| Lives in the application, not the model | File |
|---|---|
| External persistent memory (per-account notes, BM25 retrieval, inserted into the prompt in a block labeled as retrieved memory) | `artemis/memory.py`, `artemis/orchestrator.py` |
| Permissions (API keys, plans, which tiers and brains an account may use) | `artemis/business.py`, `artemis/server.py` |
| Resource limits (messages per day, API rate and credit, memory size, history length, message length) | `artemis/business.py`, `artemis/memory.py`, `artemis/server.py` |
| Tool execution (web search, web pages, sandboxed Python), with per-plan permission and daily caps | `artemis/tools.py`, `artemis/orchestrator.py`, `artemis/business.py` |
| Deployment controls (GPU start/stop, $2M monthly cap, promotion gate) | `infra/azure/*`, `artemis/engine.py` |
| Planet brains (Apollo … Pluto): product personas = a system prompt and optionally a LoRA adapter on the Prime Core | `brains/*.yaml`, `artemis/orchestrator.py`, `artemis/specialize.py` |

The **memory expert** is a neural module that learned to process memory-related text; it stores nothing between
conversations. The **memory store** is a database table; it is never trained into the weights.

## Conflict with earlier instructions

Earlier, the "ten brains" were the planet brains: prompts plus LoRA adapters, chosen by a keyword router. Those are
role labels around one model, not neural experts. The current specification governs the neural architecture: the
ten experts are trainable modules inside the model, chosen per token by a learned router. The planet brains remain
as product personas (the website and plans reference them), are documented as such, and are not counted as
experts. LoRA adapters attach only to the Prime Core, never to experts or routers.

## Checkpoints

Format 2 (`artemis/checkpoint.py`): `format_version`, `arch: artemis-prime-moe`, config, all tensors, optimizer,
step, RNG state, and a `params` manifest of every tensor's shape. Loading is strict: a missing or unexpected
tensor raises an error naming it.

Format 1 (dense, before the neural core) is refused with instructions to migrate:
`python -m artemis.checkpoint migrate --src old.pt --dst new.pt --moe-every 2`. The migration writes a new file
and a `.migration.json` record (source SHA-256, tensors copied, tensors created and how, tensors dropped: always
none, optimizer handling), never touches the source, copies every format-1 tensor unchanged, creates routers
(normal 0.02) and experts (gate/up normal 0.02, down **zeros**, so the migrated model's outputs equal the
original's), and carries AdamW moments over by parameter name.

**Existing trained weights:** none exist. No foundation model has been trained (GPU quota is 0); the only
checkpoints ever produced were test and prototype runs. So nothing needed migrating, and the migration is
verified on format-1 checkpoints written exactly as the old trainer wrote them. The tokenizer code and
vocabulary layout are unchanged; no production tokenizer exists yet either.

Serving export: dense models export as Llama; models with experts export as Qwen2-MoE (router, routed experts
with top-k and renormalized weights, shared FFN, dense-only layers). Logits match ours on CPU (test below).

## Verification (tests/test_neural_core.py)

| Claim | Test |
|---|---|
| Ten experts and routers are registered parameters, grouped by name | `test_ten_named_experts_and_registered_parameters` |
| Router keeps the top two and renormalizes; output equals the weighted sum of those experts | `test_router_picks_top_two_with_normalized_weights` |
| The forward pass calls the router and experts, and expert outputs change the logits | `test_forward_pass_calls_router_and_experts` |
| Router and selected experts receive gradients; unselected experts don't | `test_router_and_selected_experts_receive_gradients` |
| The optimizer holds exactly the trainable parameters (fails if any router/expert is left out) | `test_optimizer_holds_every_trainable_parameter` |
| Save and reload preserve config, every tensor and outputs; loading is strict | `test_save_reload_preserves_architecture_and_parameters` |
| Format-1 is refused, migrated to a new file, original unchanged, outputs equal, AdamW moments carried | `test_v1_checkpoint_is_refused_then_migrated_without_loss` |
| A migrated checkpoint resumes training with experts | `test_migrated_checkpoint_resumes_training` |
| Training lowers loss, logs routing, resumes; held-out routing beats chance; ablation runs | `test_training_learns_records_routing_and_resumes` |
| Stopping and resuming gives the same weights as an uninterrupted run | `test_stop_and_resume_matches_an_uninterrupted_run` |
| Exported Qwen2-MoE gives the same logits as our model | `test_moe_export_matches_qwen2_moe` |
| Local inference streams through the experts (routing trace) | `test_inference_streams_through_the_experts` |
| The API server loads the configured checkpoint and streams tokens through the router | `test_api_server_loads_the_checkpoint_and_streams_model_tokens` |
| LoRA adapters never wrap experts or routers | `test_lora_adapters_leave_experts_and_router_alone` |

## Prototype results

Run on 2026-10-03 with `infra/local/run-prototype.sh` on 4 CPU cores. Raw results: `docs/results/`.

**Setup.** Tokenizer: byte-level BPE, 4,096 tokens, trained on the corpus below. Data
(`proto-data-manifest.json`): ten domain files generated by `artemis/corpus.py` (Artemis AI original text,
3,000 documents each, label = domain) plus 80 CPython standard-library modules (PSF-2.0, unlabeled general text);
4.55M training tokens, 233K validation tokens, split by document. Every run: 2,000 steps × 16 × 256 tokens
(8.2M tokens), AdamW, peak LR 1e-3, cosine schedule, stopped at step 1,000 and resumed from the checkpoint.

| Run | Weights (total / per token) | Validation loss | Train time |
|---|---|---|---|
| `proto`: Prime Core + 10 experts, routing supervision 0.1 | 8.24M / 5.88M | 0.977 | 1,499 s |
| `proto-unsup`: same, routing supervision off (control) | 8.24M / 5.88M | 0.974 | 1,474 s |
| `proto-dense`: dense baseline, matched weights per token | 5.87M / 5.87M | **0.970** | **1,240 s** |

**The experts specialized (supervised run).** On held-out validation text, the router put each domain's own
expert in its top two for 99.6–99.9% of tokens at all three expert layers (chance: 20%). Removing one expert
from routing raised its own domain's loss by 1.16–1.81 nats and changed the other nine domains by at most
0.001. The ablation criterion (own-domain increase is largest and more than twice the others) holds for all ten.
At inference, a security prompt sent 40 of 40 generated tokens through the security expert at every expert
layer; a cost prompt sent 40 of 40 through the cost expert.

**Names alone did nothing (control).** With routing supervision off, the experts' names meant nothing: the hit rate
for each domain's namesake expert was 7–54% by layer (chance 20%), removing any expert changed any domain's
loss by at most 0.03, and the expert whose removal hurt a domain most was usually a differently named one.
The router still spread load and the model trained fine; the experts just didn't line up with their names.

**Against the dense baseline: no gain at this scale.** With the same weights per token and the same tokens, the
dense model reached slightly lower validation loss (0.970 vs 0.977) and lower loss on every domain (by
0.001–0.012), and trained 17% faster on CPU (the expert dispatch is a Python loop). So this prototype shows
that routing, specialization and ablation behave as designed. It does not show that experts make Artemis more
capable or more efficient. That has to be measured at GPU scale on real data, where experts usually pay off
through more total weights at the same cost per token.

Caveats: validation loss is dominated by the standard-library code (domain text loss is about 0.25); the domain
text is template-generated and easy; specialization here is induced by labeled routing supervision, which needs
labeled data at scale too.

## What is not demonstrated

- General capability. The prototype corpus is generated template text; the model learns those patterns, not engineering skill.
- Any GPU run, any model above 8M parameters, vLLM serving of the MoE export, LoRA adapters on the MoE export in vLLM.
- Specialization on real, licensed domain data, which needs that data and GPU time.
