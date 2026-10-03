"""Domain-labeled prototype corpus for the ten experts, written by this program (Artemis AI original text).

Each document belongs to one expert domain and carries that label, so training can supervise the router and
evaluation can test whether each expert really specializes. Names, services and numbers are drawn from pools
shared by every domain, so a domain can't be recognized from the entity names alone; what differs is the kind of
work being described.

This corpus is for proving the architecture on small hardware. It is template text: a model trained on it
learns these patterns, not general engineering skill. Real training replaces it with licensed data.

python -m artemis.corpus --out data/proto --docs-per-domain 3000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .model import EXPERT_NAMES

LICENSE = "Artemis-AI-original"
SERVICES = ["checkout", "billing", "search", "auth", "inventory", "notifications", "gateway", "profile", "ledger", "reports",
            "scheduler", "uploads", "chat", "pricing", "catalog", "orders"]
STACKS = ["Python", "Go", "TypeScript", "Rust", "Java", "Kotlin"]
STORES = ["PostgreSQL", "Redis", "a message queue", "object storage", "a search index", "a key-value store"]
TEAMS = ["the platform team", "the payments team", "the mobile team", "the data team", "the web team", "the core team"]
REGIONS = ["eastus", "eastus2", "westus3", "northeurope", "westeurope", "southeastasia"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _n(r, a, b):
    return r.randint(a, b)


def intent(r):
    s, t = r.choice(SERVICES), r.choice(TEAMS)
    asks = [f"add rate limiting to the {s} service", f"make the {s} page load faster", f"move {s} to {r.choice(STACKS)}",
            f"find out why {s} failed last night", f"cut the monthly bill for {s}", f"let customers export their {s} data"]
    a = r.choice(asks)
    return r.choice([
        f"Request: \"{a}.\" Intent: the user wants to {a}. Goal type: {r.choice(['change', 'investigation', 'question'])}. "
        f"Missing details: deadline, owner, and whether {t} must approve. Ask before acting.",
        f"The message \"can you {a}?\" is a request for work, not a question about how it works. Primary intent: {a}. "
        f"Secondary intent: keep {s} stable while doing it. Confidence {_n(r, 60, 98)} percent.",
        f"Classify the request: \"{a}\". Category: {r.choice(['feature', 'bug', 'cost', 'data', 'performance'])}. "
        f"The user said 'please' and 'today', so urgency is high. Restate it back: you want me to {a} today.",
        f"Two readings of \"{a}\": the user may want it done now, or may want a proposal first. The words 'can you' point to a "
        f"proposal. Clarifying question: should I change {s} directly or send a plan to {t} first?",
    ])


def architecture(r):
    s, st, db = r.choice(SERVICES), r.choice(STACKS), r.choice(STORES)
    return r.choice([
        f"Design for {s}: a stateless {st} service behind the gateway, {db} for state, and a queue between {s} and "
        f"{r.choice(SERVICES)} so a slow consumer can't block writes. Scale horizontally; no shared in-memory state.",
        f"Component boundaries: {s} owns its data in {db}; other services call its API and never read its tables. "
        f"Events are published after commit using an outbox table. This keeps coupling low and schemas private.",
        f"Trade-off: a monolith is simpler to run, while splitting {s} out lets it scale alone. At {_n(r, 2, 40)} thousand "
        f"requests per minute the split pays off. Decision: extract {s}, keep {r.choice(SERVICES)} in the monolith.",
        f"Interface contract for {s}: versioned REST endpoints, idempotency keys on writes, pagination by cursor, and "
        f"backward-compatible fields only. Breaking changes need a new version and a deprecation window.",
    ])


def planning(r):
    s, t, d = r.choice(SERVICES), r.choice(TEAMS), r.choice(DAYS)
    return r.choice([
        f"Plan for {s}: step 1, write the design note by {d}. Step 2, build behind a flag. Step 3, test in staging. "
        f"Step 4, roll out to {_n(r, 1, 10)} percent. Step 5, full rollout after a week without incidents.",
        f"Milestones: week 1 scope and estimates, week 2 to {_n(r, 3, 6)} implementation, then hardening. "
        f"Dependencies: {t} must finish the schema change first. Critical path runs through the {s} migration.",
        f"Order of work: unblock {t} first, then the {s} changes, then cleanup. Each task has an owner and an estimate in days: "
        f"design {_n(r, 1, 3)}, build {_n(r, 3, 10)}, review {_n(r, 1, 2)}. Buffer {_n(r, 10, 30)} percent for unknowns.",
        f"Roadmap item: {s} rewrite. Prerequisites listed, risks ranked, and a checkpoint every {d}. If the checkpoint slips "
        f"twice, cut scope instead of moving the date.",
    ])


def build(r):
    s, st = r.choice(SERVICES), r.choice(STACKS)
    return r.choice([
        f"Build log for {s}: compiling {_n(r, 40, 400)} {st} files, resolving dependencies from the lockfile, "
        f"linking, and packaging a container image tagged {s}:{_n(r, 1, 9)}.{_n(r, 0, 20)}.{_n(r, 0, 9)}. Build finished in {_n(r, 30, 600)} seconds.",
        f"Implementation for {s}: add a handler function, parse the request body, validate fields, call the repository "
        f"method, and return the new record as JSON. Wire the route in the router file and add the dependency to the manifest.",
        f"The build failed: missing import in the {s} module. Fix: add the import, pin the library version, and regenerate "
        f"the lockfile. Rebuild with the cache cleared to confirm the image is reproducible.",
        f"Refactor {s}: extract the duplicated parsing code into one function, rename variables for clarity, and keep the "
        f"public signature unchanged so callers in {r.choice(SERVICES)} still compile.",
    ])


def test(r):
    s = r.choice(SERVICES)
    p, f = _n(r, 50, 900), _n(r, 0, 12)
    return r.choice([
        f"Test report for {s}: {p} passed, {f} failed, {_n(r, 0, 5)} skipped. Failing case: an empty cart returns 500 "
        f"instead of 400. Add a regression test that sends an empty body and asserts the status code.",
        f"Unit test: given a valid order, when {s} saves it, then the stored total equals the sum of line items. "
        f"Edge cases: zero items, negative quantity, and a currency mismatch must each be rejected.",
        f"Flaky test in {s}: it passes alone and fails in the full suite because it depends on test order. "
        f"Fix the shared fixture, then run the suite {_n(r, 20, 100)} times to confirm it is stable.",
        f"Coverage for {s} rose to {_n(r, 60, 95)} percent. Integration tests run against a real {r.choice(STORES)} in a "
        f"container. Property-based tests generate random inputs and check that encode then decode returns the input.",
    ])


def deploy(r):
    s, g = r.choice(SERVICES), r.choice(REGIONS)
    return r.choice([
        f"Deploy {s} version {_n(r, 1, 9)}.{_n(r, 0, 20)} to {g}: canary at {_n(r, 1, 10)} percent for 30 minutes, "
        f"watch error rate, then promote. Rollback command is ready; the previous revision stays warm.",
        f"Release checklist for {s}: migrations applied and backward compatible, feature flag off by default, "
        f"health checks green in staging, on-call engineer aware. Ship window: {r.choice(DAYS)} morning.",
        f"Blue-green rollout of {s} in {g}: start the green revision, shift traffic gradually, keep blue for an hour, "
        f"then drain it. If the green revision fails health checks, traffic returns to blue automatically.",
        f"Rollback of {s}: error rate jumped after the release, so traffic moved back to the last good revision in "
        f"{_n(r, 1, 5)} minutes. The bad image is blocked from redeploying until the fix is merged.",
    ])


def memory(r):
    s, t = r.choice(SERVICES), r.choice(TEAMS)
    return r.choice([
        f"Remember for next time: {t} owns {s}, prefers short design notes, and deploys on {r.choice(DAYS)}. "
        f"Saved to the project notes so the next conversation starts with this context.",
        f"Earlier in this project we decided to keep {s} on {r.choice(STORES)}. Recall that decision before suggesting a "
        f"change; the reason was cost and team familiarity.",
        f"Session summary: we fixed the {s} timeout, left the retry change for later, and agreed to revisit the cache "
        f"size. Store this summary and drop the raw transcript.",
        f"Retrieved note from {_n(r, 2, 30)} days ago: the user dislikes long answers and wants commands they can paste. "
        f"Apply it to this reply about {s}.",
    ])


def security(r):
    s = r.choice(SERVICES)
    return r.choice([
        f"Security review of {s}: user input reaches a SQL query through string formatting. Use parameterized queries. "
        f"Severity high; any visitor could read other customers' records.",
        f"Secrets in {s}: an API key is committed in the config file. Rotate the key now, move it to the key vault, "
        f"and scan history for other leaked credentials.",
        f"Access control: the {s} admin endpoint checks login but not role, so any signed-in user can call it. "
        f"Require the admin role and log every call. Apply least privilege to the service account too.",
        f"Threat model for {s}: attackers may replay requests, guess IDs, or flood the login form. Mitigations: "
        f"signed short-lived tokens, random IDs, rate limits of {_n(r, 5, 60)} per minute, and lockout after failures.",
    ])


def observability(r):
    s, g = r.choice(SERVICES), r.choice(REGIONS)
    return r.choice([
        f"Alert: {s} p99 latency is {_n(r, 400, 3000)} ms in {g}, above the {_n(r, 200, 500)} ms objective for "
        f"{_n(r, 5, 30)} minutes. Dashboard shows the spike started with a rise in database wait time.",
        f"Logs for {s}: structured JSON with request id, route, status and duration. Trace spans link the gateway call "
        f"to the {r.choice(STORES)} query, so one slow request can be followed end to end.",
        f"Metrics to add for {s}: request rate, error rate, and duration histograms per route, plus queue depth. "
        f"Service level objective: {r.choice(['99.9', '99.5', '99.95'])} percent of requests succeed over 30 days.",
        f"Incident timeline for {s}: error budget burned {_n(r, 10, 80)} percent in two hours. The trace showed retries "
        f"amplifying load. Root cause found from logs; added an alert on retry rate.",
    ])


def cost(r):
    s, g = r.choice(SERVICES), r.choice(REGIONS)
    m = _n(r, 200, 20000)
    return r.choice([
        f"Cost report: {s} spent ${m} last month in {g}, up {_n(r, 5, 60)} percent. Most of it is idle compute at night. "
        f"Scale to zero after hours to save about ${m // 3} a month.",
        f"Budget check for {s}: forecast ${m} against a ${m + _n(r, -500, 3000)} budget. Reserved capacity for the steady base "
        f"load would cut the bill by roughly {_n(r, 20, 45)} percent.",
        f"Unit economics: {s} costs ${_n(r, 1, 90)} per thousand requests. Caching the hot reads in "
        f"{r.choice(STORES)} would halve storage reads, the largest line item.",
        f"Spending guard: stop any job that would push {s} over ${m} this month, and ask for approval above that. "
        f"Right-size the database tier; it averages {_n(r, 5, 30)} percent CPU.",
    ])


GENERATORS = {"intent": intent, "architecture": architecture, "planning": planning, "build": build, "test": test,
              "deploy": deploy, "memory": memory, "security": security, "observability": observability, "cost": cost}
assert tuple(GENERATORS) == EXPERT_NAMES


def document(r: random.Random, domain: str) -> str:
    return " ".join(GENERATORS[domain](r) for _ in range(r.randint(2, 5)))


def write(out_dir: str | Path, docs_per_domain: int = 3000, seed: int = 0) -> list[dict]:
    """Writes one JSONL file per domain ({"text", "domain"} per line) and returns data-prep source entries."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources = []
    for i, d in enumerate(EXPERT_NAMES):
        r = random.Random(seed * 1000 + i)
        p = out / f"{d}.jsonl"
        p.write_text("\n".join(json.dumps({"text": document(r, d), "domain": d}) for _ in range(docs_per_domain)) + "\n")
        sources.append({"path": str(p), "license": LICENSE, "domain": d,
                        "origin": f"generated by artemis/corpus.py (seed {seed}, {docs_per_domain} documents)",
                        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    return sources


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs-per-domain", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(json.dumps(write(a.out, a.docs_per_domain, a.seed), indent=2))


if __name__ == "__main__":
    main()
