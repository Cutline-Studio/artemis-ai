"""Artemis, the big brain: plans each request, routes sub-tasks to specialist brains, has Venus audit
the drafts, and merges one final answer.

This is the application layer around the model. The planet brains here are product personas (a system prompt,
optionally a LoRA adapter); they are not the neural experts. The ten neural experts and their learned router live
inside the model (artemis/model.py) and run on every token whichever persona is chosen.

Memory: for tiers with memory, notes saved for the account are retrieved from the external memory store
(artemis/memory.py) and placed in the prompt in a block marked as retrieved memory. Nothing is retrieved for guests.

Routing has two stages (Decision 5, "own brain"):
  1. Keyword router: deterministic scoring against each brain's keywords. Needs no model.
  2. Once the Artemis orchestrator model is trained, it writes the routing plan itself; the keyword
     router stays as the fallback when its plan is invalid.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from .backends import Backend, ModelNotReady
from .brains import Brain, load_brains
from .memory import MemoryStore, context_block
from .tools import TOOL_SPECS, Toolbox, ToolResult, parse_tool_call, result_block

# Product tiers (the ladder). Each tier is more capable than the one below it.
TIERS = {
    "gpt-1-base": {"max_specialists": 0, "audit": False, "memory": False},
    "gpt-2-sft": {"max_specialists": 1, "audit": False, "memory": False},
    "gpt-3-aligned-rag": {"max_specialists": 3, "audit": True, "memory": True},
    "gpt-4-asi-orchestrator": {"max_specialists": 10, "audit": True, "memory": True},
}
SUPPORT_BRAINS = {"venus", "mercury"}  # added by the tier rules, not by topic
MAX_TOOL_STEPS = 4  # tool calls per answer


@dataclass
class ToolAccess:
    """What one request may use: the toolbox, the tools its plan allows today, the customer's sandbox id, and
    hooks into the business engine that check and count each call."""
    toolbox: Toolbox
    allowed: list[str]
    session: str
    authorize: Callable[[str], None] = lambda name: None
    record: Callable[[str], None] = lambda name: None


class _CallFilter:
    """Streams the model's text but holds back anything that might be the start of a <tool_call>, and everything
    after one, so customers never see raw tool-call markup."""
    TAG = "<tool_call>"

    def __init__(self):
        self.buf, self.stopped = "", False

    def feed(self, piece: str) -> str:
        if self.stopped:
            return ""
        self.buf += piece
        i = self.buf.find(self.TAG)
        if i >= 0:
            out, self.buf, self.stopped = self.buf[:i], "", True
            return out
        keep = next((k for k in range(min(len(self.TAG) - 1, len(self.buf)), 0, -1) if self.TAG.startswith(self.buf[-k:])), 0)
        out, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
        return out

    def flush(self) -> str:
        out, self.buf = ("" if self.stopped else self.buf), ""
        return out


def _display_args(call: dict) -> dict:
    """Tool arguments as the customer sees them (code is shown in the result instead)."""
    a = call.get("arguments", {})
    return {k: (str(v)[:300]) for k, v in a.items() if k != "code"}


@dataclass
class Plan:
    brains: list[str]
    reason: str
    router: str


@dataclass
class Result:
    answer: str
    plan: Plan
    drafts: dict[str, str] = field(default_factory=dict)
    audit: str | None = None
    latency_ms: int = 0
    status: str = "ok"
    memories_used: int = 0
    tools_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"status": self.status, "answer": self.answer, "brains": self.plan.brains, "router": self.plan.router,
                "reason": self.plan.reason, "audit": self.audit, "latency_ms": self.latency_ms,
                "memories_used": self.memories_used, "tools_used": self.tools_used}


def keyword_route(text: str, brains: dict[str, Brain], limit: int) -> Plan:
    t = text.lower()
    scores = {}
    for b in brains.values():
        if b.kind != "specialist" or b.id in SUPPORT_BRAINS:
            continue
        hits = [k for k in b.keywords if re.search(rf"\b{re.escape(k)}", t)]
        if hits:
            scores[b.id] = (len(hits), hits)
    ranked = sorted(scores, key=lambda k: -scores[k][0])[:limit]
    reason = "; ".join(f"{k}: {', '.join(scores[k][1][:3])}" for k in ranked) or "no specialist keywords; Artemis answers directly"
    return Plan(ranked, reason, "keyword")


class Artemis:
    def __init__(self, backend: Backend, brains: dict[str, Brain] | None = None, use_model_router: bool = False,
                 memory: MemoryStore | None = None):
        self.backend = backend
        self.brains = brains or load_brains()
        self.use_model_router = use_model_router
        self.memory = memory
        self.pool = ThreadPoolExecutor(max_workers=10)

    def plan(self, text: str, tier: str) -> Plan:
        limit = TIERS[tier]["max_specialists"]
        if limit == 0:
            return Plan([], "base tier answers directly", "tier")
        if self.use_model_router:
            try:
                return self._model_route(text, limit)
            except (ModelNotReady, ValueError):
                pass  # fall back to the keyword router
        return keyword_route(text, self.brains, limit)

    def _model_route(self, text: str, limit: int) -> Plan:
        menu = "\n".join(f"{b.id}: {b.role}" for b in self.brains.values() if b.kind == "specialist" and b.id not in SUPPORT_BRAINS)
        reply = self.backend.generate("artemis", self.brains["artemis"].system_prompt(), [{"role": "user", "content":
            f"Choose at most {limit} brains for this request. Reply with JSON "
            f'{{"brains": [...], "reason": "..."}} using only these ids:\n{menu}\n\nRequest: {text}'}], max_tokens=200)
        data = json.loads(reply[reply.index("{"): reply.rindex("}") + 1])
        chosen = [b for b in data.get("brains", []) if b in self.brains and b not in SUPPORT_BRAINS][:limit]
        if not isinstance(data.get("brains"), list):
            raise ValueError("router reply has no brain list")
        return Plan(chosen, str(data.get("reason", "")), "model")

    def _prepare(self, text, tier, history, brain, account):
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}; choose one of {list(TIERS)}")
        if brain and (brain not in self.brains or self.brains[brain].kind != "specialist"):
            raise ValueError(f"unknown brain {brain!r}")
        rules = TIERS[tier]
        messages = (history or []) + [{"role": "user", "content": text}]
        plan = Plan([brain], "chosen by user", "direct") if brain else self.plan(text, tier)
        notes = self.memory.retrieve(account, text) if rules["memory"] and self.memory and account else []
        block = context_block(notes)
        sub = messages if not block else messages[:-1] + [{"role": "user", "content": f"{block}\n\n{text}"}]
        return rules, plan, sub, len(notes)

    def _audit(self, text, answer, sub):
        audit = self._ask("venus", [{"role": "user", "content": f"Request: {text}\n\nDraft answer:\n{answer}\n\nReply PASS or FAIL with reasons."}], 300)
        fixed = None
        if audit.strip().upper().startswith("FAIL"):
            fixed = self._ask("artemis", sub + [{"role": "assistant", "content": answer},
                                               {"role": "user", "content": f"Venus found problems; fix them:\n{audit}"}])
        return audit, fixed

    def handle(self, text: str, tier: str = "gpt-4-asi-orchestrator", history: list[dict] | None = None,
               brain: str | None = None, account: str | None = None, tools: "ToolAccess | None" = None) -> Result:
        """Answer one request. `brain` sends it straight to one specialist (using a single planet GPT)."""
        result = None
        for ev in self._run(text, tier, history, brain, account, tools):
            if ev["type"] == "result":
                result = ev["result"]
        return result

    def handle_stream(self, text: str, tier: str = "gpt-4-asi-orchestrator", history: list[dict] | None = None,
                      brain: str | None = None, account: str | None = None, tools: "ToolAccess | None" = None):
        """Like handle, but yields events as they happen: plan, tool_call / tool_result, token (text pieces of the
        final answer), replace (the audit rejected the streamed answer; here is the fixed one), then done or training."""
        for ev in self._run(text, tier, history, brain, account, tools):
            if ev["type"] == "result":
                r = ev["result"]
                if r.status == "training":
                    yield {"type": "training", "status": "training", "message": r.answer}
                else:
                    yield {"type": "done", **r.to_dict()}
            else:
                yield ev

    def _run(self, text, tier, history, brain, account, tools):
        t0 = time.monotonic()
        rules, plan, sub, n_mem = self._prepare(text, tier, history, brain, account)
        yield {"type": "plan", "brains": plan.brains, "router": plan.router, "reason": plan.reason, "memories_used": n_mem}
        drafts, used = {}, []
        try:
            if len(plan.brains) <= 1 and not (rules["audit"] and plan.brains):
                final = (plan.brains[0] if plan.brains else "artemis", sub)
            else:  # several specialists draft in parallel (no tools), then Artemis merges with tools
                futures = {b: self.pool.submit(self._ask, b, sub) for b in plan.brains}
                drafts = {b: f.result() for b, f in futures.items()}
                final = ("artemis", self._merge_messages(text, drafts))
            loop = self._respond(*final, tools)
            while True:
                try:
                    ev = next(loop)
                except StopIteration as stop:
                    answer = stop.value
                    break
                if ev["type"] == "tool_call":
                    used.append(ev["name"])
                yield ev
            audit = None
            if rules["audit"] and plan.brains:
                audit, fixed = self._audit(text, answer, sub)
                if fixed:
                    answer = fixed
                    yield {"type": "replace", "text": fixed}
            yield {"type": "result", "result": Result(answer, plan, drafts, audit, int((time.monotonic() - t0) * 1000),
                                                      memories_used=n_mem, tools_used=used)}
        except ModelNotReady as e:
            yield {"type": "result", "result": Result(str(e), plan, latency_ms=int((time.monotonic() - t0) * 1000),
                                                      status="training", memories_used=n_mem)}

    def _respond(self, brain_id: str, messages: list[dict], tools: "ToolAccess | None", max_tokens: int = 512):
        """One answer, using tools when allowed: the model may call a tool, see its result, and continue, up to
        MAX_TOOL_STEPS times. Yields token / tool_call / tool_result events; returns the visible answer text."""
        allowed = tools.allowed if tools else []
        system = self.brains[brain_id].system_prompt() + (tools.toolbox.instructions(allowed) if allowed else "")
        msgs, shown = list(messages), []
        for step in range(MAX_TOOL_STEPS + 1):
            filt = _CallFilter() if allowed else None
            raw = []
            for piece in self._stream_raw(brain_id, system, msgs, max_tokens):
                raw.append(piece)
                out = filt.feed(piece) if filt else piece
                if out:
                    yield {"type": "token", "text": out}
            reply = "".join(raw)
            call = parse_tool_call(reply) if allowed else None
            if call is None:
                tail = filt.flush() if filt else ""
                if tail:
                    yield {"type": "token", "text": tail}
                shown.append(reply)
                break
            cut = reply.index("<tool_call>")
            shown.append(reply[:cut])
            if step == MAX_TOOL_STEPS:
                break
            yield {"type": "tool_call", "name": call["name"], "arguments": _display_args(call)}
            result = self._run_tool(call, tools)
            yield {"type": "tool_result", "name": result.name, "ok": result.ok, **result.display}
            msgs += [{"role": "assistant", "content": reply[:reply.index("</tool_call>") + len("</tool_call>")]},
                     {"role": "user", "content": result_block(result)}]
        return "".join(shown).strip()

    def _run_tool(self, call: dict, tools: "ToolAccess") -> ToolResult:
        name = call["name"]
        if call.get("error"):
            return ToolResult(name or "unknown", False, f"error: {call['error']}", {"summary": "Tool call was malformed"})
        if name not in TOOL_SPECS:
            return ToolResult(name, False, f"error: there is no tool called {name!r}", {"summary": f"Unknown tool {name}"})
        try:
            tools.authorize(name)
        except Exception as e:  # PaymentRequired / RateLimited from the business engine
            return ToolResult(name, False, f"error: not available: {e}", {"summary": str(e)})
        if name not in tools.allowed:
            return ToolResult(name, False, "error: this tool is not available right now", {"summary": f"{name} is not available"})
        tools.record(name)
        return tools.toolbox.execute(name, call["arguments"], tools.session)

    def _stream_raw(self, brain_id: str, system: str, messages: list[dict], max_tokens: int):
        stream = getattr(self.backend, "stream", None)
        if stream is None:  # backend without streaming: one piece
            yield self.backend.generate(brain_id, system, messages, max_tokens)
            return
        yield from stream(brain_id, system, messages, max_tokens)

    def _ask(self, brain_id: str, messages: list[dict], max_tokens: int = 512) -> str:
        return self.backend.generate(brain_id, self.brains[brain_id].system_prompt(), messages, max_tokens)

    def _merge_messages(self, text: str, drafts: dict[str, str]) -> list[dict]:
        parts = "\n\n".join(f"[{self.brains[b].name}]\n{d}" for b, d in drafts.items())
        return [{"role": "user", "content": f"Request: {text}\n\nSpecialist drafts:\n{parts}\n\nWrite one final answer."}]

    def _merge(self, text: str, drafts: dict[str, str]) -> str:
        return self._ask("artemis", self._merge_messages(text, drafts))
