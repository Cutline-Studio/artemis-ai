"""Artemis AI API server.

Consumer (the website and apps):
  POST /v1/chat     {"message": "...", "tier": optional, "brain": optional, "history": [...]}
                    optional Authorization: Bearer art-... authenticates the account; otherwise a free guest
  POST /v1/chat/stream   same body; answers as Server-Sent Events: plan, tool_call/tool_result..., token..., replace?,
                         done | training. Chat can use tools (web search, reading pages, running Python) as the plan allows.
Developer API (OpenAI-compatible request/response shape):
  POST /v1/chat/completions   Authorization: Bearer art-...   {"model": "artemis-aligned", "messages": [...]}
  POST /v1/waitlist           {"email": "..."}
  GET/POST /v1/memory, DELETE /v1/memory/<id>   Authorization: Bearer art-...   external memory notes
Public:
  GET  /v1/plans    plans and API prices
  GET  /v1/brains   the eleven brains
  GET  /v1/status   whether Artemis's own model is serving yet

ARTEMIS_DATABASE_URL (PostgreSQL) stores accounts, keys, usage and the waitlist; without it a local SQLite file is used.
ARTEMIS_INFERENCE_URL points at Artemis's own vLLM server; ARTEMIS_CHECKPOINT + ARTEMIS_TOKENIZER instead load a
checkpoint into this process (artemis/infer.py). With neither, every answer has status "training".
The X-Artemis-Account header is ignored for permissions. Memory endpoints require an API key.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .backends import ArtemisServerBackend, LocalBackend, NotReadyBackend
from .business import Business, PaymentRequired, RateLimited, Unauthorized
from .memory import MemoryStore
from .orchestrator import TIERS, Artemis, ToolAccess
from .tools import Toolbox, session_id

ALLOWED_ORIGINS = {o.strip() for o in os.environ.get("ARTEMIS_ALLOWED_ORIGINS",
                   "https://cutline-industries.studio,https://cutline-industries.vercel.app").split(",")}
MAX_MESSAGE_CHARS = 8000
MAX_HISTORY = 20  # messages of earlier conversation sent with each request; older ones are dropped
TRUST_PROXY = os.environ.get("ARTEMIS_TRUST_PROXY") == "1"


def build_app() -> tuple[Artemis, Business]:
    url, ckpt = os.environ.get("ARTEMIS_INFERENCE_URL"), os.environ.get("ARTEMIS_CHECKPOINT")
    if url:
        backend = ArtemisServerBackend(url, os.environ.get("ARTEMIS_INFERENCE_KEY", ""))
    elif ckpt:
        backend = LocalBackend(ckpt, os.environ["ARTEMIS_TOKENIZER"])
    else:
        backend = NotReadyBackend()
    biz = Business(os.environ.get("ARTEMIS_DATABASE_URL") or os.environ.get("ARTEMIS_DB", "artemis.db"))
    app = Artemis(backend, use_model_router=os.environ.get("ARTEMIS_MODEL_ROUTER") == "1", memory=MemoryStore(biz.db))
    app.toolbox = Toolbox.from_env()
    return app, biz


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def make_handler(app: Artemis, biz: Business):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            origin = self.headers.get("Origin", "")
            if origin in ALLOWED_ORIGINS:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self.send_response(204)
            if self.headers.get("Origin", "") in ALLOWED_ORIGINS:
                self.send_header("Access-Control-Allow-Origin", self.headers["Origin"])
                self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Artemis-Account, Authorization")
            self.end_headers()

        def do_GET(self):
            if self.path == "/v1/plans":
                return self._send(200, biz.public_plans())
            if self.path == "/v1/brains":
                return self._send(200, {"brains": [{"id": b.id, "name": b.name, "title": b.title, "kind": b.kind,
                                                    "role": b.role} for b in app.brains.values()]})
            if self.path == "/v1/status":
                info = app.backend.model.info() if isinstance(app.backend, LocalBackend) else None
                toolbox = getattr(app, "toolbox", None)
                health = (app.backend.readiness() if hasattr(app.backend, "readiness") else
                          {"configured": True, "serving": False, "status": "unverified", "quality": "unverified"})
                return self._send(200, {**health, "tiers": list(TIERS),
                                        "tools": toolbox.status() if toolbox else {}, **({"model": info} if info else {})})
            if self.path == "/v1/memory":
                return self._guard(lambda: self._send(200, {"memories": app.memory.list(self._api_account())}))
            self._send(404, {"error": "not found"})

        def do_DELETE(self):
            if self.path.startswith("/v1/memory/"):
                return self._guard(lambda: self._send(200, {"deleted": app.memory.delete(self._api_account(), self.path.rsplit("/", 1)[1])}))
            self._send(404, {"error": "not found"})

        def _guard(self, fn):
            try:
                fn()
            except Unauthorized as e:
                self._send(401, {"error": str(e)})
            except ValueError as e:
                self._send(400, {"error": str(e)})

        def _api_account(self) -> str:
            auth = self.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                raise Unauthorized("missing API key")
            return biz.account_for_key(auth[7:])

        def _client(self) -> str:
            # Behind Azure Container Apps ingress the socket peer is the proxy; it appends the real client as the last X-Forwarded-For entry.
            fwd = self.headers.get("X-Forwarded-For", "") if TRUST_PROXY else ""
            return fwd.rsplit(",", 1)[-1].strip() or self.client_address[0]

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 64_000:
                raise OverflowError
            return json.loads(self.rfile.read(n) or b"{}")

        def do_POST(self):
            try:
                req = self._body()
                if self.path == "/v1/chat":
                    return self._chat(req)
                if self.path == "/v1/chat/completions":
                    return self._completions(req)
                if self.path == "/v1/chat/stream":
                    return self._chat_stream(req)
                if self.path == "/v1/memory":
                    return self._send(200, {"id": app.memory.add(self._api_account(), str(req.get("text", "")), "api")})
                if self.path == "/v1/waitlist":
                    added = biz.join_waitlist(str(req.get("email", "")), f"ip:{self._client()}", str(req.get("source", "website"))[:40])
                    return self._send(200, {"ok": True, "already_joined": not added})
                self._send(404, {"error": "not found"})
            except OverflowError:
                self._send(413, {"error": "request too large"})
            except Unauthorized as e:
                self._send(401, {"error": str(e)})
            except PaymentRequired as e:
                self._send(402, {"error": str(e), "upgrade": "/v1/plans"})
            except RateLimited as e:
                self._send(429, {"error": str(e)})
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})

        def _chat_request(self, req: dict):
            account = self._api_account() if self.headers.get("Authorization") else f"guest:{self._client()}"
            msg = str(req.get("message", "")).strip()
            if not msg or len(msg) > MAX_MESSAGE_CHARS:
                raise ValueError(f"message must be 1-{MAX_MESSAGE_CHARS} characters")
            history = [{"role": m["role"], "content": str(m["content"])[:MAX_MESSAGE_CHARS]} for m in (req.get("history") or [])[-MAX_HISTORY:]
                       if isinstance(m, dict) and m.get("role") in ("user", "assistant") and "content" in m]
            return account, msg, history, biz.authorize_chat(account, req.get("tier"), req.get("brain"))

        def _tools(self, account: str) -> ToolAccess | None:
            toolbox = getattr(app, "toolbox", None)
            if toolbox is None:
                return None
            allowed = [t for t in biz.allowed_tools(account) if t in toolbox.available()]
            return ToolAccess(toolbox, allowed, session_id(account),
                              authorize=lambda name: biz.authorize_tool(account, name), record=lambda name: biz.record_tool(account, name))

        def _chat_stream(self, req: dict):
            account, msg, history, tier = self._chat_request(req)
            events = app.handle_stream(msg, tier, history, req.get("brain"), tools=self._tools(account))
            self.send_response(200)
            origin = self.headers.get("Origin", "")
            if origin in ALLOWED_ORIGINS:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                for ev in events:
                    if ev["type"] == "done":
                        biz.record_message(account)
                        ev.update(plan=biz.plan_of(account).id, tier=tier)
                    self.wfile.write(f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the visitor closed the page; stop generating

        def _chat(self, req: dict):
            account, msg, history, tier = self._chat_request(req)
            result = app.handle(msg, tier, history, req.get("brain"), tools=self._tools(account))
            if result.status == "ok":
                biz.record_message(account)
            code = 200 if result.status == "ok" else (503 if result.status in ("training", "unavailable") else 502)
            self._send(code, {**result.to_dict(), "plan": biz.plan_of(account).id, "tier": tier})

        def _completions(self, req: dict):
            auth = self.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                raise Unauthorized("missing API key")
            account = biz.account_for_key(auth[7:])
            model = req.get("model", "")
            tier = biz.authorize_api(account, model)
            messages = req.get("messages") or []
            if not messages or messages[-1].get("role") != "user":
                raise ValueError("messages must end with a user message")
            result = app.handle(str(messages[-1]["content"]), tier, messages[:-1], account=account)
            if result.status != "ok":
                return self._send(503 if result.status in ("training", "unavailable") else 502,
                                  {"error": {"type": result.status, "message": result.answer}})
            prompt_tokens = sum(approx_tokens(str(m.get("content", ""))) for m in messages)
            completion_tokens = approx_tokens(result.answer)
            cost = biz.record_api_call(account, model, prompt_tokens, completion_tokens)
            self._send(200, {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
                             "model": model, "choices": [{"index": 0, "finish_reason": "stop",
                                                          "message": {"role": "assistant", "content": result.answer}}],
                             "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                                       "total_tokens": prompt_tokens + completion_tokens, "cost_usd": round(cost, 6)}})

        def log_message(self, fmt, *args):  # keep request text out of logs
            pass

    return Handler


def main():
    app, biz = build_app()
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), make_handler(app, biz)).serve_forever()


if __name__ == "__main__":
    main()
