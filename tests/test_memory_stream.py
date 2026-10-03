"""External memory (separate from the weights) and the streaming chat endpoint."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from artemis.business import Business
from artemis.memory import MAX_CHARS, MemoryStore, context_block
from artemis.orchestrator import Artemis
from artemis.server import make_handler


class Recorder:
    """Stands in for Artemis's own model; records prompts and streams its reply in pieces."""

    def __init__(self):
        self.prompts = []

    def generate(self, brain, system, messages, max_tokens=512):
        self.prompts.append(messages[-1]["content"])
        return "PASS" if brain == "venus" else f"{brain} reply"

    def stream(self, brain, system, messages, max_tokens=512):
        self.prompts.append(messages[-1]["content"])
        yield from ["Hello", " from", f" {brain}"]


def test_memory_is_per_account_and_retrieved_by_relevance():
    mem = MemoryStore(Business(":memory:").db)
    a = mem.add("acct:a", "The checkout service deploys on Fridays and is owned by the payments team.")
    mem.add("acct:a", "Prefers short answers with commands to paste.")
    mem.add("acct:b", "Checkout secret plans for account b only.")
    got = mem.retrieve("acct:a", "when does checkout deploy?")
    assert [n["id"] for n in got][:1] == [a]
    assert all("account b" not in n["text"] for n in got)
    assert mem.retrieve("acct:a", "zebra") == []
    assert mem.delete("acct:a", a) and not mem.delete("acct:b", a)
    with pytest.raises(ValueError):
        mem.add("acct:a", "x" * (MAX_CHARS + 1))
    block = context_block(got)
    assert block.startswith("[Retrieved memory") and "not part of the model's training" in block


def test_orchestrator_puts_memory_in_context_only_for_accounts():
    biz = Business(":memory:")
    mem = MemoryStore(biz.db)
    mem.add("acct:a", "The user's main service is called ledger.")
    rec = Recorder()
    app = Artemis(rec, memory=mem)
    r = app.handle("What is my ledger service?", "gpt-3-aligned-rag", account="acct:a")
    assert r.memories_used == 1 and any("[Retrieved memory" in p and "ledger" in p for p in rec.prompts)
    rec.prompts.clear()
    r = app.handle("What is my ledger service?", "gpt-3-aligned-rag")  # guest: no account, no memory
    assert r.memories_used == 0 and not any("Retrieved memory" in p for p in rec.prompts)
    rec.prompts.clear()
    app.handle("What is my ledger service?", "gpt-2-sft", account="acct:a")  # tier without memory
    assert not any("Retrieved memory" in p for p in rec.prompts)


def _server(backend):
    biz = Business(":memory:")
    from artemis.memory import MemoryStore
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Artemis(backend, memory=MemoryStore(biz.db)), biz))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, biz, f"http://127.0.0.1:{srv.server_port}"


def _events(base, body, headers=None):
    req = urllib.request.Request(base + "/v1/chat/stream", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req) as r:
        assert r.headers["Content-Type"] == "text/event-stream"
        text = r.read().decode()
    return [json.loads(chunk.split("data: ", 1)[1]) for chunk in text.strip().split("\n\n")]


def test_stream_endpoint_sends_plan_tokens_done():
    srv, biz, base = _server(Recorder())
    try:
        evs = _events(base, {"message": "hi there", "history": [{"role": "user", "content": "earlier"}, {"role": "bogus", "content": "x"}]})
        assert evs[0]["type"] == "plan"
        tokens = [e["text"] for e in evs if e["type"] == "token"]
        assert "".join(tokens) == "Hello from artemis" and len(tokens) == 3
        assert evs[-1]["type"] == "done" and evs[-1]["answer"] == "Hello from artemis" and evs[-1]["plan"] == "free"
    finally:
        srv.shutdown()


def test_stream_reports_training_when_no_model():
    from artemis.backends import NotReadyBackend
    srv, biz, base = _server(NotReadyBackend())
    try:
        evs = _events(base, {"message": "hi"})
        assert evs[-1]["type"] == "training" and "still training" in evs[-1]["message"]
        req = urllib.request.Request(base + "/v1/chat/stream", data=b'{"message": ""}', headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 400
    finally:
        srv.shutdown()


def test_memory_api_requires_an_api_key():
    srv, biz, base = _server(Recorder())
    biz.grant_api_credit("acct:dev", 5)
    key = biz.create_api_key("acct:dev")

    def call(method, path, body=None, auth=True):
        req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body else None,
                                     headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if auth else {})})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)
    try:
        assert call("POST", "/v1/memory", {"text": "remember me"}, auth=False)[0] == 401
        code, body = call("POST", "/v1/memory", {"text": "Project codename is Falcon."})
        assert code == 200
        assert [m["text"] for m in call("GET", "/v1/memory")[1]["memories"]] == ["Project codename is Falcon."]
        assert call("DELETE", f"/v1/memory/{body['id']}")[1] == {"deleted": True}
        assert call("GET", "/v1/memory")[1]["memories"] == []
    finally:
        srv.shutdown()
