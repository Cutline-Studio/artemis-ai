"""Regression checks with simulated inference; these do not certify trained model quality."""
import json
import io
import urllib.error
import urllib.request
import urllib.parse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from unittest.mock import patch

import pytest

from artemis.backends import ArtemisServerBackend, ModelUnavailable, NotReadyBackend
from artemis.business import Business
from artemis.orchestrator import Artemis, ToolAccess
from artemis.server import make_handler
from artemis.smoke_chat import check, conversation
from artemis.tools import LocalCodeRunner, Toolbox
from test_tools import FakeSearch, ScriptedModel, call


@contextmanager
def server(handler):
    """Exercise real HTTP request parsing and handlers in memory, without TCP sockets."""
    base = f"http://test-{id(handler)}"
    previous = urllib.request.urlopen

    class Socket:
        def __init__(self, incoming):
            self.incoming, self.output = io.BytesIO(incoming), bytearray()

        def makefile(self, *args):
            return self.incoming

        def sendall(self, data):
            self.output.extend(data)

    def open_request(req, *args, **kwargs):
        if isinstance(req, str):
            req = urllib.request.Request(req)
        if not req.full_url.startswith(base + "/"):
            return previous(req, *args, **kwargs)
        url = urllib.parse.urlsplit(req.full_url)
        path = url.path + ("?" + url.query if url.query else "")
        body = req.data or b""
        headers = {**dict(req.header_items()), "Content-Length": str(len(body))}
        raw = f"{req.get_method()} {path} HTTP/1.0\r\n"
        raw += "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n"
        sock = Socket(raw.encode() + body)
        handler(sock, ("127.0.0.1", 1234), None)
        head, _, payload = bytes(sock.output).partition(b"\r\n\r\n")
        code = int(head.split(b" ", 2)[1])
        response = io.BytesIO(payload)
        response.status = code
        if code >= 400:
            raise urllib.error.HTTPError(req.full_url, code, "test response", {}, response)
        return response

    with patch("urllib.request.urlopen", open_request):
        yield base


def request(base, path, body=None, headers=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        response = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        response = e
    with response:
        return response.status, response.read().decode()


def events(text):
    return [json.loads(part.split("data: ", 1)[1]) for part in text.strip().split("\n\n")]


@pytest.mark.parametrize("reply,status", [("", "empty_response"),
    ('<tool_call>{"name":"web_search"', "tool_protocol_error")])
def test_incomplete_output_has_error_terminal_event(reply, status):
    app = Artemis(ScriptedModel([reply]))
    access = ToolAccess(Toolbox(FakeSearch(), None), ["web_search"], "test")
    evs = list(app.handle_stream("hi", "gpt-1-base", tools=access))
    assert evs[-1]["type"] == "error" and evs[-1]["status"] == status
    assert not any(e["type"] == "done" for e in evs)
    assert "<tool_call>" not in "".join(e["text"] for e in evs if e["type"] == "token")


def test_unavailable_tools_do_not_leak_markup():
    app = Artemis(ScriptedModel([call("run_python", code="print(1)")]))
    evs = list(app.handle_stream("hi", "gpt-1-base"))
    assert evs[-1]["status"] == "tool_unavailable"
    assert not any(e["type"] == "token" for e in evs)


def test_live_smoke_checks_a_real_local_python_result_through_api_handlers():
    biz = Business(":memory:")
    biz.set_plan("paid", "enterprise")
    key = biz.create_api_key("paid")
    app = Artemis(ScriptedModel([call("run_python", code="print(6*7)"), "42"]))
    app.toolbox = Toolbox(None, LocalCodeRunner())
    with server(make_handler(app, biz)) as base:
        check(list(conversation(base, "multiply", key)), "run_python", expected=42)


@pytest.mark.parametrize("evs", [[], [{"type": "training", "status": "training"}],
    [{"type": "done", "status": "ok", "answer": "42"}],
    [{"type": "tool_result", "name": "run_python", "ok": True, "output": "41"},
     {"type": "done", "status": "ok", "answer": "42"}]])
def test_live_smoke_rejects_missing_model_skipped_tool_and_wrong_result(evs):
    with pytest.raises(ValueError):
        check(evs, "run_python", expected=42)


def test_audit_and_correction_keep_tool_evidence_and_history():
    model = ScriptedModel(["specialist draft", call("web_search", query="release"), "Wrong date.",
                           "FAIL: correct the date from the source", "Corrected date, citing https://example.com/py."])
    app = Artemis(model)
    result = app.handle("research release", "gpt-3-aligned-rag", brain="neptune",
                        history=[{"role": "user", "content": "Earlier context"}],
                        tools=ToolAccess(Toolbox(FakeSearch(), None), ["web_search"], "test"))
    assert result.status == "ok" and result.answer.startswith("Corrected")
    for seen in model.seen[1:]:
        assert any("Earlier context" in m["content"] for m in seen["messages"])
    for seen in model.seen[-2:]:
        assert any('<tool_result name="web_search"' in m["content"] for m in seen["messages"])


def test_spoofed_account_cannot_get_paid_tier_or_python():
    biz = Business(":memory:")
    biz.set_plan("paid", "plus")
    biz.grant_api_credit("paid", 1)
    key = biz.create_api_key("paid")
    app = Artemis(ScriptedModel([call("run_python", code="print(6*7)"), "42"]))
    app.toolbox = Toolbox(None, LocalCodeRunner())
    with server(make_handler(app, biz)) as base:
        code, _ = request(base, "/v1/chat", {"message": "hi", "tier": "gpt-3-aligned-rag"},
                          {"X-Artemis-Account": "paid"})
        assert code == 402
        assert request(base, "/v1/chat", {"message": "hi"}, {"Authorization": "Bearer invalid"})[0] == 401
        code, body = request(base, "/v1/chat/stream", {"message": "multiply", "tier": "gpt-1-base"},
                             {"Authorization": f"Bearer {key}"})
        evs = events(body)
        assert code == 200 and evs[-1]["type"] == "done" and evs[-1]["answer"] == "42"
        assert next(e for e in evs if e["type"] == "tool_result")["output"].strip() == "42"
        assert biz.used("paid", "tool_call") == 1 and biz.used("paid", "message") == 1


def test_training_is_503_and_does_not_charge_messages():
    biz = Business(":memory:")
    app = Artemis(NotReadyBackend())
    with server(make_handler(app, biz)) as base:
        code, text = request(base, "/v1/chat", {"message": "hi"})
        assert code == 503 and json.loads(text)["status"] == "training"
        status = json.loads(request(base, "/v1/status")[1])
        assert not status["serving"] and not status["configured"]
        assert biz.used("guest:127.0.0.1", "message") == 0


class Inference(BaseHTTPRequestHandler):
    """Simulates the HTTP contract, including optional usage and a terminal marker."""
    seen = []
    broken = False

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.seen.append(req)
        self.send_response(200)
        self.end_headers()
        if not req.get("stream"):
            self.wfile.write(json.dumps({"choices": [{"message": {"content": "Hello"}}]}).encode())
            return
        has_result = any(m["content"].startswith("<tool_result") for m in req["messages"])
        reply = "42" if has_result else call("run_python", code="print(6*7)")
        for i in range(0, len(reply), 2):
            chunk = {"choices": [{"delta": {"content": reply[i:i+2]}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b'data: {"choices": [], "usage": {"total_tokens": 10}}\n\n')
        if not self.broken:
            self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


def test_http_inference_to_python_to_customer_stream():
    Inference.seen = []
    with server(Inference) as inference:
        backend = ArtemisServerBackend(inference, timeout=2)
        app = Artemis(backend)
        app.toolbox = Toolbox(None, LocalCodeRunner())
        biz = Business(":memory:")
        biz.set_plan("paid", "enterprise")
        key = biz.create_api_key("paid")
        with server(make_handler(app, biz)) as base:
            status = json.loads(request(base, "/v1/status")[1])
            assert status["serving"] and status["quality"] == "unverified"
            _, text = request(base, "/v1/chat/stream", {"message": "multiply", "tier": "gpt-1-base"},
                              {"Authorization": f"Bearer {key}"})
            evs = events(text)
            assert evs[-1]["type"] == "done" and evs[-1]["answer"] == "42"
            assert next(e for e in evs if e["type"] == "tool_result")["ok"]
            assert "<tool_result" in Inference.seen[-1]["messages"][-1]["content"]


def test_interrupted_inference_stream_is_not_success(monkeypatch):
    monkeypatch.setattr(Inference, "broken", True)
    with server(Inference) as base:
        with pytest.raises(ModelUnavailable, match="ended early"):
            list(ArtemisServerBackend(base, timeout=2).stream("artemis", "", []))


def test_unreachable_inference_is_unavailable_not_training():
    class Offline(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_error(503)

        def log_message(self, *args):
            pass

    with server(Offline) as base:
        backend = ArtemisServerBackend(base, timeout=1)
        assert backend.readiness() == {"configured": True, "serving": False,
                                       "status": "unavailable", "quality": "unverified"}
        biz = Business(":memory:")
        with server(make_handler(Artemis(backend), biz)) as api:
            code, body = request(api, "/v1/chat", {"message": "hi"})
            assert code == 503 and json.loads(body)["status"] == "unavailable"
            _, body = request(api, "/v1/chat/stream", {"message": "hi"})
            assert events(body)[-1]["type"] == "error"
            assert biz.used("guest:127.0.0.1", "message") == 0
