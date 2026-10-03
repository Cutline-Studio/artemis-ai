import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from artemis.business import Business, PaymentRequired, RateLimited, Unauthorized
from artemis.orchestrator import Artemis
from artemis.server import make_handler


class Fake:
    def generate(self, brain, system, messages, max_tokens=512):
        return "PASS" if brain == "venus" else f"{brain} answer"


@pytest.fixture
def biz():
    return Business(":memory:")


def test_plans_mirror_chatgpt_claude_grok(biz):
    assert set(biz.plans) == {"free", "plus", "pro", "team", "enterprise"}
    assert biz.cfg["plans"]["plus"]["price_usd_month"] == 20 and biz.cfg["plans"]["pro"]["price_usd_month"] == 200
    assert biz.plans["free"].tier == "gpt-2-sft" and biz.plans["pro"].tier == "gpt-4-asi-orchestrator"
    assert set(biz.cfg["api_models"]) == {"artemis-base", "artemis-sft", "artemis-aligned", "artemis-orchestrator"}


def test_free_plan_limits(biz):
    assert biz.authorize_chat("u1", None, None) == "gpt-2-sft"
    with pytest.raises(PaymentRequired):
        biz.authorize_chat("u1", "gpt-4-asi-orchestrator", None)
    with pytest.raises(PaymentRequired):
        biz.authorize_chat("u1", None, "pluto")
    for _ in range(30):
        biz.record_message("u1")
    with pytest.raises(RateLimited):
        biz.authorize_chat("u1", None, None)


def test_upgrading_unlocks_tiers_and_brains(biz):
    biz.set_plan("u2", "pro")
    assert biz.authorize_chat("u2", None, "pluto") == "gpt-4-asi-orchestrator"
    for _ in range(5000):
        biz.record_message("u2")
    biz.authorize_chat("u2", None, None)  # unlimited


def test_api_keys_credit_and_metering(biz):
    with pytest.raises(PaymentRequired):
        biz.create_api_key("dev")
    biz.grant_api_credit("dev", 5)
    key = biz.create_api_key("dev")
    assert key.startswith("art-") and biz.account_for_key(key) == "dev"
    cost = biz.record_api_call("dev", "artemis-orchestrator", 1_000_000, 100_000)
    assert cost == pytest.approx(5.0 + 2.5)
    with pytest.raises(PaymentRequired):
        biz.authorize_api("dev", "artemis-orchestrator")  # credit used up
    biz.revoke_api_key(key)
    with pytest.raises(Unauthorized):
        biz.account_for_key(key)


def _post(base, path, body, headers=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_http_plans_chat_and_developer_api():
    biz = Business(":memory:")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Artemis(Fake()), biz))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        plans = json.load(urllib.request.urlopen(base + "/v1/plans"))
        assert "pro" in plans["plans"]
        code, body = _post(base, "/v1/chat", {"message": "fix this python bug"}, {"X-Artemis-Account": "alice"})
        assert code == 200 and body["plan"] == "free" and body["tier"] == "gpt-2-sft" and body["brains"] == ["saturn"]
        code, _ = _post(base, "/v1/chat", {"message": "hi", "tier": "gpt-4-asi-orchestrator"}, {"X-Artemis-Account": "alice"})
        assert code == 402
        code, _ = _post(base, "/v1/chat/completions", {"model": "artemis-sft", "messages": [{"role": "user", "content": "hi"}]})
        assert code == 401
        biz.grant_api_credit("devco", 5)
        key = biz.create_api_key("devco")
        code, body = _post(base, "/v1/chat/completions", {"model": "artemis-aligned", "messages": [{"role": "user", "content": "calculate 2+2"}]},
                           {"Authorization": f"Bearer {key}"})
        assert code == 200 and body["object"] == "chat.completion" and body["usage"]["cost_usd"] > 0
        assert body["choices"][0]["message"]["role"] == "assistant"
    finally:
        srv.shutdown()


def test_waitlist(biz):
    assert biz.join_waitlist("Ada@Example.com", "ip:1") is True
    assert biz.join_waitlist("ada@example.com", "ip:1") is False  # stored once, case-insensitive
    assert biz.waitlist_count() == 1
    with pytest.raises(ValueError):
        biz.join_waitlist("not-an-email", "ip:1")
    for i in range(3):
        biz.join_waitlist(f"u{i}@example.com", "ip:1")
    with pytest.raises(RateLimited):
        biz.join_waitlist("late@example.com", "ip:1")


def test_state_survives_restart(tmp_path):
    db = str(tmp_path / "artemis.db")
    a = Business(db)
    a.set_plan("bob", "pro")
    a.join_waitlist("bob@example.com", "ip:2")
    b = Business(db)
    assert b.plan_of("bob").id == "pro" and b.waitlist_count() == 1


def test_http_waitlist():
    biz = Business(":memory:")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Artemis(Fake()), biz))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        assert _post(base, "/v1/waitlist", {"email": "x@example.com"}) == (200, {"ok": True, "already_joined": False})
        assert _post(base, "/v1/waitlist", {"email": "x@example.com"})[1]["already_joined"] is True
        assert _post(base, "/v1/waitlist", {"email": "bad"})[0] == 400
    finally:
        srv.shutdown()


def test_waitlist_limit_uses_forwarded_client(monkeypatch):
    import artemis.server as server
    monkeypatch.setattr(server, "TRUST_PROXY", True)
    biz = Business(":memory:")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Artemis(Fake()), biz))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        # One visitor hitting the limit must not block another visitor behind the same proxy.
        for i in range(5):
            assert _post(base, "/v1/waitlist", {"email": f"a{i}@example.com"}, {"X-Forwarded-For": "1.1.1.1"})[0] == 200
        assert _post(base, "/v1/waitlist", {"email": "a5@example.com"}, {"X-Forwarded-For": "1.1.1.1"})[0] == 429
        assert _post(base, "/v1/waitlist", {"email": "b@example.com"}, {"X-Forwarded-For": "2.2.2.2"})[0] == 200
        # A client-supplied entry can't dodge the limit: only the proxy's last entry counts.
        assert _post(base, "/v1/waitlist", {"email": "a6@example.com"}, {"X-Forwarded-For": "9.9.9.9, 1.1.1.1"})[0] == 429
    finally:
        srv.shutdown()
