import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from artemis.backends import NotReadyBackend
from artemis.brains import SPECIALIST_IDS, load_brains
from artemis.orchestrator import Artemis, keyword_route
from artemis.business import Business
from artemis.server import make_handler


class FakeBackend:
    """Test double that records calls; stands in for Artemis's own server."""

    def __init__(self, venus="PASS"):
        self.calls, self.venus = [], venus

    def generate(self, brain, system, messages, max_tokens=512):
        self.calls.append(brain)
        assert system.startswith("You are")
        if brain == "venus":
            return self.venus
        return f"{brain} draft"


def test_eleven_brains_with_planet_names():
    brains = load_brains()
    assert set(brains) == {"artemis", *SPECIALIST_IDS}
    assert brains["artemis"].kind == "orchestrator"
    assert {b.name for b in brains.values() if b.kind == "specialist"} == {
        "Apollo", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"}
    assert "Cutline" not in "".join(b.system_prompt() for b in brains.values())


@pytest.mark.parametrize("text,expected", [
    ("Fix this python bug in my function", "saturn"),
    ("Calculate the probability of rolling two sixes", "uranus"),
    ("Make a launch plan and roadmap for Q1", "apollo"),
    ("Research and compare sources on battery chemistry", "neptune"),
    ("Is this release ready to ship?", "mars"),
    ("Design the database and cloud architecture to scale", "pluto"),
    ("What's in this screenshot?", "jupiter"),
    ("Rewrite this email in a friendlier tone", "earth"),
])
def test_keyword_router_picks_the_right_planet(text, expected):
    assert keyword_route(text, load_brains(), 3).brains[0] == expected


def test_top_tier_routes_audits_and_merges():
    fb = FakeBackend()
    r = Artemis(fb).handle("Fix the bug in this python script and calculate its runtime percent", "gpt-4-asi-orchestrator")
    assert r.status == "ok" and {"saturn", "uranus"} <= set(r.plan.brains)
    assert "mercury" not in fb.calls and "venus" in fb.calls and fb.calls[-2] == "artemis"  # memory comes from the store, not a model call
    assert r.audit == "PASS"


def test_venus_fail_triggers_a_fix():
    fb = FakeBackend(venus="FAIL: wrong number")
    Artemis(fb).handle("calculate 2+2", "gpt-4-asi-orchestrator")
    assert fb.calls[-1] == "artemis" and fb.calls[-2] == "venus"


def test_lower_tiers_use_fewer_brains():
    fb = FakeBackend()
    r = Artemis(fb).handle("Fix this python bug and calculate the probability", "gpt-2-sft")
    assert len(r.plan.brains) == 1 and "venus" not in fb.calls and "mercury" not in fb.calls
    fb2 = FakeBackend()
    r2 = Artemis(fb2).handle("anything at all", "gpt-1-base")
    assert r2.plan.brains == [] and fb2.calls == ["artemis"]


def test_direct_planet_gpt():
    fb = FakeBackend()
    r = Artemis(fb).handle("hello", "gpt-2-sft", brain="neptune")
    assert r.plan.brains == ["neptune"] and r.answer == "neptune draft"
    with pytest.raises(ValueError):
        Artemis(fb).handle("hello", "gpt-2-sft", brain="artemis")


def test_not_ready_reports_training():
    r = Artemis(NotReadyBackend()).handle("hello", "gpt-4-asi-orchestrator")
    assert r.status == "training" and "training" in r.answer


def test_http_api():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Artemis(FakeBackend()), Business(":memory:")))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        brains = json.load(urllib.request.urlopen(f"{base}/v1/brains"))["brains"]
        assert len(brains) == 11
        req = urllib.request.Request(f"{base}/v1/chat", data=json.dumps({"message": "write python code", "tier": "gpt-2-sft"}).encode(),
                                     headers={"Content-Type": "application/json", "Origin": "https://cutline-industries.studio"})
        with urllib.request.urlopen(req) as resp:
            body = json.load(resp)
            assert resp.headers["Access-Control-Allow-Origin"] == "https://cutline-industries.studio"
        assert body["status"] == "ok" and body["brains"] == ["saturn"]
        bad = urllib.request.Request(f"{base}/v1/chat", data=b'{"message": ""}', headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 400
    finally:
        srv.shutdown()
