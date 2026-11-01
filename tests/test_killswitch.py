"""The trainer's read-only kill switch gate: only a fresh, correctly signed running state lets training run."""
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from artemis import train
from artemis.killswitch import KillSwitch, KillSwitchStop


def keypair():
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    return key, pem


def signed(key, stopped=False, ts=None, sign_text=False):
    payload = base64.b64encode(json.dumps({"stopped": stopped, "ts": time.time() if ts is None else ts}).encode()).decode()
    message = payload.encode() if sign_text else base64.b64decode(payload)
    return {"payload": payload, "signature": base64.b64encode(key.sign(message, ec.ECDSA(hashes.SHA256()))).decode()}


@pytest.fixture
def site():
    """A local stand-in for the kill switch state URL; tests set .state to a callable returning the JSON body."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(server.state()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    server.url = f"http://127.0.0.1:{server.server_port}/api/state"
    yield server
    server.shutdown()


def test_fresh_signed_running_state_passes(site):
    key, pem = keypair()
    site.state = lambda: signed(key)
    assert KillSwitch(site.url, pem).check()["stopped"] is False


def test_signature_over_the_base64_payload_text_also_verifies(site):
    key, pem = keypair()
    site.state = lambda: signed(key, sign_text=True)
    assert KillSwitch(site.url, pem).check()["stopped"] is False


def test_owner_stop_is_refused(site):
    key, pem = keypair()
    site.state = lambda: signed(key, stopped=True)
    with pytest.raises(KillSwitchStop, match="stopped"):
        KillSwitch(site.url, pem).check()


def test_state_signed_by_another_key_is_refused(site):
    attacker, _ = keypair()
    _, owner_pem = keypair()
    site.state = lambda: signed(attacker)
    with pytest.raises(KillSwitchStop, match="signature"):
        KillSwitch(site.url, owner_pem).check()


def test_replayed_old_running_state_is_refused(site):
    key, pem = keypair()
    site.state = lambda: signed(key, ts=time.time() - 3600)
    with pytest.raises(KillSwitchStop, match="stale"):
        KillSwitch(site.url, pem).check()


def test_tampered_payload_is_refused(site):
    key, pem = keypair()
    good = signed(key)
    good["payload"] = base64.b64encode(json.dumps({"stopped": False, "ts": time.time(), "x": 1}).encode()).decode()
    site.state = lambda: good
    with pytest.raises(KillSwitchStop, match="signature"):
        KillSwitch(site.url, pem).check()


def test_unreachable_site_counts_as_stop():
    _, pem = keypair()
    with pytest.raises(KillSwitchStop, match="unreachable"):
        KillSwitch("http://127.0.0.1:9/api/state", pem, timeout_s=2).check()


def test_half_configured_environment_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTEMIS_KILLSWITCH_STATE_URL", "http://127.0.0.1:9/")
    monkeypatch.delenv("ARTEMIS_KILLSWITCH_STATE_PUBKEY", raising=False)
    with pytest.raises(KillSwitchStop):
        KillSwitch.from_env()


def _corpus(tmp_path):
    from artemis.corpus import write as write_corpus
    from artemis.data import prepare
    from artemis.tokenizer import train_tokenizer
    sources = write_corpus(tmp_path / "corpus", 60, seed=1)
    train_tokenizer([s["path"] for s in sources], 2048, tmp_path / "tok.json")
    prepare(sources, str(tmp_path / "tok.json"), str(tmp_path / "data"), val_fraction=0.1)
    return tmp_path / "data"


def _env(monkeypatch, tmp_path, site, pem):
    (tmp_path / "state.pub.pem").write_bytes(pem)
    monkeypatch.setenv("ARTEMIS_KILLSWITCH_STATE_URL", site.url)
    monkeypatch.setenv("ARTEMIS_KILLSWITCH_STATE_PUBKEY", str(tmp_path / "state.pub.pem"))


def test_training_does_not_start_while_the_owner_has_stopped_artemis(monkeypatch, tmp_path, site):
    key, pem = keypair()
    site.state = lambda: signed(key, stopped=True)
    _env(monkeypatch, tmp_path, site, pem)
    with pytest.raises(SystemExit, match="not starting"):
        train.main(["--size", "smoke", "--data", str(_corpus(tmp_path)), "--out", str(tmp_path / "run"), "--steps", "4"])
    assert not (tmp_path / "run" / "latest.pt").exists()


def test_a_stop_mid_run_saves_a_checkpoint_and_ends_training(monkeypatch, tmp_path, site):
    key, pem = keypair()
    calls = {"n": 0}

    def state():
        calls["n"] += 1
        return signed(key, stopped=calls["n"] > 1)  # running at launch, stopped at the first in-run check

    site.state = state
    _env(monkeypatch, tmp_path, site, pem)
    result = train.main(["--size", "smoke", "--data", str(_corpus(tmp_path)), "--out", str(tmp_path / "run"), "--steps", "20",
                         "--batch", "2", "--eval-every", "100", "--eval-batches", "1", "--ckpt-every", "100", "--killswitch-every", "3"])
    assert result["stopped"] == "the owner has stopped Artemis" and result["step"] == 3
    from artemis import checkpoint
    assert checkpoint.read(tmp_path / "run" / "latest.pt")["step"] == 3


def test_gpu_training_refuses_to_start_without_a_kill_switch(monkeypatch, tmp_path):
    monkeypatch.delenv("ARTEMIS_KILLSWITCH_STATE_URL", raising=False)
    monkeypatch.delenv("ARTEMIS_KILLSWITCH_STATE_PUBKEY", raising=False)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    with pytest.raises(SystemExit, match="needs the kill switch"):
        train.main(["--size", "smoke", "--data", str(tmp_path), "--out", str(tmp_path / "run")])
