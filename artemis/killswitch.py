"""Read-only check of the owner's kill switch before and during training.

The kill switch site publishes {"payload": base64(json {"stopped": bool, "ts": unix seconds}), "signature": base64(DER
ECDSA P-256 / SHA-256 over DOMAIN + the decoded payload)}. Training may run only while a fresh, correctly signed state
says stopped=false. Anything else (stopped, bad signature, stale or missing timestamp, unreachable site) counts as stop: the check fails closed. This module only
reads the public state URL. It never writes to, resets or enrolls anything on the kill switch.

Configure with ARTEMIS_KILLSWITCH_STATE_URL and ARTEMIS_KILLSWITCH_STATE_PUBKEY (path to the owner's state public key,
PEM). GPU training refuses to start without both.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
from dataclasses import dataclass

DOMAIN = b"artemis-killswitch-state-v1\n"  # signed prefix, so no other message signed by the owner key passes as a state
MAX_AGE_S = 300  # the site signs a new timestamp on every request, so an older state is a replay or a stuck cache


class KillSwitchStop(RuntimeError):
    """Training must not run: the owner stopped it, or the state could not be trusted."""


@dataclass
class KillSwitch:
    url: str
    pubkey_pem: bytes
    max_age_s: float = MAX_AGE_S
    timeout_s: float = 15.0

    @classmethod
    def from_env(cls) -> "KillSwitch | None":
        url, key = os.environ.get("ARTEMIS_KILLSWITCH_STATE_URL"), os.environ.get("ARTEMIS_KILLSWITCH_STATE_PUBKEY")
        if not url and not key:
            return None
        if not (url and key):
            raise KillSwitchStop("set both ARTEMIS_KILLSWITCH_STATE_URL and ARTEMIS_KILLSWITCH_STATE_PUBKEY")
        with open(key, "rb") as f:
            return cls(url, f.read())

    def fetch(self) -> dict:
        with urllib.request.urlopen(self.url, timeout=self.timeout_s) as r:  # fixed owner URL, read-only GET
            return json.loads(r.read())

    def verify(self, doc: dict, now: float | None = None) -> dict:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec

        try:
            raw, sig = base64.b64decode(doc["payload"], validate=True), base64.b64decode(doc["signature"], validate=True)
        except (KeyError, TypeError, ValueError) as e:
            raise KillSwitchStop(f"malformed kill switch state: {e}") from None
        key = serialization.load_pem_public_key(self.pubkey_pem)
        if not isinstance(key, ec.EllipticCurvePublicKey):
            raise KillSwitchStop("kill switch public key must be an EC key")
        try:
            key.verify(sig, DOMAIN + raw, ec.ECDSA(hashes.SHA256()))
        except InvalidSignature:
            raise KillSwitchStop("kill switch signature does not verify") from None
        try:
            state = json.loads(raw)
        except ValueError:
            state = None
        if not isinstance(state, dict):
            raise KillSwitchStop("kill switch payload is not a JSON object")
        ts, stopped = state.get("ts"), state.get("stopped")
        if not isinstance(ts, (int, float)) or abs((now or time.time()) - ts) > self.max_age_s:
            raise KillSwitchStop(f"kill switch state is stale or undated (ts={ts})")
        if stopped is not False:
            raise KillSwitchStop("the owner has stopped Artemis")
        return state

    def check(self) -> dict:
        """Raises KillSwitchStop unless a fresh signed state says running."""
        try:
            doc = self.fetch()
        except KillSwitchStop:
            raise
        except Exception as e:  # unreachable or unreadable counts as stop
            raise KillSwitchStop(f"kill switch state unreachable: {type(e).__name__}") from None
        return self.verify(doc)
