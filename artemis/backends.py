"""Where brain replies come from. Every backend runs Artemis's own weights; there is no outside-model backend."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Iterator, Protocol


class ModelNotReady(RuntimeError):
    """Raised while Artemis's own model has not yet passed evaluation."""


class ModelUnavailable(RuntimeError):
    """A configured inference service failed; this does not mean it is training."""


class Backend(Protocol):
    def generate(self, brain: str, system: str, messages: list[dict], max_tokens: int = 512) -> str: ...

    def stream(self, brain: str, system: str, messages: list[dict], max_tokens: int = 512) -> Iterator[str]: ...


class NotReadyBackend:
    """Used until the first Artemis model passes its evaluation gate."""

    def readiness(self):
        return {"configured": False, "serving": False, "status": "training", "quality": "unverified"}

    def generate(self, brain, system, messages, max_tokens=512):
        raise ModelNotReady("Artemis is still training; chat opens when the first model passes evaluation.")

    def stream(self, brain, system, messages, max_tokens=512):
        raise ModelNotReady("Artemis is still training; chat opens when the first model passes evaluation.")


class LocalBackend:
    """Artemis's own checkpoint loaded in this process (artemis/infer.py): the Prime Core and its experts run here."""

    def __init__(self, checkpoint_path: str, tokenizer_path: str):
        from .infer import LocalModel  # torch is only needed when a checkpoint is configured
        self.model = LocalModel(checkpoint_path, tokenizer_path)

    def generate(self, brain, system, messages, max_tokens=512):
        return self.model.generate(messages, brain, system, max_tokens)

    def stream(self, brain, system, messages, max_tokens=512):
        yield from self.model.stream(messages, brain, system, max_tokens)

    def readiness(self):
        return {"configured": True, "serving": True, "status": "loaded", "quality": "unverified"}


class ArtemisServerBackend:
    """Artemis's own inference server (vLLM serving the foundation plus one adapter per brain).

    vLLM exposes an OpenAI-compatible HTTP API; `model` selects the brain's adapter.
    """

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 60.0):
        self.base_url, self.api_key, self.timeout = base_url.rstrip("/"), api_key, timeout
        self._health_lock = threading.Lock()
        self._health = None
        self._health_at = 0.0

    def readiness(self):
        """Probe an actual generation, cached for 15 seconds. Availability is not a quality evaluation."""
        with self._health_lock:
            if self._health is not None and time.monotonic() - self._health_at < 15:
                return dict(self._health)
            probe = ArtemisServerBackend(self.base_url, self.api_key, min(self.timeout, 5.0))
            try:
                reply = probe.generate("artemis", "Reply briefly.", [{"role": "user", "content": "Hello"}], 8)
                available = isinstance(reply, str) and bool(reply.strip())
            except ModelUnavailable:
                available = False
            self._health = {"configured": True, "serving": available,
                            "status": "available" if available else "unavailable", "quality": "unverified"}
            self._health_at = time.monotonic()
            return dict(self._health)

    def generate(self, brain, system, messages, max_tokens=512):
        body = json.dumps({"model": brain, "max_tokens": max_tokens,
                           "messages": [{"role": "system", "content": system}, *messages]}).encode()
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                content = json.load(r)["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("invalid inference content")
                return content
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as e:
            raise ModelUnavailable("Artemis inference service is unavailable; please retry later.") from e

    def stream(self, brain, system, messages, max_tokens=512):
        body = json.dumps({"model": brain, "max_tokens": max_tokens, "stream": True,
                           "messages": [{"role": "system", "content": system}, *messages]}).encode()
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                finished = False
                for raw in r:
                    line = raw.decode().strip()
                    if line == "data: [DONE]":
                        finished = True
                        break
                    if not line.startswith("data:"):
                        continue
                    data = json.loads(line[5:])
                    if "error" in data:
                        raise ModelUnavailable("Artemis inference service returned an error; please retry later.")
                    choices = data.get("choices", [])
                    if not choices:  # optional usage-only chunk
                        continue
                    piece = choices[0].get("delta", {}).get("content")
                    if piece:
                        if not isinstance(piece, str):
                            raise ValueError("invalid inference content")
                        yield piece
                if not finished:
                    raise ModelUnavailable("Artemis inference stream ended early; please retry later.")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as e:
            raise ModelUnavailable("Artemis inference service is unavailable; please retry later.") from e
