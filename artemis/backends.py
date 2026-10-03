"""Where brain replies come from. Every backend runs Artemis's own weights; there is no outside-model backend."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterator, Protocol


class ModelNotReady(RuntimeError):
    """Raised while Artemis's own model has not yet passed evaluation."""


class Backend(Protocol):
    def generate(self, brain: str, system: str, messages: list[dict], max_tokens: int = 512) -> str: ...

    def stream(self, brain: str, system: str, messages: list[dict], max_tokens: int = 512) -> Iterator[str]: ...


class NotReadyBackend:
    """Used until the first Artemis model passes its evaluation gate."""

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


class ArtemisServerBackend:
    """Artemis's own inference server (vLLM serving the foundation plus one adapter per brain).

    vLLM exposes an OpenAI-compatible HTTP API; `model` selects the brain's adapter.
    """

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 60.0):
        self.base_url, self.api_key, self.timeout = base_url.rstrip("/"), api_key, timeout

    def generate(self, brain, system, messages, max_tokens=512):
        body = json.dumps({"model": brain, "max_tokens": max_tokens,
                           "messages": [{"role": "system", "content": system}, *messages]}).encode()
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError) as e:
            raise ModelNotReady(f"Artemis inference server unavailable: {e}") from e

    def stream(self, brain, system, messages, max_tokens=512):
        body = json.dumps({"model": brain, "max_tokens": max_tokens, "stream": True,
                           "messages": [{"role": "system", "content": system}, *messages]}).encode()
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    piece = json.loads(line[5:])["choices"][0].get("delta", {}).get("content")
                    if piece:
                        yield piece
        except (urllib.error.URLError, TimeoutError) as e:
            raise ModelNotReady(f"Artemis inference server unavailable: {e}") from e
