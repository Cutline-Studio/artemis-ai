"""Run Artemis's own checkpoint directly (CPU or one GPU): load, generate, stream, and show expert routing.

python -m artemis.infer --checkpoint runs/proto/latest.pt --tokenizer runs/tok-proto.json --prompt "Deploy checkout" --trace
The API server uses this through backends.LocalBackend when ARTEMIS_CHECKPOINT is set.
"""
from __future__ import annotations

import argparse
import json
import threading
from collections import Counter

import torch

from . import checkpoint
from .tokenizer import format_chat, load_tokenizer


class LocalModel:
    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.ck = checkpoint.load_model(checkpoint_path, self.device)
        self.model.eval()
        self.tok = load_tokenizer(tokenizer_path)
        if self.tok.get_vocab_size() > self.model.cfg.vocab_size:
            raise checkpoint.CheckpointError(f"tokenizer has {self.tok.get_vocab_size()} tokens; model has {self.model.cfg.vocab_size}")
        self.stop = [i for i in (self.tok.token_to_id("<|end|>"), self.tok.token_to_id("<|eos|>")) if i is not None]
        self.lock = threading.Lock()  # one generation at a time per loaded model
        self.checkpoint_path = checkpoint_path

    def info(self) -> dict:
        cfg = self.model.cfg
        return {"checkpoint": self.checkpoint_path, "step": self.ck["step"], "format_version": self.ck["format_version"],
                "arch": self.ck["arch"], "params": cfg.param_count(), "active_params": cfg.active_param_count(),
                "expert_layers": list(cfg.moe_layers), "experts": list(cfg.expert_names)}

    def prompt_ids(self, messages: list[dict], brain: str = "artemis", system: str = "") -> list[int]:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        return self.tok.encode(format_chat(msgs, brain) + "<|assistant|>").ids

    def stream(self, messages, brain="artemis", system="", max_tokens=256, temperature=0.8, top_k=50, trace=None):
        """Yields text pieces as tokens are produced. If trace is a Counter, it counts (layer, expert) selections."""
        ids = self.prompt_ids(messages, brain, system)[-(self.model.cfg.max_seq_len - 1):]
        with self.lock:
            for m in self.model.mixtures:
                m.keep_trace = trace is not None
            try:
                idx = torch.tensor([ids], device=self.device)
                new, shown = [], ""
                for t in self.model.stream(idx, max_tokens, temperature, top_k, self.stop):
                    if trace is not None:
                        names = self.model.cfg.expert_names
                        for li, m in enumerate(self.model.mixtures):
                            for e in m.trace[0, -1].tolist():
                                trace[(self.model.cfg.moe_layers[li], names[e])] += 1
                    if t in self.stop:
                        break
                    new.append(t)
                    text = self.tok.decode(new)
                    if len(text) > len(shown) and not text.endswith("�"):
                        yield text[len(shown):]
                        shown = text
            finally:
                for m in self.model.mixtures:
                    m.keep_trace, m.trace = False, None

    def generate(self, messages, brain="artemis", system="", max_tokens=256, **kw) -> str:
        return "".join(self.stream(messages, brain, system, max_tokens, **kw))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw", action="store_true", help="continue the prompt as plain text instead of chat format")
    ap.add_argument("--trace", action="store_true", help="report which experts the router chose")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    lm = LocalModel(a.checkpoint, a.tokenizer)
    trace = Counter() if a.trace else None
    if a.raw:
        for m in lm.model.mixtures:
            m.keep_trace = a.trace
        idx = torch.tensor([lm.tok.encode(a.prompt).ids])
        out = []
        for t in lm.model.stream(idx, a.max_tokens, a.temperature, 50, lm.stop):
            out.append(t)
            if trace is not None:
                for li, m in enumerate(lm.model.mixtures):
                    for e in m.trace[0, -1].tolist():
                        trace[(lm.model.cfg.moe_layers[li], lm.model.cfg.expert_names[e])] += 1
        text = lm.tok.decode(out)
    else:
        text = lm.generate([{"role": "user", "content": a.prompt}], max_tokens=a.max_tokens, temperature=a.temperature, trace=trace)
    result = {"model": lm.info(), "prompt": a.prompt, "output": text}
    if trace is not None:
        result["routing"] = {f"layer {l}": dict(Counter({e: c for (ll, e), c in trace.items() if ll == l}).most_common())
                             for l in lm.model.cfg.moe_layers}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
