"""Turn the Artemis foundation into a specialist brain.

Each specialist is a LoRA adapter (small extra weights, ~1% of the model) trained on that brain's
conversations, on top of the shared foundation. One GPU server can then serve all ten brains at once.
Loss is computed only on the assistant's replies, not on the prompt.

Adapters attach to the Prime Core (attention and its dense feed-forward) only. The ten neural experts and their
routers are never wrapped, so an adapter can't blur what each expert learned; a full fine-tune trains every weight,
experts and routers included.

python -m artemis.specialize --foundation runs/1b/latest.pt --brain saturn --data data/saturn.jsonl --tokenizer runs/tok.json --out runs/brains
data/<brain>.jsonl lines: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn

from . import checkpoint
from .brains import load_brains
from .model import ArtemisLM
from .tokenizer import clean, load_tokenizer

LORA_TARGETS = ("q", "k", "v", "o", "gate", "up", "down")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.randn(rank, base.in_features) * (1 / math.sqrt(base.in_features)))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))  # starts as a no-op
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scale


def add_lora(model: ArtemisLM, rank: int = 16, alpha: float = 32.0) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for mname, module in list(model.named_modules()):
        if ".moe" in mname:  # experts and routers keep their own weights
            continue
        for name in LORA_TARGETS:
            child = getattr(module, name, None)
            if isinstance(child, nn.Linear):
                lora = LoRALinear(child, rank, alpha)
                setattr(module, name, lora)
                params += [lora.a, lora.b]
    return params


def lora_state(model: nn.Module) -> dict:
    return {k: v for k, v in model.state_dict().items() if k.endswith((".a", ".b"))}


def encode_example(tok, system: str, messages: list[dict], brain: str, max_len: int):
    """Token ids plus labels that are -100 everywhere except assistant reply tokens."""
    ids, labels = [], []

    def add(text: str, train: bool):
        t = tok.encode(text).ids
        ids.extend(t)
        labels.extend(t if train else [-100] * len(t))

    add(f"<|bos|><|brain:{brain}|><|system|>{system}<|end|>", False)
    for m in messages:
        add(f"<|{m['role']}|>", False)
        add(clean(m["content"]) + "<|end|>", m["role"] == "assistant")
    # keep the end of long examples, where the reply is; the start of the prompt is what gets cut
    return ids[-max_len:], labels[-max_len:]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--foundation", required=True)
    ap.add_argument("--brain", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    a = ap.parse_args(argv)

    brains = load_brains()
    if a.brain not in brains:
        raise SystemExit(f"unknown brain: {a.brain}")
    brain = brains[a.brain]
    full = brain.model.get("specialization") == "full-finetune"  # Artemis (the orchestrator) trains every weight
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = checkpoint.load_model(a.foundation)
    cfg = model.cfg
    params = list(model.parameters()) if full else add_lora(model, a.rank, alpha=2.0 * a.rank)
    model.to(device).train()
    tok = load_tokenizer(a.tokenizer)
    from .tools import TOOL_SPECS, Toolbox
    tool_note = Toolbox().instructions(list(TOOL_SPECS))  # same tool instructions the server adds at inference

    def system_for(messages):
        uses_tools = any("<tool_call>" in m["content"] for m in messages)
        return brain.system_prompt() + (tool_note if uses_tools else "")
    examples = [encode_example(tok, system_for(m), m, a.brain, cfg.max_seq_len + 1)
                for m in (json.loads(l)["messages"] for l in Path(a.data).read_text().splitlines() if l.strip())]
    examples = [e for e in examples if any(t != -100 for t in e[1][1:])]
    if not examples:
        raise SystemExit("no training examples with an assistant reply inside the context window")
    opt = torch.optim.AdamW(params, lr=a.lr)
    losses = []
    for step in range(a.steps):
        ids, labels = random.choice(examples)
        x = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor([labels[1:]], device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if full:
        checkpoint.save(out / f"{a.brain}.full.pt", model, None, a.steps, vars(a),
                        extra={"brain": a.brain, "foundation": str(a.foundation)})
    else:
        torch.save({"format_version": checkpoint.FORMAT_VERSION, "brain": a.brain, "rank": a.rank, "alpha": 2.0 * a.rank, "lora": lora_state(model), "foundation": str(a.foundation),
                    "config": cfg.to_dict()}, out / f"{a.brain}.adapter.pt")
    result = {"brain": a.brain, "first_loss": round(losses[0], 4), "last_loss": round(sum(losses[-10:]) / min(10, len(losses)), 4),
              "weights": "full" if full else "adapter", "trainable_params": sum(p.numel() for p in params), "total_params": cfg.param_count()}
    (out / f"{a.brain}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
