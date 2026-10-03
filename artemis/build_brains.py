"""One command: regenerate the self-knowledge data and train every designated weight on top of the foundation.

python -m artemis.build_brains --foundation runs/1b/latest.pt --tokenizer runs/tok.json --out runs/brains \
    [--extra-data data/brains]   # optional per-brain domain conversations: data/brains/<id>.jsonl

Artemis gets a full fine-tune (runs/brains/artemis.full.pt); each planet gets an adapter (<id>.adapter.pt).
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .brains import load_brains
from .knowledge import build
from .specialize import main as specialize


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--foundation", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-data", default=None)
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args(argv)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        build(tmp)
        for b in load_brains():
            data = Path(tmp) / f"{b}.jsonl"
            extra = Path(a.extra_data) / f"{b}.jsonl" if a.extra_data else None
            if extra and extra.exists():
                data.write_text(data.read_text() + extra.read_text())
            results[b] = specialize(["--foundation", a.foundation, "--brain", b, "--data", str(data),
                                     "--tokenizer", a.tokenizer, "--out", a.out, "--steps", str(a.steps)])
    (Path(a.out) / "build.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
