"""Command line for the data steps.

python -m artemis.pipeline tokenizer --files corpus/*.txt --vocab 32000 --out runs/tok.json
python -m artemis.pipeline prepare --sources sources.yaml --tokenizer runs/tok.json --out runs/data
sources.yaml: list of {path, license, origin}
"""
from __future__ import annotations

import argparse
import json

import yaml

from .data import prepare
from .tokenizer import train_tokenizer


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tokenizer")
    t.add_argument("--files", nargs="+", required=True)
    t.add_argument("--vocab", type=int, default=32000)
    t.add_argument("--out", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--sources", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "tokenizer":
        tok = train_tokenizer(a.files, a.vocab, a.out)
        print(json.dumps({"vocab_size": tok.get_vocab_size(), "out": a.out}))
    else:
        m = prepare(yaml.safe_load(open(a.sources)), a.tokenizer, a.out)
        print(json.dumps({k: m[k] for k in ("train_tokens", "val_tokens", "dtype")}))


if __name__ == "__main__":
    main()
