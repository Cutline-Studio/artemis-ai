"""Download FineWeb-Edu sample shards and write train/val token shards for a matched-token ablation.

python infra/vast/prep_fineweb.py --out /w/data --shards 2 --tok-sample-mb 400
FineWeb-Edu is ODC-By 1.0. Both ablation arms read the same shards, tokenizer and seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from artemis.tokenizer import load_tokenizer, train_tokenizer  # noqa: E402

BASE = os.environ.get("FINEWEB_BASE") or "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/{:03d}_00000.parquet"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--tok-sample-mb", type=int, default=400)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--val-fraction", type=float, default=0.005)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(a.shards):
        dst = out / f"raw_{i:03d}.parquet"
        if not dst.exists():
            urllib.request.urlretrieve(BASE.format(i), dst)
        files.append(dst)
        print(f"downloaded {dst.name} {dst.stat().st_size / 1e9:.2f} GB", flush=True)

    sample = out / "tok_sample.txt"
    with open(sample, "w") as f:
        written = 0
        for batch in pq.ParquetFile(files[0]).iter_batches(columns=["text"], batch_size=2000):
            for t in batch.column(0).to_pylist():
                f.write(t + "\n")
                written += len(t)
            if written > a.tok_sample_mb * 1_000_000:
                break
    tok_path = out / "tokenizer.json"
    train_tokenizer([str(sample)], a.vocab, tok_path)
    tok = load_tokenizer(tok_path)
    eos = tok.token_to_id("<|eos|>")
    counts = {"train": 0, "val": 0}
    sinks = {s: open(out / f"{s}.bin", "wb") for s in counts}
    docs = 0
    for p in files:
        for batch in pq.ParquetFile(p).iter_batches(columns=["text"], batch_size=4096):
            texts = batch.column(0).to_pylist()
            for t, enc in zip(texts, tok.encode_batch(texts)):
                split = "val" if int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < a.val_fraction else "train"
                ids = np.array(enc.ids + [eos], dtype=np.uint16)
                sinks[split].write(ids.tobytes())
                counts[split] += len(ids)
                docs += 1
        print(f"tokenized {p.name}: train {counts['train']:,} val {counts['val']:,}", flush=True)
    for f in sinks.values():
        f.close()
    manifest = {"dtype": "uint16", "train_tokens": counts["train"], "val_tokens": counts["val"], "docs": docs,
                "tokenizer": str(tok_path), "tokenizer_sha256": hashlib.sha256(tok_path.read_bytes()).hexdigest(),
                "sources": [{"origin": BASE.format(i), "license": "ODC-By-1.0", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                            for i, p in enumerate(files)], "domain_labels": False}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}), flush=True)
    for p in files:
        os.remove(p)


if __name__ == "__main__":
    main()
