"""Data preparation: tokenize licensed sources into shards with a manifest for lineage.

Each source may name an expert domain ("domain": "security"), and JSONL lines may carry their own "domain".
Domain-labeled tokens get a label shard (train.dom / val.dom, one byte per token: expert index, 255 = unlabeled)
used for routing supervision and per-expert evaluation. Documents are split between train and validation by a
hash of their text, so every domain appears in validation; a very long document is split at its tail instead.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .model import EXPERT_NAMES
from .tokenizer import load_tokenizer

UNLABELED = 255


def _iter_docs(path: Path, default_domain):
    if path.suffix == ".jsonl":
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                yield rec["text"], rec.get("domain", default_domain)
    else:
        yield path.read_text(), default_domain


def prepare(sources: list[dict], tokenizer_path: str, out_dir: str, val_fraction: float = 0.01) -> dict:
    """sources: [{"path", "license", "origin", optional "domain"}]. Writes train/val .bin (+ .dom) and manifest.json."""
    tok = load_tokenizer(tokenizer_path)
    eos = tok.token_to_id("<|eos|>")
    split = {"train": ([], []), "val": ([], [])}
    manifest = {"tokenizer": str(tokenizer_path), "tokenizer_sha256": hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
                "experts": list(EXPERT_NAMES), "sources": []}
    seen = set()
    labeled = False
    for src in sources:
        p = Path(src["path"])
        if not src.get("license"):
            raise ValueError(f"{p}: every source needs a recorded license")
        counts = {"train": 0, "val": 0}
        for text, domain in _iter_docs(p, src.get("domain")):
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:  # exact-duplicate removal
                continue
            seen.add(digest)
            if domain is not None and domain not in EXPERT_NAMES:
                raise ValueError(f"{p}: unknown domain {domain!r}")
            label = EXPERT_NAMES.index(domain) if domain else UNLABELED
            labeled = labeled or label != UNLABELED
            ids = tok.encode(text).ids + [eos]
            if len(ids) > 4096:  # long single document: its tail goes to validation
                cut = len(ids) - max(1, int(len(ids) * val_fraction))
                parts = [("train", ids[:cut]), ("val", ids[cut:])]
            else:
                parts = [("val" if int(digest[:8], 16) / 0xFFFFFFFF < val_fraction else "train", ids)]
            for name, chunk in parts:
                split[name][0].extend(chunk)
                split[name][1].extend([label] * len(chunk))
                counts[name] += len(chunk)
        manifest["sources"].append({**src, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                    "tokens": counts["train"] + counts["val"], "train_tokens": counts["train"], "val_tokens": counts["val"]})
    dtype = np.uint32 if tok.get_vocab_size() > 65535 else np.uint16
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, (ids, labels) in split.items():
        np.array(ids, dtype=dtype).tofile(out / f"{name}.bin")
        if labeled:
            np.array(labels, dtype=np.uint8).tofile(out / f"{name}.dom")
    manifest.update(dtype=np.dtype(dtype).name, train_tokens=len(split["train"][0]), val_tokens=len(split["val"][0]),
                    domain_labels=bool(labeled))
    if labeled:
        manifest["domain_tokens"] = {d: {s: int(np.sum(np.array(split[s][1]) == i)) for s in split} for i, d in enumerate(EXPERT_NAMES)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class TokenStream:
    """Random fixed-length windows from a token shard, with per-token domain labels when the shard has them."""

    def __init__(self, path: Path, dtype: str, seq_len: int, seed: int = 0):
        self.data = np.memmap(path, dtype=dtype, mode="r")
        dom = Path(path).with_suffix(".dom")
        self.labels = np.memmap(dom, dtype=np.uint8, mode="r") if dom.exists() else None
        self.seq_len = seq_len
        if len(self.data) <= seq_len + 1:
            raise ValueError(f"{path} has {len(self.data)} tokens; need more than {seq_len + 1}")
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size: int, with_domains: bool = False):
        import torch
        starts = self.rng.integers(0, len(self.data) - self.seq_len - 1, batch_size)
        x = np.stack([self.data[s:s + self.seq_len] for s in starts]).astype(np.int64)
        y = np.stack([self.data[s + 1:s + 1 + self.seq_len] for s in starts]).astype(np.int64)
        if not with_domains:
            return torch.from_numpy(x), torch.from_numpy(y)
        if self.labels is None:
            d = np.full_like(x, -1)
        else:
            d = np.stack([self.labels[s:s + self.seq_len] for s in starts]).astype(np.int64)
            d[d == UNLABELED] = -1
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(d)

    def windows(self, batch_size: int, max_batches: int):
        """Deterministic, non-overlapping windows for evaluation."""
        import torch
        n = min(max_batches * batch_size, (len(self.data) - 1) // self.seq_len)
        # evenly spaced over the whole shard, so every domain in it is represented
        all_starts = np.linspace(0, len(self.data) - self.seq_len - 1, n).astype(np.int64)
        for i in range(0, n, batch_size):
            starts = all_starts[i:i + batch_size]
            x = np.stack([self.data[s:s + self.seq_len] for s in starts]).astype(np.int64)
            y = np.stack([self.data[s + 1:s + 1 + self.seq_len] for s in starts]).astype(np.int64)
            if self.labels is not None:
                d = np.stack([self.labels[s:s + self.seq_len] for s in starts]).astype(np.int64)
                d[d == UNLABELED] = -1
            else:
                d = np.full_like(x, -1)
            yield torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(d)
