"""Evaluations for the neural core. Naming an expert proves nothing; these measure what each expert does.

validation_loss  next-token loss on held-out data (the router terms are excluded)
per_expert       for each labeled domain: next-token loss, how often the router puts that domain's expert in its
                 top two, and the full domain x expert routing matrix
ablation         remove one expert at a time from routing; how much each domain's loss rises. An expert has
                 specialized when removing it hurts its own domain clearly more than the others.

python -m artemis.evaluate --checkpoint runs/proto/latest.pt --data runs/data --out runs/proto/eval.json \
    [--baseline runs/proto-dense/latest.pt]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def validation_loss(model, stream, batch_size: int, batches: int, device) -> float:
    was = model.training
    model.eval()
    losses = []
    for x, y, _ in stream.windows(batch_size, batches):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            model(x.to(device), y.to(device))
        losses.append(float(model.last_stats["lm_loss"]))
    model.train(was)
    return float(np.mean(losses))


@torch.no_grad()
def _domain_losses(model, stream, batch_size, batches, device, n_domains):
    tot, cnt = np.zeros(n_domains), np.zeros(n_domains)
    for x, y, d in stream.windows(batch_size, batches):
        logits, _ = model(x.to(device))
        tok = F.cross_entropy(logits.float().flatten(0, 1), y.to(device).flatten(), reduction="none").cpu().numpy()
        dd = d.flatten().numpy()
        for i in range(n_domains):
            m = dd == i
            tot[i] += tok[m].sum()
            cnt[i] += m.sum()
    return tot / np.maximum(cnt, 1), cnt


@torch.no_grad()
def per_expert(model, stream, batch_size: int = 8, batches: int = 20, device=torch.device("cpu")) -> dict:
    model.eval()
    names = list(model.cfg.expert_names)
    n = len(names)
    for m in model.mixtures:
        m.keep_trace = True
    routes = np.zeros((len(model.mixtures), n, n))  # layer, domain, expert -> routing slots
    hits, tokens = np.zeros((len(model.mixtures), n)), np.zeros(n)
    tot, cnt = np.zeros(n), np.zeros(n)
    for x, y, d in stream.windows(batch_size, batches):
        logits, _ = model(x.to(device))
        tok = F.cross_entropy(logits.float().flatten(0, 1), y.to(device).flatten(), reduction="none").cpu().numpy()
        dd = d.flatten().numpy()
        for i in range(n):
            mask = dd == i
            tot[i] += tok[mask].sum()
            cnt[i] += mask.sum()
        tokens += np.bincount(dd[dd >= 0], minlength=n)[:n]
        for li, m in enumerate(model.mixtures):
            tr = m.trace.reshape(-1, m.k).cpu().numpy()
            lab = dd >= 0
            np.add.at(routes[li], (np.repeat(dd[lab], m.k), tr[lab].reshape(-1)), 1)
            hits[li] += np.bincount(dd[lab], weights=(tr[lab] == dd[lab, None]).any(1), minlength=n)[:n]
    for m in model.mixtures:
        m.keep_trace, m.trace = False, None
    k = model.cfg.experts_per_token
    out = {}
    for i, name in enumerate(names):
        out[name] = {"tokens": int(tokens[i]), "loss": round(float(tot[i] / max(cnt[i], 1)), 4),
                     "router_hit_rate": [round(float(hits[l, i] / max(tokens[i], 1)), 4) for l in range(len(model.mixtures))],
                     "chance_hit_rate": round(k / n, 4)}
    matrix = (routes / np.maximum(routes.sum(-1, keepdims=True), 1)).round(4).tolist()
    return {"domains": out, "routing_matrix": {"rows": "domain", "cols": "expert", "names": names, "layers": matrix}}


@torch.no_grad()
def ablation(model, stream, batch_size: int = 8, batches: int = 20, device=torch.device("cpu")) -> dict:
    model.eval()
    names = list(model.cfg.expert_names)
    n = len(names)
    base, _ = _domain_losses(model, stream, batch_size, batches, device, n)
    delta = np.zeros((n, n))  # removed expert x domain
    for e in range(n):
        for m in model.mixtures:
            m.disabled = {e}
        loss, _ = _domain_losses(model, stream, batch_size, batches, device, n)
        delta[e] = loss - base
        for m in model.mixtures:
            m.disabled = set()
    own = np.diag(delta)
    others = (delta.sum(1) - own) / (n - 1)
    return {"base_loss": dict(zip(names, base.round(4).tolist())),
            "loss_increase": {"rows": "removed expert", "cols": "domain", "names": names, "matrix": delta.round(4).tolist()},
            "specialization": {nm: {"own_domain_increase": round(float(own[i]), 4), "other_domains_mean": round(float(others[i]), 4),
                                    "specialized": bool(own[i] > 0 and own[i] > 2 * max(others[i], 0) and int(np.argmax(delta[:, i])) == i)}
                               for i, nm in enumerate(names)}}


def run(checkpoint_path, data_dir, batch_size=8, batches=20, baseline=None) -> dict:
    from . import checkpoint
    from .data import TokenStream
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    model, ck = checkpoint.load_model(checkpoint_path, device)
    val = TokenStream(Path(data_dir) / "val.bin", manifest["dtype"], model.cfg.max_seq_len)
    t0 = time.time()
    report = {"checkpoint": str(checkpoint_path), "step": ck["step"], "params": model.cfg.param_count(),
              "active_params": model.cfg.active_param_count(), "val_loss": round(validation_loss(model, val, batch_size, batches, device), 4)}
    if model.mixtures:
        report["per_expert"] = per_expert(model, val, batch_size, batches, device)
        report["ablation"] = ablation(model, val, batch_size, batches, device)
    if baseline:
        bm, bck = checkpoint.load_model(baseline, device)
        bval = TokenStream(Path(data_dir) / "val.bin", manifest["dtype"], bm.cfg.max_seq_len)
        bl, _ = _domain_losses(bm.eval(), bval, batch_size, batches, device, len(model.cfg.expert_names))
        report["baseline"] = {"checkpoint": str(baseline), "step": bck["step"], "params": bm.cfg.param_count(),
                              "active_params": bm.cfg.active_param_count(),
                              "val_loss": round(validation_loss(bm, bval, batch_size, batches, device), 4),
                              "domain_loss": dict(zip(model.cfg.expert_names, bl.round(4).tolist()))}
    report["eval_seconds"] = round(time.time() - t0, 1)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batches", type=int, default=20)
    a = ap.parse_args(argv)
    r = run(a.checkpoint, a.data, a.batch, a.batches, a.baseline)
    Path(a.out).write_text(json.dumps(r, indent=2))
    print(json.dumps({k: v for k, v in r.items() if k not in ("per_expert", "ablation")}, indent=2))


if __name__ == "__main__":
    main()
