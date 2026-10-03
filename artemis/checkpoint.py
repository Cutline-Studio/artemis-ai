"""Artemis checkpoint format, loading, and versioned migration.

Format 2 (current): {"format_version": 2, "arch": "artemis-prime-moe", "config", "model", "optim", "step",
                     "args", "rng", "params": {name: shape}, "migration": optional record}
Format 1 (before the neural core): same keys without format_version/arch/params; a dense model only.

Loading is strict: a missing or unexpected tensor is an error that names it. A format-1 checkpoint is never
upgraded silently: run the migration, which writes a new file and leaves the original untouched.

python -m artemis.checkpoint migrate --src runs/1b/latest.pt --dst runs/1b/latest.v2.pt --moe-every 2
python -m artemis.checkpoint inspect runs/1b/latest.v2.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from .model import ArtemisLM, ModelConfig, parameter_groups

FORMAT_VERSION = 2
ARCH = "artemis-prime-moe"


class CheckpointError(RuntimeError):
    pass


def save(path: str | Path, model: ArtemisLM, opt=None, step: int = 0, args: dict | None = None, extra: dict | None = None):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save({"format_version": FORMAT_VERSION, "arch": ARCH, "config": model.cfg.to_dict(), "model": model.state_dict(),
                "optim": opt.state_dict() if opt is not None else None, "step": step, "args": args or {},
                "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()},
                "params": {k: list(v.shape) for k, v in model.state_dict().items()}, **(extra or {})}, tmp)
    tmp.replace(path)  # atomic: a crash never leaves a half-written checkpoint


def read(path: str | Path, map_location="cpu") -> dict:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    version = ck.get("format_version", 1)
    if version == 1:
        raise CheckpointError(f"{path} is a format-1 (dense, pre-neural-core) checkpoint. Migrate it first, keeping the original: "
                              f"python -m artemis.checkpoint migrate --src {path} --dst <new path>")
    if version != FORMAT_VERSION or ck.get("arch") != ARCH:
        raise CheckpointError(f"{path}: unsupported checkpoint format {version} / {ck.get('arch')}")
    return ck


def load_model(path: str | Path, map_location="cpu") -> tuple[ArtemisLM, dict]:
    """Build the model the checkpoint describes and load every tensor strictly."""
    ck = read(path, map_location)
    model = ArtemisLM(ModelConfig(**ck["config"]))
    load_state(model, ck["model"], str(path))
    return model.to(map_location), ck


def load_state(model: ArtemisLM, state: dict, source: str = "checkpoint") -> None:
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise CheckpointError(f"{source} does not match the model: missing {result.missing_keys}, unexpected {result.unexpected_keys}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def migrate_v1(src: str | Path, dst: str | Path, moe_layers=(), moe_every: int = 0, expert_ffn_mult: float = 1.0,
               seed: int = 0) -> dict:
    """Format 1 (dense) -> format 2 (Prime Core + experts). Writes dst; never modifies src.

    Every format-1 tensor is copied unchanged (the Prime Core keeps the same names). New tensors:
      router weights: normal(0, 0.02)
      expert gate/up: normal(0, 0.02)
      expert down:    zeros, so the migrated model's outputs equal the original's until training updates them
    AdamW moments are carried over for every existing parameter; new parameters start with empty moments.
    """
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise CheckpointError("migration must write a new file; the original checkpoint is preserved")
    if dst.exists():
        raise CheckpointError(f"{dst} already exists; refusing to overwrite")
    old = torch.load(src, map_location="cpu", weights_only=False)
    if old.get("format_version", 1) != 1:
        raise CheckpointError(f"{src} is already format {old.get('format_version')}")
    old_cfg = ModelConfig(**old["config"])
    cfg = ModelConfig(**{**old_cfg.to_dict(), "moe_layers": tuple(moe_layers), "moe_every": moe_every,
                         "expert_ffn_mult": expert_ffn_mult})
    if not cfg.moe_layers:
        raise CheckpointError("choose expert layers with --moe-layers or --moe-every")
    torch.manual_seed(seed)
    dense, model = ArtemisLM(old_cfg), ArtemisLM(cfg)
    load_state(dense, old["model"], str(src))  # proves the source is a complete format-1 model
    new_state = model.state_dict()
    copied, created = [], {}
    for k in new_state:
        if k in old["model"]:
            if old["model"][k].shape != new_state[k].shape:
                raise CheckpointError(f"{k}: shape {tuple(old['model'][k].shape)} cannot become {tuple(new_state[k].shape)}")
            new_state[k] = old["model"][k].clone()
            copied.append(k)
        elif ".moe.router." in k:
            new_state[k] = torch.randn_like(new_state[k]) * 0.02
            created[k] = "normal(0, 0.02)"
        elif k.endswith("down.weight"):
            new_state[k] = torch.zeros_like(new_state[k])
            created[k] = "zeros (expert starts as a no-op)"
        else:
            new_state[k] = torch.randn_like(new_state[k]) * 0.02
            created[k] = "normal(0, 0.02)"
    dropped = sorted(set(old["model"]) - set(new_state))
    if dropped:
        raise CheckpointError(f"these format-1 tensors have no place in format 2: {dropped}")
    model.load_state_dict(new_state)

    optim, optim_note = None, "absent in source"
    if old.get("optim"):
        old_names = [n for n, _ in dense.named_parameters()]
        new_names = [n for n, _ in model.named_parameters()]
        index = {n: i for i, n in enumerate(new_names)}
        o = old["optim"]
        state = {index[old_names[i]]: s for i, s in o["state"].items()}
        if len(o["param_groups"]) != 1:
            raise CheckpointError("source optimizer has several parameter groups; migrate it by hand")
        groups = [{**o["param_groups"][0], "params": list(range(len(new_names)))}]
        optim = {"state": state, "param_groups": groups}
        optim_note = f"AdamW moments carried for {len(state)} parameters; {len(new_names) - len(old_names)} new parameters start empty"

    record = {"from_version": 1, "to_version": FORMAT_VERSION, "source": str(src), "source_sha256": _sha256(src),
              "copied_tensors": len(copied), "new_tensors": created, "dropped_tensors": dropped, "optimizer": optim_note,
              "step": old.get("step", 0), "moe_layers": list(cfg.moe_layers)}
    torch.save({"format_version": FORMAT_VERSION, "arch": ARCH, "config": cfg.to_dict(), "model": model.state_dict(),
                "optim": optim, "step": old.get("step", 0), "args": old.get("args", {}), "rng": old.get("rng"),
                "params": {k: list(v.shape) for k, v in model.state_dict().items()}, "migration": record}, dst)
    Path(str(dst) + ".migration.json").write_text(json.dumps(record, indent=2))
    return record


def inspect(path: str | Path) -> dict:
    model, ck = load_model(path)
    groups = parameter_groups(model)
    params = dict(model.named_parameters())
    return {"format_version": ck["format_version"], "arch": ck["arch"], "step": ck["step"], "moe_layers": list(model.cfg.moe_layers),
            "experts": list(model.cfg.expert_names), "param_count": model.cfg.param_count(),
            "active_param_count": model.cfg.active_param_count(),
            "groups": {g: sum(params[n].numel() for n in names) for g, names in groups.items()},
            "migration": ck.get("migration")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("migrate")
    m.add_argument("--src", required=True)
    m.add_argument("--dst", required=True)
    m.add_argument("--moe-layers", type=int, nargs="*", default=[])
    m.add_argument("--moe-every", type=int, default=0)
    m.add_argument("--expert-ffn-mult", type=float, default=1.0)
    i = sub.add_parser("inspect")
    i.add_argument("path")
    a = ap.parse_args(argv)
    if a.cmd == "migrate":
        print(json.dumps(migrate_v1(a.src, a.dst, a.moe_layers, a.moe_every, a.expert_ffn_mult), indent=2)[:4000])
    else:
        print(json.dumps(inspect(a.path), indent=2))


if __name__ == "__main__":
    main()
