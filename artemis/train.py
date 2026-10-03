"""Pretrain the Artemis model (Prime Core + experts) from random initialization.

Single process:   python -m artemis.train --size proto --data runs/data --out runs/proto
8 GPUs (1 VM):    torchrun --nproc_per_node 8 -m artemis.train --size 1b --data ... --out ...
Resumes automatically from the latest checkpoint in --out (same model config required).

Loss = next-token loss + balance_coef * load-balancing loss + router_z_coef * router z-loss
       + route_coef * routing-supervision loss (only on domain-labeled tokens; --route-coef 0 turns it off).
metrics.jsonl records each part, per-layer expert load, router entropy, and router/expert gradient norms.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

from . import checkpoint
from .data import TokenStream
from .evaluate import validation_loss
from .model import ArtemisLM, ModelConfig, parameter_groups

CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"


def model_config(size: str) -> ModelConfig:
    return ModelConfig(**yaml.safe_load(CONFIGS.read_text())[size])


def lr_at(step: int, max_steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def optimizer_audit(model: ArtemisLM, opt) -> dict:
    """Proves the optimizer holds exactly the trainable parameters: every router, expert and Prime Core weight."""
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    params = dict(model.named_parameters())
    missing = [n for n, p in params.items() if p.requires_grad and id(p) not in in_opt]
    extra = len(in_opt - {id(p) for p in params.values()})
    if missing or extra:
        raise RuntimeError(f"optimizer does not match the model: missing {missing}, {extra} unknown tensors")
    groups = parameter_groups(model)
    return {"trainable_tensors": len(in_opt), "trainable_params": sum(p.numel() for p in params.values() if p.requires_grad),
            "groups": {g: {"tensors": len(n), "params": sum(params[k].numel() for k in n)} for g, n in groups.items()}}


def grad_norms(model: ArtemisLM) -> dict:
    params = dict(model.named_parameters())
    out = {}
    for g, names in parameter_groups(model).items():
        if g == "prime_core" or not names:
            continue
        sq = sum(float(params[n].grad.float().pow(2).sum()) for n in names if params[n].grad is not None)
        out[g] = round(math.sqrt(sq), 6)
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="smoke")
    ap.add_argument("--data", required=True, help="directory with train.bin, val.bin, manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200, help="length of the run (sets the learning-rate schedule)")
    ap.add_argument("--stop-at", type=int, default=0, help="stop early at this step (resume later with the same --steps)")
    ap.add_argument("--batch", type=int, default=16, help="sequences per device per micro-step")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--route-coef", type=float, default=0.1, help="routing supervision weight on domain-labeled tokens")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-batches", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)

    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_proc = rank == 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(args.data) / "manifest.json").read_text())
    cfg = model_config(args.size)
    torch.manual_seed(args.seed + rank)
    model = ArtemisLM(cfg).to(device)
    model.route_coef = args.route_coef
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=device.type == "cuda")

    step = 0
    latest = out / "latest.pt"
    if latest.exists():
        ck = checkpoint.read(latest, map_location=device)
        if ck["config"] != cfg.to_dict():
            raise SystemExit(f"{latest} was trained with a different model config; use a new --out or the same --size")
        checkpoint.load_state(model, ck["model"], str(latest))
        if ck.get("optim"):
            opt.load_state_dict(ck["optim"])
        step = ck["step"]
        if ck.get("rng"):
            torch.set_rng_state(ck["rng"]["torch"].cpu())
            np.random.set_state(ck["rng"]["numpy"])
            random.setstate(ck["rng"]["python"])
        if main_proc:
            print(f"resumed from step {step}" + (" (migrated checkpoint)" if ck.get("migration") else ""))
    audit = optimizer_audit(model, opt)

    raw = model
    if ddp:
        # an expert that no token on this GPU picked has no gradient this step; DDP must allow for that
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=bool(cfg.moe_layers))

    train = TokenStream(Path(args.data) / "train.bin", manifest["dtype"], cfg.max_seq_len, seed=args.seed + rank + step)
    val = TokenStream(Path(args.data) / "val.bin", manifest["dtype"], cfg.max_seq_len, seed=0)
    log = open(out / "metrics.jsonl", "a") if main_proc else None
    if main_proc:
        (out / "run.json").write_text(json.dumps({"config": cfg.to_dict(), "params": cfg.param_count(),
                                                  "active_params": cfg.active_param_count(), "optimizer": audit,
                                                  "args": vars(args), "data_manifest": manifest}, indent=2))
    tokens_per_step = args.batch * args.accum * cfg.max_seq_len * (dist.get_world_size() if ddp else 1)
    t0 = time.time()
    end = min(args.stop_at, args.steps) if args.stop_at else args.steps
    while step < end:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        total, parts = 0.0, {}
        train.rng = np.random.default_rng([args.seed, rank, step])  # batches depend only on the step, so resume is exact
        for micro in range(args.accum):
            x, y, d = train.batch(args.batch, with_domains=True)
            sync = not ddp or micro == args.accum - 1
            ctx = model.no_sync() if ddp and not sync else torch.enable_grad()
            with ctx, torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, loss = model(x.to(device), y.to(device), d.to(device) if cfg.moe_layers else None)
                (loss / args.accum).backward()
            total += loss.item() / args.accum
            for k in ("lm_loss", "balance_loss", "z_loss", "route_loss"):
                if k in raw.last_stats:
                    parts[k] = parts.get(k, 0.0) + float(raw.last_stats[k]) / args.accum
        if not math.isfinite(total):
            raise RuntimeError(f"non-finite loss at step {step}; stopping before the run is wasted")
        step += 1
        record_eval = step % args.eval_every == 0 or step == end
        norms = grad_norms(raw) if main_proc and cfg.moe_layers and (record_eval or step == 1) else None
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if main_proc:
            rec = {"step": step, "loss": round(total, 4), **{k: round(v, 4) for k, v in parts.items()},
                   "lr": opt.param_groups[0]["lr"], "grad_norm": round(grad_norm, 3),
                   "tokens": step * tokens_per_step, "elapsed_s": round(time.time() - t0, 1)}
            if cfg.moe_layers and "load" in raw.last_stats:
                rec["expert_load"] = [[round(v, 3) for v in layer] for layer in raw.last_stats["load"].tolist()]
                rec["router_entropy"] = [round(v, 3) for v in raw.last_stats["entropy"].tolist()]
            if norms is not None:
                rec["router_expert_grad_norms"] = norms
            if record_eval:
                rec["val_loss"] = round(validation_loss(raw, val, args.batch, args.eval_batches, device), 4)
                print(json.dumps({k: v for k, v in rec.items() if k not in ("expert_load", "router_expert_grad_norms")}), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % args.ckpt_every == 0 or step == end:
                checkpoint.save(latest, raw, opt, step, vars(args))
    result = {}
    if main_proc:
        result = json.loads(open(out / "metrics.jsonl").read().strip().splitlines()[-1])
        log.close()
    if ddp:
        dist.destroy_process_group()
    return result


if __name__ == "__main__":
    main()
