"""Artemis's decision engine (Stage 1 of "own brain"): chooses the next training experiment,
enforces the spending cap and the compute ramp, and decides what gets promoted.

No outside AI is involved. Selection uses UCB1 over experiment arms; every result is appended to an
experiment log so later rounds (and later, Artemis's own model) learn from the history.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

ENGINE_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "engine.yaml"


@dataclass(frozen=True)
class Arm:
    size: str
    lr: float
    code_share: float
    warmup_frac: float

    @property
    def key(self) -> str:
        return f"{self.size}|lr={self.lr}|code={self.code_share}|warm={self.warmup_frac}"


class BudgetExceeded(RuntimeError):
    pass


class DecisionEngine:
    def __init__(self, log_path: str | Path, config_path: str | Path = ENGINE_CONFIG):
        self.cfg = yaml.safe_load(Path(config_path).read_text())
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        sp = self.cfg["search_space"]
        self.arms = [Arm(*c) for c in itertools.product(sp["size"], sp["lr"], sp["code_share"], sp["warmup_frac"])]

    # ---- history -------------------------------------------------------------------------
    def history(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        return [json.loads(l) for l in self.log_path.read_text().splitlines() if l.strip()]

    def record(self, arm: Arm, val_loss: float | None, cost_usd: float, month: str, stable: bool, vms: int) -> dict:
        rec = {"arm": arm.key, "size": arm.size, "val_loss": val_loss, "cost_usd": round(cost_usd, 2), "month": month,
               "stable": stable, "vms": vms, "score": None if val_loss is None or not stable else round(-val_loss, 4)}
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    # ---- money ---------------------------------------------------------------------------
    def spent(self, month: str) -> float:
        return sum(r["cost_usd"] for r in self.history() if r["month"] == month)

    def estimate_cost(self, vms: int, hours: float) -> float:
        return vms * hours * self.cfg["vm_hourly_usd"]

    def check_budget(self, month: str, vms: int, hours: float) -> float:
        cost = self.estimate_cost(vms, hours)
        remaining = self.cfg["monthly_cap_usd"] - self.spent(month)
        if cost > remaining:
            raise BudgetExceeded(f"run needs ${cost:,.0f} but only ${remaining:,.0f} remains of the monthly cap; "
                                 "compute stays off until the owner approves or the month resets")
        return cost

    # ---- compute ramp --------------------------------------------------------------------
    def allowed_vms(self) -> int:
        """Advance one ramp stage per stable run whose loss improved on the best earlier run at that stage."""
        stages, hist = self.cfg["ramp_stages"], self.history()
        level = 0
        for i, vms in enumerate(stages[:-1]):
            runs = [r for r in hist if r["vms"] == vms]
            best_before, advanced = math.inf, False
            for r in runs:
                if r["stable"] and r["val_loss"] is not None:
                    if r["val_loss"] < best_before and best_before != math.inf:
                        advanced = True
                    best_before = min(best_before, r["val_loss"])
            if not advanced:
                break
            level = i + 1
        return stages[level]

    # ---- selection -----------------------------------------------------------------------
    def next_arm(self) -> Arm:
        hist = [r for r in self.history() if r["score"] is not None]
        pulls = {a.key: [r["score"] for r in hist if r["arm"] == a.key] for a in self.arms}
        # Only consider sizes the ramp has unlocked: a size is open once the previous size has a stable result.
        sizes = self.cfg["search_space"]["size"]
        done_sizes = {r["size"] for r in hist}
        open_sizes = {sizes[0]} | {sizes[i + 1] for i, s in enumerate(sizes[:-1]) if s in done_sizes}
        candidates = [a for a in self.arms if a.size in open_sizes]
        untried = [a for a in candidates if not pulls[a.key]]
        if untried:
            return untried[0]
        n = sum(len(v) for v in pulls.values())
        alpha = self.cfg["ucb_alpha"]
        return max(candidates, key=lambda a: sum(pulls[a.key]) / len(pulls[a.key]) + alpha * math.sqrt(math.log(n) / len(pulls[a.key])))

    # ---- promotion -----------------------------------------------------------------------
    def should_promote(self, candidate_loss: float, current_loss: float | None) -> bool:
        if current_loss is None:
            return True
        return candidate_loss < current_loss - self.cfg["promotion_margin"]
