"""Artemis AI's business engine: plans, entitlements, usage limits, API keys and metering.

The same structure as ChatGPT, Claude and Grok: subscription plans decide which ladder tier and brains a
user gets and how many messages per day; developers use API keys billed per million tokens.
State is kept in SQLite so the API server can enforce limits across restarts.
"""
from __future__ import annotations

import hashlib
import secrets
import re
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from .brains import SPECIALIST_IDS
from .store import Store

BUSINESS_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "business.yaml"
UNLIMITED = float("inf")
ALL_TOOLS = ("web_search", "web_fetch", "run_python")
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[a-z]{2,}")


class PaymentRequired(Exception):
    """The plan does not include this (HTTP 402)."""


class RateLimited(Exception):
    """A usage limit was reached (HTTP 429)."""


class Unauthorized(Exception):
    """Unknown or revoked API key (HTTP 401)."""


@dataclass(frozen=True)
class Plan:
    id: str
    tier: str
    brains: frozenset[str]
    messages_per_day: float
    voice_minutes_per_month: float
    api_access: bool
    raw: dict
    tools: frozenset[str] = frozenset()
    tool_calls_per_day: float = 0


def load_business(path: Path = BUSINESS_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _limit(v) -> float:
    return UNLIMITED if v == "unlimited" else float(v)


def load_plans(cfg: dict) -> dict[str, Plan]:
    plans = {}
    for pid, p in cfg["plans"].items():
        brains = frozenset(SPECIALIST_IDS) if p["brains"] == "all" else frozenset(p["brains"])
        tools = frozenset(ALL_TOOLS) if p.get("tools") == "all" else frozenset(p.get("tools") or ())
        plans[pid] = Plan(pid, p["tier"], brains, _limit(p["messages_per_day"]), _limit(p["voice_minutes_per_month"]),
                          bool(p.get("api_access")), p, tools, _limit(p.get("tool_calls_per_day", 0)))
    return plans


class Business:
    def __init__(self, db_path: str | Path, config_path: Path = BUSINESS_CONFIG):
        self.cfg = load_business(config_path)
        self.plans = load_plans(self.cfg)
        self.db = Store(str(db_path))

    # ---- accounts and keys ---------------------------------------------------------------
    def set_plan(self, account: str, plan: str) -> None:
        if plan not in self.plans:
            raise ValueError(f"unknown plan {plan!r}")
        self.db.execute("INSERT INTO accounts VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET plan=excluded.plan",
                        (account, plan, time.time()))

    def plan_of(self, account: str) -> Plan:
        row = self.db.one("SELECT plan FROM accounts WHERE id=?", (account,))
        return self.plans[row[0] if row else "free"]

    def create_api_key(self, account: str) -> str:
        if not self.plan_of(account).api_access and not self._has_api_credit(account):
            raise PaymentRequired("API access needs an Enterprise plan or developer credit")
        key = "art-" + secrets.token_urlsafe(32)
        self.db.execute("INSERT INTO api_keys VALUES (?, ?, ?, 0)", (self._hash(key), account, time.time()))
        return key  # shown once; only the hash is stored

    def revoke_api_key(self, key: str) -> None:
        self.db.execute("UPDATE api_keys SET revoked=1 WHERE hash=?", (self._hash(key),))

    def account_for_key(self, key: str) -> str:
        row = self.db.one("SELECT account FROM api_keys WHERE hash=? AND revoked=0", (self._hash(key),))
        if not row:
            raise Unauthorized("invalid or revoked API key")
        return row[0]

    def grant_api_credit(self, account: str, usd: float) -> None:
        self._record(account, "api_credit", usd, None, 0.0)

    # ---- entitlement checks --------------------------------------------------------------
    def authorize_chat(self, account: str, tier: str | None, brain: str | None) -> str:
        """Return the tier this request runs at, or raise if the plan does not allow it."""
        plan = self.plan_of(account)
        from .orchestrator import TIERS
        order = list(TIERS)
        use = tier or plan.tier
        if use not in TIERS:
            raise ValueError(f"unknown tier {use!r}")
        if order.index(use) > order.index(plan.tier):
            raise PaymentRequired(f"{use} needs a higher plan than {plan.id}")
        if brain and brain not in plan.brains:
            raise PaymentRequired(f"the {brain} brain is not included in the {plan.id} plan")
        if self.used(account, "message") >= plan.messages_per_day:
            raise RateLimited(f"daily message limit for the {plan.id} plan reached; it resets at 00:00 UTC")
        return use

    def authorize_api(self, account: str, model: str) -> str:
        models = self.cfg["api_models"]
        if model not in models:
            raise ValueError(f"unknown model {model!r}; choose one of {list(models)}")
        recent = self.db.one("SELECT COUNT(*) FROM usage WHERE account=? AND kind='api_call' AND ts>?",
                             (account, time.time() - 60))[0]
        if recent >= self.cfg["api_rate_limit_per_minute"]:
            raise RateLimited("API rate limit reached; retry in a minute")
        if not self.plan_of(account).api_access and self.api_balance(account) <= 0:
            raise PaymentRequired("API credit is used up; add credit to continue")
        return models[model]["tier"]

    def allowed_tools(self, account: str) -> list[str]:
        """Tools this account may still use today (plan permission and daily cap)."""
        plan = self.plan_of(account)
        if self.used(account, "tool_call") >= plan.tool_calls_per_day:
            return []
        return [t for t in ALL_TOOLS if t in plan.tools]

    def authorize_tool(self, account: str, tool: str) -> None:
        plan = self.plan_of(account)
        if tool not in plan.tools:
            raise PaymentRequired(f"the {tool} tool is not included in the {plan.id} plan")
        if self.used(account, "tool_call") >= plan.tool_calls_per_day:
            raise RateLimited(f"daily tool limit for the {plan.id} plan reached; it resets at 00:00 UTC")

    def record_tool(self, account: str, tool: str) -> None:
        self._record(account, "tool_call", 1, tool, 0.0)

    # ---- metering ------------------------------------------------------------------------
    def record_message(self, account: str) -> None:
        self._record(account, "message", 1, None, 0.0)

    def record_api_call(self, account: str, model: str, input_tokens: int, output_tokens: int) -> float:
        m = self.cfg["api_models"][model]
        cost = input_tokens / 1e6 * m["input_per_mtok"] + output_tokens / 1e6 * m["output_per_mtok"]
        self._record(account, "api_call", input_tokens + output_tokens, model, cost)
        return cost

    def used(self, account: str, kind: str) -> float:
        return self.db.one("SELECT COALESCE(SUM(amount),0) FROM usage WHERE account=? AND day=? AND kind=?",
                           (account, self._today(), kind))[0]

    def api_balance(self, account: str) -> float:
        credit = self.db.one("SELECT COALESCE(SUM(amount),0) FROM usage WHERE account=? AND kind='api_credit'", (account,))[0]
        spent = self.db.one("SELECT COALESCE(SUM(cost_usd),0) FROM usage WHERE account=? AND kind='api_call'", (account,))[0]
        return credit - spent

    # ---- waitlist ------------------------------------------------------------------------
    def join_waitlist(self, email: str, requester: str, source: str = "website") -> bool:
        """Store an email once. Returns False if it was already on the list."""
        email = email.strip().lower()
        if len(email) > 254 or not EMAIL_RE.fullmatch(email):
            raise ValueError("enter a valid email address")
        if self.used(requester, "waitlist_signup") >= 5:
            raise RateLimited("too many sign-ups from this connection today")
        self._record(requester, "waitlist_signup", 1, None, 0.0)
        if self.db.one("SELECT 1 FROM waitlist WHERE email=?", (email,)):
            return False
        self.db.execute("INSERT INTO waitlist VALUES (?, ?, ?) ON CONFLICT(email) DO NOTHING", (email, time.time(), source))
        return True

    def waitlist_count(self) -> int:
        return int(self.db.one("SELECT COUNT(*) FROM waitlist")[0])

    def public_plans(self) -> dict:
        return {"plans": self.cfg["plans"], "api_models": self.cfg["api_models"]}

    # ---- internals -----------------------------------------------------------------------
    def _has_api_credit(self, account: str) -> bool:
        return self.api_balance(account) > 0

    def _record(self, account, kind, amount, model, cost):
        self.db.execute("INSERT INTO usage VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (account, self._today(), kind, amount, model, cost, time.time()))

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    @staticmethod
    def _hash(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()
