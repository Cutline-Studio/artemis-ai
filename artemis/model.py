"""Artemis neural core: the Prime Core plus ten trainable expert modules, trained from random initialization.

Prime Core: one shared decoder-only transformer (token embedding, grouped-query attention with rotary
embeddings, RMSNorm, a dense SwiGLU feed-forward in every layer, tied output head). It processes the input,
routes computation, combines expert outputs and generates every token.

Experts: at the layers listed in `moe_layers`, a learned router scores ten SwiGLU expert modules
(Intent, Architecture, Planning, Build, Test, Deploy, Memory, Security, Observability, Cost) for each token,
keeps the top two, renormalizes their weights to sum to 1, and adds the weighted expert outputs back into the
Prime Core's residual stream alongside its own dense feed-forward.

Training terms besides next-token loss: a load-balancing loss (keeps tokens spread across experts), a router
z-loss (keeps router logits small), and an optional routing-supervision loss on tokens whose domain is labeled
(teaches the router which expert a domain belongs to). Names alone do not make an expert a specialist;
`artemis/evaluate.py` measures what each expert actually does.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERT_NAMES = ("intent", "architecture", "planning", "build", "test", "deploy", "memory", "security",
                "observability", "cost")


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_mult: float = 8 / 3
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0
    # Expert layers. Either list them in moe_layers or set moe_every (every Nth layer, starting at layer N-1).
    moe_layers: tuple[int, ...] = ()
    moe_every: int = 0
    n_experts: int = len(EXPERT_NAMES)
    experts_per_token: int = 2
    expert_ffn_mult: float = 1.0
    expert_names: tuple[str, ...] = EXPERT_NAMES
    balance_coef: float = 0.01
    router_z_coef: float = 0.001

    def __post_init__(self):
        if self.moe_every and not self.moe_layers:
            self.moe_layers = tuple(range(self.moe_every - 1, self.n_layers, self.moe_every))
        self.moe_layers = tuple(int(i) for i in self.moe_layers)
        self.expert_names = tuple(self.expert_names)
        if any(not 0 <= i < self.n_layers for i in self.moe_layers):
            raise ValueError(f"moe_layers {self.moe_layers} outside 0..{self.n_layers - 1}")
        if self.moe_layers and (len(self.expert_names) != self.n_experts or not 1 <= self.experts_per_token <= self.n_experts):
            raise ValueError("need one name per expert and 1 <= experts_per_token <= n_experts")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["moe_layers"], d["expert_names"] = list(self.moe_layers), list(self.expert_names)
        return d

    def _ffn_dim(self) -> int:
        return 64 * math.ceil(int(self.ffn_mult * self.d_model) / 64)

    def _expert_dim(self) -> int:
        return 64 * math.ceil(int(self.expert_ffn_mult * self.d_model) / 64)

    def _attn_params(self) -> int:
        head_dim = self.d_model // self.n_heads
        return self.d_model * (self.n_heads * head_dim) * 2 + self.d_model * (self.n_kv_heads * head_dim) * 2

    def param_count(self) -> int:
        """All weights (embedding counted once; the output head shares it)."""
        dense = self.vocab_size * self.d_model + self.n_layers * (self._attn_params() + 3 * self.d_model * self._ffn_dim()
                                                                  + 2 * self.d_model) + self.d_model
        return dense + len(self.moe_layers) * (self.d_model * self.n_experts + self.n_experts * 3 * self.d_model * self._expert_dim())

    def active_param_count(self) -> int:
        """Weights each token passes through (router plus its top-k experts only); compares compute with a dense model."""
        unused = self.n_experts - self.experts_per_token
        return self.param_count() - len(self.moe_layers) * unused * 3 * self.d_model * self._expert_dim()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(head_dim: int, seq_len: int, theta: float, device=None):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = cos[: x.shape[-2]], sin[: x.shape[-2]]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv = cfg.n_heads, cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.n_heads // self.n_kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(b, t, -1))


class FeedForward(nn.Module):
    """SwiGLU feed-forward. The Prime Core's dense FFN and every expert use this block."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ExpertMixture(nn.Module):
    """Learned top-k router over the ten expert modules for one layer."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.k, self.n = cfg.experts_per_token, cfg.n_experts
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(FeedForward(cfg.d_model, cfg._expert_dim()) for _ in range(cfg.n_experts))
        self.disabled: set[int] = set()  # experts removed from routing (ablation evaluations)
        self.trace: torch.Tensor | None = None  # last top-k expert ids, kept only when tracing is switched on
        self.keep_trace = False

    def forward(self, x, domains=None):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        logits = self.router(flat).float()
        if self.disabled:
            logits = logits.masked_fill(torch.isin(torch.arange(self.n, device=x.device),
                                                   torch.tensor(sorted(self.disabled), device=x.device)), float("-inf"))
        probs = logits.softmax(-1)
        top_w, top_i = probs.topk(self.k, dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True)  # normalized routing weights
        out = torch.zeros_like(flat)
        for e in range(self.n):
            rows, slot = (top_i == e).nonzero(as_tuple=True)
            if rows.numel():
                out.index_add_(0, rows, self.experts[e](flat[rows]) * top_w[rows, slot, None].to(flat.dtype))
        # Load balancing (Switch Transformer form): fraction of routing slots each expert receives times
        # the mean router probability it gets, scaled so a perfectly even spread scores 1.0.
        load = F.one_hot(top_i, self.n).sum(1).float().mean(0) / self.k
        balance = self.n * (load * probs.mean(0)).sum()
        z = torch.logsumexp(logits.masked_fill(torch.isinf(logits), -1e4), -1).pow(2).mean()
        supervised = None
        if domains is not None:
            d = domains.reshape(-1)
            mask = d >= 0
            if mask.any():
                supervised = F.nll_loss(probs[mask].clamp_min(1e-9).log(), d[mask])
        if self.keep_trace:
            self.trace = top_i.detach().view(*shape[:-1], self.k)
        stats = {"load": load.detach(), "entropy": -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean().detach(),
                 "top_mass": probs.topk(self.k, -1).values.sum(-1).mean().detach()}
        return out.view(shape), balance, z, supervised, stats


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, moe: bool):
        super().__init__()
        self.n1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.n2, self.ffn = RMSNorm(cfg.d_model), FeedForward(cfg.d_model, cfg._ffn_dim())
        self.moe = ExpertMixture(cfg) if moe else None

    def forward(self, x, cos, sin, domains=None):
        x = x + self.attn(self.n1(x), cos, sin)
        h = self.n2(x)
        if self.moe is None:
            return x + self.ffn(h), None
        routed, balance, z, supervised, stats = self.moe(h, domains)
        return x + self.ffn(h) + routed, (balance, z, supervised, stats)


class ArtemisLM(nn.Module):
    """The full Artemis model: Prime Core with the expert layers wired into its forward pass."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0 and cfg.n_heads % cfg.n_kv_heads == 0
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i in cfg.moe_layers) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        cos, sin = rope_tables(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.route_coef = 0.0  # weight of the routing-supervision loss; set by the trainer
        self.last_stats: dict = {}
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith(("o.weight", "down.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    @property
    def mixtures(self) -> list[ExpertMixture]:
        return [b.moe for b in self.blocks if b.moe is not None]

    def forward(self, idx, targets=None, domains=None):
        """Returns (logits, loss). loss = next-token loss + the router terms; each part is in self.last_stats."""
        x = self.embed(idx)
        aux = []
        for blk in self.blocks:
            x, a = blk(x, self.cos, self.sin, domains)
            if a is not None:
                aux.append(a)
        logits = self.head(self.norm(x))
        loss = None
        stats = {}
        if aux:
            balance = torch.stack([a[0] for a in aux]).mean()
            z = torch.stack([a[1] for a in aux]).mean()
            sup = [a[2] for a in aux if a[2] is not None]
            stats = {"balance_loss": balance.detach(), "z_loss": z.detach(),
                     "load": torch.stack([a[3]["load"] for a in aux]), "entropy": torch.stack([a[3]["entropy"] for a in aux]),
                     "top_mass": torch.stack([a[3]["top_mass"] for a in aux])}
        if targets is not None:
            lm = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1), ignore_index=-100)
            loss = lm
            stats["lm_loss"] = lm.detach()
            if aux:
                loss = loss + self.cfg.balance_coef * balance + self.cfg.router_z_coef * z
                if sup and self.route_coef:
                    route = torch.stack(sup).mean()
                    loss = loss + self.route_coef * route
                    stats["route_loss"] = route.detach()
        self.last_stats = stats
        return logits, loss

    @torch.no_grad()
    def stream(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int = 50, eos_id=None):
        """Yields one new token id at a time."""
        eos = set([eos_id] if isinstance(eos_id, int) else (eos_id or []))
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.max_seq_len:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            yield int(nxt[0, 0])
            if int(nxt[0, 0]) in eos:
                break

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int = 50, eos_id: int | None = None):
        new = list(self.stream(idx, max_new_tokens, temperature, top_k, eos_id)) if idx.size(0) == 1 else None
        if new is not None:
            return torch.cat([idx, torch.tensor([new], dtype=idx.dtype, device=idx.device)], dim=1)
        for _ in range(max_new_tokens):  # batched sampling
            logits, _ = self(idx[:, -self.cfg.max_seq_len:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx


def parameter_groups(model: ArtemisLM) -> dict[str, list[str]]:
    """Every trainable parameter name, grouped as prime_core / router / expert:<name>."""
    names = model.cfg.expert_names
    groups: dict[str, list[str]] = {"prime_core": [], "router": [], **{f"expert:{n}": [] for n in names}}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".moe.router." in name:
            groups["router"].append(name)
        elif ".moe.experts." in name:
            groups[f"expert:{names[int(name.split('.moe.experts.')[1].split('.')[0])]}"].append(name)
        else:
            groups["prime_core"].append(name)
    return groups
