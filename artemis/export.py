"""Export Artemis weights to a layout vLLM serves, plus each brain adapter in PEFT layout.

Dense models export as Llama. Models with expert layers export as Qwen2-MoE, which has the same pieces:
per-layer router ("mlp.gate"), routed experts with top-k and renormalized weights (norm_topk_prob), a shared
dense feed-forward (our Prime Core FFN, "mlp.shared_expert"), and dense-only layers ("mlp_only_layers").
Two exact adjustments: Qwen2-MoE scales the shared FFN by sigmoid(shared_expert_gate(x)), so the gate is
exported as zeros (sigmoid(0) = 0.5) and the shared down projection is doubled; and its q/k/v projections carry
biases, exported as zeros. The export test checks the logits match.

Artemis applies rotary embeddings to interleaved pairs; Llama/Qwen rotate halves. The q/k projection rows
(and the matching LoRA B rows) are permuted so both give identical results.

python -m artemis.export --checkpoint runs/1b/latest.pt --tokenizer runs/tok.json --out serve/artemis-1b \
    --adapters runs/brains/*.adapter.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .checkpoint import read
from .model import ModelConfig
from .tokenizer import SPECIAL_TOKENS


def _permute(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    out = w.shape[0]
    head_dim = out // n_heads
    return w.view(n_heads, head_dim // 2, 2, -1).transpose(1, 2).reshape(out, -1)


def hf_config(cfg: ModelConfig, tok_path: str) -> dict:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    return {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama", "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.d_model, "intermediate_size": cfg._ffn_dim(), "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads, "num_key_value_heads": cfg.n_kv_heads, "max_position_embeddings": cfg.max_seq_len,
        "rms_norm_eps": 1e-6, "rope_theta": cfg.rope_theta, "tie_word_embeddings": True, "hidden_act": "silu",
        "attention_bias": False, "mlp_bias": False, "torch_dtype": "bfloat16",
        "bos_token_id": tok.token_to_id("<|bos|>"), "eos_token_id": tok.token_to_id("<|end|>"),
        "pad_token_id": tok.token_to_id("<|pad|>"),
    }


def hf_moe_config(cfg: ModelConfig, tok_path: str) -> dict:
    base = hf_config(cfg, tok_path)
    base.update({"architectures": ["Qwen2MoeForCausalLM"], "model_type": "qwen2_moe", "num_experts": cfg.n_experts,
                 "num_experts_per_tok": cfg.experts_per_token, "norm_topk_prob": True, "moe_intermediate_size": cfg._expert_dim(),
                 "shared_expert_intermediate_size": cfg._ffn_dim(), "decoder_sparse_step": 1,
                 "mlp_only_layers": [i for i in range(cfg.n_layers) if i not in cfg.moe_layers],
                 "router_aux_loss_coef": cfg.balance_coef, "output_router_logits": False, "use_sliding_window": False,
                 "artemis_expert_names": list(cfg.expert_names)})
    return base


def export_model(ckpt_path: str, tok_path: str, out_dir: str, dtype=torch.bfloat16) -> Path:
    ck = read(ckpt_path)
    cfg = ModelConfig(**ck["config"])
    sd = ck["model"]
    moe = bool(cfg.moe_layers)
    hf = {"model.embed_tokens.weight": sd["embed.weight"], "model.norm.weight": sd["norm.weight"]}
    names = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
             "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
    for i in range(cfg.n_layers):
        p = f"blocks.{i}."
        L = f"model.layers.{i}."
        hf[L + "input_layernorm.weight"] = sd[p + "n1.weight"]
        hf[L + "post_attention_layernorm.weight"] = sd[p + "n2.weight"]
        sparse = i in cfg.moe_layers
        for ours, theirs in names.items():
            part = "attn" if ours in ("q", "k", "v", "o") else "ffn"
            w = sd[f"{p}{part}.{ours}.weight"]
            if ours == "q":
                w = _permute(w, cfg.n_heads)
            elif ours == "k":
                w = _permute(w, cfg.n_kv_heads)
            if sparse and part == "ffn":
                theirs = theirs.replace("mlp.", "mlp.shared_expert.")
                w = w * 2 if ours == "down" else w  # undoes the 0.5 from sigmoid(0) on the shared-expert gate
            hf[L + theirs + ".weight"] = w
            if moe and ours in ("q", "k", "v"):
                hf[L + theirs + ".bias"] = torch.zeros(w.shape[0])
        if sparse:
            hf[L + "mlp.gate.weight"] = sd[p + "moe.router.weight"]
            hf[L + "mlp.shared_expert_gate.weight"] = torch.zeros(1, cfg.d_model)
            for e in range(cfg.n_experts):
                for part in ("gate", "up", "down"):
                    hf[f"{L}mlp.experts.{e}.{part}_proj.weight"] = sd[f"{p}moe.experts.{e}.{part}.weight"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.to(dtype).contiguous() for k, v in hf.items()}, out / "model.safetensors")
    (out / "config.json").write_text(json.dumps(hf_moe_config(cfg, tok_path) if moe else hf_config(cfg, tok_path), indent=2))
    _export_tokenizer(tok_path, out)
    return out


def _export_tokenizer(tok_path: str, out: Path) -> None:
    (out / "tokenizer.json").write_text(Path(tok_path).read_text())
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<|bos|>", "eos_token": "<|end|>", "pad_token": "<|pad|>",
        "additional_special_tokens": SPECIAL_TOKENS[3:], "model_max_length": 1_000_000,
        "chat_template": "{{ '<|bos|>' }}{% for m in messages %}{{ '<|' + m['role'] + '|>' + m['content'] + '<|end|>' }}{% endfor %}"
                         "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"}, indent=2))


def export_adapter(adapter_path: str, out_dir: str) -> Path:
    ad = torch.load(adapter_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ad["config"])
    names = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
             "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
    tensors = {}
    for key, w in ad["lora"].items():
        # e.g. blocks.3.attn.q.a  or  blocks.3.ffn.up.b  (experts and routers never get adapters)
        _, layer, _, proj, ab = key.split(".")
        if ab == "b" and proj in ("q", "k"):
            w = _permute(w, cfg.n_heads if proj == "q" else cfg.n_kv_heads)
        target = names[proj]
        if int(layer) in cfg.moe_layers and proj in ("gate", "up", "down"):
            target = target.replace("mlp.", "mlp.shared_expert.")
            w = w * 2 if (proj == "down" and ab == "b") else w
        tensors[f"base_model.model.model.layers.{layer}.{target}.lora_{'A' if ab == 'a' else 'B'}.weight"] = w.contiguous()
    out = Path(out_dir) / ad["brain"]
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out / "adapter_model.safetensors")
    (out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": ad["rank"], "lora_alpha": ad.get("alpha", ad["rank"] * 2),
        "target_modules": (r".*\.(self_attn\.(q|k|v|o)_proj|mlp\.(shared_expert\.)?(gate|up|down)_proj)$" if cfg.moe_layers
                           else sorted({v.split(".")[-1] for v in names.values()})), "lora_dropout": 0.0, "bias": "none",
        "base_model_name_or_path": "artemis-foundation"}, indent=2))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapters", nargs="*", default=[])
    a = ap.parse_args(argv)
    print(export_model(a.checkpoint, a.tokenizer, a.out))
    for p in a.adapters:
        print(export_adapter(p, Path(a.out) / "adapters"))


if __name__ == "__main__":
    main()
