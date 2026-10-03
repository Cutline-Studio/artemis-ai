"""Evidence that the Prime Core and the ten experts are real, wired in, trained, saved, migrated, and served."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from artemis import checkpoint
from artemis.corpus import write as write_corpus
from artemis.data import TokenStream, prepare
from artemis.model import EXPERT_NAMES, ArtemisLM, ExpertMixture, ModelConfig, parameter_groups
from artemis.tokenizer import train_tokenizer
from artemis.train import main as train_main, model_config, optimizer_audit


def tiny(**kw):
    return ModelConfig(**{"vocab_size": 256, "d_model": 64, "n_layers": 3, "n_heads": 4, "n_kv_heads": 2, "max_seq_len": 32,
                          "moe_layers": (1,), "expert_ffn_mult": 0.5, **kw})


@pytest.fixture(scope="module")
def labeled(tmp_path_factory):
    d = tmp_path_factory.mktemp("labeled")
    sources = write_corpus(d / "corpus", 300, seed=1)
    train_tokenizer([s["path"] for s in sources], 2048, d / "tok.json")
    m = prepare(sources, str(d / "tok.json"), str(d / "data"), val_fraction=0.1)
    return {"dir": d, "tok": d / "tok.json", "data": d / "data", "manifest": m}


def test_ten_named_experts_and_registered_parameters():
    model = ArtemisLM(tiny())
    assert model.cfg.expert_names == EXPERT_NAMES == ("intent", "architecture", "planning", "build", "test", "deploy",
                                                       "memory", "security", "observability", "cost")
    (mix,) = model.mixtures
    assert len(mix.experts) == 10 and mix.router.out_features == 10 and mix.k == 2
    groups = parameter_groups(model)
    assert groups["router"] == ["blocks.1.moe.router.weight"]
    for i, name in enumerate(EXPERT_NAMES):
        assert groups[f"expert:{name}"] == [f"blocks.1.moe.experts.{i}.{p}.weight" for p in ("gate", "up", "down")]
    assert sum(len(v) for v in groups.values()) == len(list(model.parameters()))
    assert model.cfg.param_count() == sum(p.numel() for p in model.parameters())
    assert model.cfg.active_param_count() < model.cfg.param_count()


def test_router_picks_top_two_with_normalized_weights():
    torch.manual_seed(0)
    cfg = tiny()
    mix = ExpertMixture(cfg)
    x = torch.randn(2, 5, cfg.d_model)
    out, balance, z, sup, stats = mix(x)
    flat = x.reshape(-1, cfg.d_model)
    probs = mix.router(flat).softmax(-1)
    w, idx = probs.topk(2, -1)
    w = w / w.sum(-1, keepdim=True)
    expected = torch.stack([sum(w[t, j] * mix.experts[int(idx[t, j])](flat[t]) for j in range(2)) for t in range(flat.size(0))])
    assert torch.allclose(out.reshape(-1, cfg.d_model), expected, atol=1e-5)
    assert torch.allclose(w.sum(-1), torch.ones(flat.size(0)))
    assert torch.allclose(stats["load"].sum(), torch.tensor(1.0)) and balance > 0 and z > 0 and sup is None


def test_forward_pass_calls_router_and_experts():
    model = ArtemisLM(tiny())
    calls = Counter()
    (mix,) = model.mixtures
    mix.router.register_forward_hook(lambda *a: calls.update(["router"]))
    for i, e in enumerate(mix.experts):
        e.register_forward_hook(lambda *a, i=i: calls.update([EXPERT_NAMES[i]]))
    model(torch.randint(0, 256, (4, 32)))
    assert calls["router"] == 1
    assert sum(calls[n] for n in EXPERT_NAMES) >= 2  # at least the top-two experts of some tokens ran
    # experts' outputs reach the logits: zeroing every expert changes them
    a, _ = model(torch.ones(1, 8, dtype=torch.long))
    with torch.no_grad():
        for e in mix.experts:
            e.down.weight.zero_()
    b, _ = model(torch.ones(1, 8, dtype=torch.long))
    assert not torch.allclose(a, b)


def test_router_and_selected_experts_receive_gradients():
    torch.manual_seed(0)
    model = ArtemisLM(tiny())
    model.route_coef = 0.1
    x = torch.randint(0, 256, (4, 32))
    d = torch.randint(0, 10, (4, 32))
    for m in model.mixtures:
        m.keep_trace = True
    _, loss = model(x, x, d)
    loss.backward()
    (mix,) = model.mixtures
    used = set(mix.trace.flatten().tolist())
    assert mix.router.weight.grad is not None and mix.router.weight.grad.abs().sum() > 0
    for i, e in enumerate(mix.experts):
        g = e.gate.weight.grad
        if i in used:
            assert g is not None and g.abs().sum() > 0, EXPERT_NAMES[i]
        else:
            assert g is None or g.abs().sum() == 0
    assert {"lm_loss", "balance_loss", "z_loss", "route_loss"} <= set(model.last_stats)


def test_optimizer_holds_every_trainable_parameter():
    model = ArtemisLM(tiny())
    opt = torch.optim.AdamW(model.parameters())
    audit = optimizer_audit(model, opt)
    assert audit["trainable_params"] == model.cfg.param_count()
    assert audit["groups"]["router"]["tensors"] == 1 and audit["groups"]["expert:security"]["tensors"] == 3
    partial = torch.optim.AdamW([p for n, p in model.named_parameters() if ".moe." not in n])
    with pytest.raises(RuntimeError, match="moe.router"):
        optimizer_audit(model, partial)


def test_save_reload_preserves_architecture_and_parameters(tmp_path):
    torch.manual_seed(0)
    model = ArtemisLM(tiny())
    opt = torch.optim.AdamW(model.parameters())
    model(torch.randint(0, 256, (2, 16)), torch.randint(0, 256, (2, 16)))[1].backward()
    opt.step()
    checkpoint.save(tmp_path / "c.pt", model, opt, 7)
    again, ck = checkpoint.load_model(tmp_path / "c.pt")
    assert again.cfg == model.cfg and ck["step"] == 7 and ck["format_version"] == 2
    for (n, p), (n2, p2) in zip(model.state_dict().items(), again.state_dict().items()):
        assert n == n2 and torch.equal(p, p2)
    ids = torch.randint(0, 256, (1, 16))
    assert torch.equal(model.eval()(ids)[0], again.eval()(ids)[0])
    # strict: a missing tensor is named, never skipped
    bad = torch.load(tmp_path / "c.pt", weights_only=False)
    del bad["model"]["blocks.1.moe.experts.7.up.weight"]
    torch.save(bad, tmp_path / "bad.pt")
    with pytest.raises(checkpoint.CheckpointError, match="experts.7.up.weight"):
        checkpoint.load_model(tmp_path / "bad.pt")


def _v1_checkpoint(path):
    """A format-1 checkpoint exactly as the old trainer wrote it (dense model, no version fields)."""
    torch.manual_seed(1)
    cfg = tiny(moe_layers=())
    model = ArtemisLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        model(torch.randint(0, 256, (2, 16)), torch.randint(0, 256, (2, 16)))[1].backward()
        opt.step()
        opt.zero_grad()
    old = cfg.to_dict()
    for k in ("moe_layers", "moe_every", "n_experts", "experts_per_token", "expert_ffn_mult", "expert_names", "balance_coef", "router_z_coef"):
        old.pop(k)
    torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "step": 2, "config": old, "args": {}}, path)
    return model, opt


def test_v1_checkpoint_is_refused_then_migrated_without_loss(tmp_path):
    src = tmp_path / "v1.pt"
    dense, opt = _v1_checkpoint(src)
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    with pytest.raises(checkpoint.CheckpointError, match="format-1"):
        checkpoint.load_model(src)
    with pytest.raises(checkpoint.CheckpointError, match="new file"):
        checkpoint.migrate_v1(src, src, moe_layers=(1,))
    record = checkpoint.migrate_v1(src, tmp_path / "v2.pt", moe_layers=(1,), expert_ffn_mult=0.5)
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before  # original untouched
    assert record["dropped_tensors"] == [] and record["copied_tensors"] == len(dense.state_dict())
    assert len(record["new_tensors"]) == 1 + 10 * 3
    assert json.loads((tmp_path / "v2.pt.migration.json").read_text())["source_sha256"] == before
    moe, ck = checkpoint.load_model(tmp_path / "v2.pt")
    ids = torch.randint(0, 256, (2, 20))
    assert torch.allclose(dense.eval()(ids)[0], moe.eval()(ids)[0], atol=1e-6)  # experts start as no-ops
    # AdamW moments for existing weights carried over by name
    new_opt = torch.optim.AdamW(moe.parameters(), lr=1e-3)
    new_opt.load_state_dict(ck["optim"])
    names = [n for n, _ in moe.named_parameters()]
    old_state = opt.state_dict()["state"][0]["exp_avg"]
    assert torch.equal(new_opt.state_dict()["state"][names.index("embed.weight")]["exp_avg"], old_state)


def test_migrated_checkpoint_resumes_training(tmp_path, labeled):
    # format-1 smoke run -> migrate -> the trainer resumes it with experts and keeps counting steps
    cfg = model_config("smoke")
    model = ArtemisLM(cfg)
    old = {k: v for k, v in cfg.to_dict().items() if k in ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
                                                             "ffn_mult", "max_seq_len", "rope_theta", "dropout")}
    torch.save({"model": model.state_dict(), "optim": None, "step": 5, "config": old, "args": {}}, tmp_path / "v1.pt")
    run = tmp_path / "run"
    run.mkdir()
    checkpoint.migrate_v1(tmp_path / "v1.pt", run / "latest.pt", moe_layers=(1,), expert_ffn_mult=0.5)
    r = train_main(["--size", "smoke-moe", "--data", str(labeled["data"]), "--out", str(run), "--steps", "15",
                    "--batch", "4", "--eval-every", "15", "--eval-batches", "2", "--ckpt-every", "15"])
    assert r["step"] == 15
    steps = [json.loads(l)["step"] for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert steps[0] == 6 and steps[-1] == 15


def test_training_learns_records_routing_and_resumes(labeled, tmp_path):
    out = tmp_path / "moe"
    args = ["--size", "smoke-moe", "--data", str(labeled["data"]), "--out", str(out), "--batch", "8", "--lr", "3e-3",
            "--eval-every", "60", "--eval-batches", "4", "--ckpt-every", "60"]
    train_main(args + ["--steps", "120"])
    recs = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]
    assert recs[-1]["lm_loss"] < recs[0]["lm_loss"] * 0.75
    assert recs[-1]["route_loss"] < recs[0]["route_loss"]
    assert len(recs[0]["expert_load"]) == 1 and len(recs[0]["expert_load"][0]) == 10
    norms = recs[0]["router_expert_grad_norms"]
    assert norms["router"] > 0 and sum(v > 0 for k, v in norms.items() if k.startswith("expert:")) >= 2
    run = json.loads((out / "run.json").read_text())
    assert run["optimizer"]["trainable_params"] == run["params"]
    train_main(args + ["--steps", "150"])
    steps = [json.loads(l)["step"] for l in (out / "metrics.jsonl").read_text().splitlines()]
    assert steps[120] == 121 and steps[-1] == 150 and len(steps) == 150

    from artemis.evaluate import ablation, per_expert
    model, _ = checkpoint.load_model(out / "latest.pt")
    val = TokenStream(labeled["data"] / "val.bin", labeled["manifest"]["dtype"], model.cfg.max_seq_len)
    pe = per_expert(model, val, 8, 6)
    assert set(pe["domains"]) == set(EXPERT_NAMES) and all(v["tokens"] > 0 for v in pe["domains"].values())
    hit = sum(v["router_hit_rate"][0] for v in pe["domains"].values()) / 10
    assert hit > 0.2 + 0.1, f"supervised router should beat chance (0.2) on held-out domains, got {hit}"
    ab = ablation(model, val, 8, 3)
    assert len(ab["loss_increase"]["matrix"]) == 10


def test_moe_export_matches_qwen2_moe(labeled, tmp_path):
    transformers = pytest.importorskip("transformers")
    from artemis.export import export_model
    run = tmp_path / "run"
    train_main(["--size", "smoke-moe", "--data", str(labeled["data"]), "--out", str(run), "--steps", "10",
                "--batch", "4", "--eval-every", "10", "--eval-batches", "2", "--ckpt-every", "10"])
    out = export_model(str(run / "latest.pt"), str(labeled["tok"]), str(tmp_path / "hf"), dtype=torch.float32)
    assert json.loads((out / "config.json").read_text())["architectures"] == ["Qwen2MoeForCausalLM"]
    hf = transformers.AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    ours, _ = checkpoint.load_model(run / "latest.pt")
    ids = torch.randint(0, 2048, (2, 40))
    with torch.no_grad():
        a, _ = ours.eval()(ids)
        b = hf(ids).logits
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def test_inference_streams_through_the_experts(labeled, tmp_path):
    from artemis.infer import LocalModel
    run = tmp_path / "run"
    train_main(["--size", "smoke-moe", "--data", str(labeled["data"]), "--out", str(run), "--steps", "10",
                "--batch", "4", "--eval-every", "10", "--eval-batches", "2", "--ckpt-every", "10"])
    lm = LocalModel(str(run / "latest.pt"), str(labeled["tok"]))
    assert lm.info()["expert_layers"] == [1] and lm.info()["format_version"] == 2
    trace = Counter()
    torch.manual_seed(0)
    pieces = list(lm.stream([{"role": "user", "content": "Deploy checkout to eastus"}], max_tokens=12, trace=trace))
    assert pieces and all(isinstance(p, str) for p in pieces)
    assert sum(trace.values()) >= 2 and {e for _, e in trace} <= set(EXPERT_NAMES)


def test_lora_adapters_leave_experts_and_router_alone():
    from artemis.specialize import add_lora
    model = ArtemisLM(tiny())
    add_lora(model, rank=4)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and not any(".moe." in n for n in trainable)
    assert type(model.blocks[1].moe.experts[0].gate).__name__ == "Linear"


def test_stop_and_resume_matches_an_uninterrupted_run(labeled, tmp_path):
    args = ["--size", "smoke-moe", "--data", str(labeled["data"]), "--batch", "4", "--lr", "3e-3", "--steps", "30",
            "--eval-every", "100", "--eval-batches", "2", "--ckpt-every", "100"]
    train_main(args + ["--out", str(tmp_path / "straight")])
    train_main(args + ["--out", str(tmp_path / "resumed"), "--stop-at", "12"])
    train_main(args + ["--out", str(tmp_path / "resumed")])
    a, _ = checkpoint.load_model(tmp_path / "straight" / "latest.pt")
    b, ck = checkpoint.load_model(tmp_path / "resumed" / "latest.pt")
    assert ck["step"] == 30
    for (n, p), (_, q) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.allclose(p, q, atol=1e-6), n


def test_api_server_loads_the_checkpoint_and_streams_model_tokens(labeled, tmp_path, monkeypatch):
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    from artemis.server import build_app, make_handler
    run = tmp_path / "run"
    train_main(["--size", "smoke-moe", "--data", str(labeled["data"]), "--out", str(run), "--steps", "10",
                "--batch", "4", "--eval-every", "10", "--eval-batches", "2", "--ckpt-every", "10"])
    monkeypatch.delenv("ARTEMIS_INFERENCE_URL", raising=False)
    monkeypatch.setenv("ARTEMIS_CHECKPOINT", str(run / "latest.pt"))
    monkeypatch.setenv("ARTEMIS_TOKENIZER", str(labeled["tok"]))
    monkeypatch.setenv("ARTEMIS_DB", ":memory:")
    app, biz = build_app()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app, biz))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        status = json.load(urllib.request.urlopen(base + "/v1/status"))
        assert status["serving"] and status["model"]["checkpoint"] == str(run / "latest.pt")
        assert status["model"]["experts"] == list(EXPERT_NAMES) and status["model"]["step"] == 10
        calls = Counter()
        for m in app.backend.model.model.mixtures:
            m.router.register_forward_hook(lambda *a: calls.update(["router"]))
        req = urllib.request.Request(base + "/v1/chat/stream", data=json.dumps({"message": "hi", "tier": "gpt-1-base"}).encode(),
                                     headers={"Content-Type": "application/json"})
        body = urllib.request.urlopen(req).read().decode()
        events = [json.loads(c.split("data: ", 1)[1]) for c in body.strip().split("\n\n")]
        assert events[-1]["type"] == "done" and calls["router"] >= 1
    finally:
        srv.shutdown()
