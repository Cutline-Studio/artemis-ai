"""End-to-end: tokenizer -> data shards -> train from scratch -> resume -> generate."""
import json
import sysconfig
from pathlib import Path

import pytest
import torch

from artemis.data import prepare
from artemis.model import ArtemisLM
from artemis.tokenizer import load_tokenizer, train_tokenizer
from artemis.train import main as train_main, model_config


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    # Smoke-test corpus: Python standard library source (PSF license), never used for real training.
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    text = "\n".join(p.read_text(errors="ignore") for p in sorted(stdlib.glob("*.py"))[:60])
    d = tmp_path_factory.mktemp("corpus")
    f = d / "stdlib.txt"
    f.write_text(text)
    tok = d / "tok.json"
    train_tokenizer([str(f)], 2048, tok)
    data = d / "data"
    manifest = prepare([{"path": str(f), "license": "PSF-2.0", "origin": "python stdlib (smoke test only)"}], str(tok), str(data))
    return {"dir": d, "tok": tok, "data": data, "manifest": manifest}


def test_prepare_records_lineage(corpus):
    m = corpus["manifest"]
    assert m["train_tokens"] > 100_000
    assert m["sources"][0]["license"] == "PSF-2.0" and len(m["sources"][0]["sha256"]) == 64


def test_prepare_rejects_unlicensed(corpus, tmp_path):
    with pytest.raises(ValueError, match="license"):
        prepare([{"path": str(corpus["dir"] / "stdlib.txt")}], str(corpus["tok"]), str(tmp_path / "x"))


def test_train_learns_and_resumes(corpus, tmp_path):
    out = tmp_path / "run"
    first = train_main(["--size", "smoke", "--data", str(corpus["data"]), "--out", str(out), "--steps", "120",
                        "--batch", "8", "--lr", "3e-3", "--eval-every", "60", "--ckpt-every", "60"])
    lines = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]
    assert lines[-1]["loss"] < lines[0]["loss"] * 0.75, "loss should fall well below the random-init loss"
    assert first["val_loss"] < 6.0
    # resume: continuing to 150 starts at step 120, not from scratch
    train_main(["--size", "smoke", "--data", str(corpus["data"]), "--out", str(out), "--steps", "150",
                "--batch", "8", "--lr", "3e-3", "--eval-every", "30", "--ckpt-every", "30"])
    steps = [json.loads(l)["step"] for l in (out / "metrics.jsonl").read_text().splitlines()]
    assert steps[120] == 121 and steps[-1] == 150 and len(steps) == 150

    ck = torch.load(out / "latest.pt", weights_only=False)
    model = ArtemisLM(model_config("smoke"))
    model.load_state_dict(ck["model"])
    tok = load_tokenizer(corpus["tok"])
    ids = torch.tensor([tok.encode("def ").ids])
    out_ids = model.generate(ids, 20)
    assert out_ids.shape[1] == ids.shape[1] + 20


def test_specialize_brain_adapter(corpus, tmp_path):
    from artemis.specialize import main as spec_main
    found = tmp_path / "found"
    train_main(["--size", "smoke", "--data", str(corpus["data"]), "--out", str(found), "--steps", "40",
                "--batch", "8", "--lr", "3e-3", "--eval-every", "40", "--ckpt-every", "40"])
    data = tmp_path / "saturn.jsonl"
    data.write_text("\n".join(json.dumps({"messages": [{"role": "user", "content": f"Write a function that returns {i}."},
                                                         {"role": "assistant", "content": f"def f():\n    return {i}"}]})
                              for i in range(20)))
    r = spec_main(["--foundation", str(found / "latest.pt"), "--brain", "saturn", "--data", str(data),
                   "--tokenizer", str(corpus["tok"]), "--out", str(tmp_path / "brains"), "--steps", "80", "--lr", "3e-3"])
    assert r["last_loss"] < r["first_loss"]
    assert r["trainable_params"] < r["total_params"] * 0.2
    assert (tmp_path / "brains" / "saturn.adapter.pt").exists()


def test_export_matches_llama_layout(corpus, tmp_path):
    transformers = pytest.importorskip("transformers")
    from artemis.export import export_model
    run = tmp_path / "run"
    train_main(["--size", "smoke", "--data", str(corpus["data"]), "--out", str(run), "--steps", "20",
                "--batch", "4", "--eval-every", "20", "--ckpt-every", "20"])
    out = export_model(str(run / "latest.pt"), str(corpus["tok"]), str(tmp_path / "hf"), dtype=torch.float32)
    hf = transformers.AutoModelForCausalLM.from_pretrained(out, torch_dtype=torch.float32).eval()
    ck = torch.load(run / "latest.pt", weights_only=False)
    ours = ArtemisLM(model_config("smoke")).eval()
    ours.load_state_dict(ck["model"])
    ids = torch.randint(0, 2048, (2, 48))
    with torch.no_grad():
        a, _ = ours(ids)
        b = hf(ids).logits
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def test_business_knowledge_goes_into_designated_weights(corpus, tmp_path):
    """Self-knowledge (identity, plans, prices, brains) is trained into Artemis's full weights."""
    from artemis.knowledge import build
    from artemis.specialize import main as spec_main
    counts = build(tmp_path / "sft")
    assert counts["artemis"] > counts["saturn"] > 0  # orchestrator also learns routing
    text = (tmp_path / "sft" / "artemis.jsonl").read_text()
    assert "$20 a month" in text and "$200 a month" in text and "I'm Artemis" in text and "Cutline" not in text
    found = tmp_path / "found"
    train_main(["--size", "smoke", "--data", str(corpus["data"]), "--out", str(found), "--steps", "40",
                "--batch", "8", "--lr", "3e-3", "--eval-every", "40", "--ckpt-every", "40"])
    r = spec_main(["--foundation", str(found / "latest.pt"), "--brain", "artemis", "--data", str(tmp_path / "sft" / "artemis.jsonl"),
                   "--tokenizer", str(corpus["tok"]), "--out", str(tmp_path / "brains"), "--steps", "150", "--lr", "1e-3"])
    assert r["weights"] == "full" and r["last_loss"] < r["first_loss"]
    assert (tmp_path / "brains" / "artemis.full.pt").exists()
