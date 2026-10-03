"""Artemis tokenizer: byte-level BPE trained on Artemis's own corpus."""
from __future__ import annotations

import re
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .brains import SPECIALIST_IDS

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"] + [
    f"<|brain:{b}|>" for b in ("artemis",) + SPECIALIST_IDS
]


def train_tokenizer(files: list[str], vocab_size: int, out_path: str | Path) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train(files, trainer)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


CONTROL = re.compile(r"<\|[a-z_]+(?::[a-z_]+)?\|>")


def clean(text: str) -> str:
    """Removes control-token look-alikes from message text, so a user or a web page can't write "<|system|>" and
    have it read as a real turn boundary."""
    return CONTROL.sub("", text)


def format_chat(messages: list[dict], brain: str = "artemis") -> str:
    """Render a conversation in Artemis's chat format."""
    parts = [f"<|bos|><|brain:{brain}|>"]
    for m in messages:
        parts.append(f"<|{m['role']}|>{clean(m['content'])}<|end|>")
    return "".join(parts)
