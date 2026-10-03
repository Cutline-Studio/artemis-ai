"""External persistent memory: notes saved per account in the database and retrieved into the prompt.

This is not the model's weights. Weights hold what Artemis learned in training and change only by training;
memory holds facts about one account, can be read, edited and deleted at any time, and reaches the model only as
text placed in its context, inside a block labeled as retrieved memory. (The "memory" expert is a neural module
inside the model; it does not store anything between conversations.)

Retrieval is lexical (BM25 over the account's notes), so it needs no model and is the same on every server.
Limits are enforced here, in the application: notes per account, characters per note, notes per prompt.
"""
from __future__ import annotations

import math
import re
import time
import uuid
from collections import Counter

from .store import Store

MAX_NOTES = 2000
MAX_CHARS = 2000
WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return WORD.findall(text.lower())


class MemoryStore:
    def __init__(self, db: Store):
        self.db = db

    def add(self, account: str, text: str, source: str = "user") -> str:
        text = text.strip()
        if not text or len(text) > MAX_CHARS:
            raise ValueError(f"a memory must be 1-{MAX_CHARS} characters")
        if self.db.one("SELECT COUNT(*) FROM memories WHERE account=?", (account,))[0] >= MAX_NOTES:
            raise ValueError(f"memory is full ({MAX_NOTES} notes); delete some first")
        mid = uuid.uuid4().hex
        self.db.execute("INSERT INTO memories VALUES (?, ?, ?, ?, ?)", (mid, account, text, time.time(), source[:40]))
        return mid

    def list(self, account: str) -> list[dict]:
        rows = self.db.all("SELECT id, text, created, source FROM memories WHERE account=? ORDER BY created", (account,))
        return [{"id": r[0], "text": r[1], "created": r[2], "source": r[3]} for r in rows]

    def delete(self, account: str, memory_id: str) -> bool:
        if not self.db.one("SELECT 1 FROM memories WHERE account=? AND id=?", (account, memory_id)):
            return False
        self.db.execute("DELETE FROM memories WHERE account=? AND id=?", (account, memory_id))
        return True

    def retrieve(self, account: str, query: str, k: int = 3) -> list[dict]:
        """The k notes that best match the query (BM25); notes sharing no words with it are never returned."""
        notes = self.list(account)
        q = set(_words(query))
        if not notes or not q:
            return []
        docs = [Counter(_words(n["text"])) for n in notes]
        avg = sum(sum(d.values()) for d in docs) / len(docs)
        df = Counter(w for d in docs for w in d)
        scored = []
        for n, d in zip(notes, docs):
            length = sum(d.values())
            s = sum(math.log(1 + (len(docs) - df[w] + 0.5) / (df[w] + 0.5)) * d[w] * 2.2 / (d[w] + 1.2 * (0.25 + 0.75 * length / avg))
                    for w in q if w in d)
            if s > 0:
                scored.append((s, n))
        return [n for _, n in sorted(scored, key=lambda x: -x[0])[:k]]


def context_block(notes: list[dict]) -> str:
    """How retrieved notes are shown to the model: clearly marked as external memory, not instructions."""
    if not notes:
        return ""
    lines = "\n".join(f"- {n['text']}" for n in notes)
    return ("[Retrieved memory: notes saved for this account in Artemis's memory store, not part of the model's training. "
            "Use them only if relevant.]\n" + lines + "\n[End of retrieved memory]")
