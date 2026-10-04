"""Check real deployed model -> tool -> result -> answer conversations.

python -m artemis.smoke_chat --url https://YOUR-API --with-python
Optional paid account key is read from ARTEMIS_SMOKE_API_KEY, never a command argument.
These calls consume the account's normal message and tool allowances.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import urllib.error
import urllib.request


def conversation(base, prompt, key="", timeout=180):
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/stream",
        data=json.dumps({"message": prompt, "tier": "gpt-1-base"}).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = []
        for raw in response:
            line = raw.decode().rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip())
            elif not line and data:
                yield json.loads("\n".join(data))
                data = []
        if data:
            yield json.loads("\n".join(data))


def check(events, name, expected=None):
    if not events or events[-1].get("type") != "done" or events[-1].get("status") != "ok":
        terminal = events[-1].get("status", events[-1].get("type")) if events else "missing stream"
        raise ValueError(f"conversation did not complete successfully: {terminal}")
    answer = events[-1].get("answer", "").strip()
    if not answer or "<tool_call>" in answer:
        raise ValueError("missing or malformed final answer")
    results = [e for e in events if e.get("type") == "tool_result" and e.get("name") == name and e.get("ok")]
    if not results:
        raise ValueError(f"model did not successfully use {name}")
    if expected is not None:
        if not any(str(expected) == r.get("output", "").strip() for r in results):
            raise ValueError("Python output does not match the computed answer")
        if not re.search(rf"\b{expected}\b", answer):
            raise ValueError("final answer does not include the computed result")
    else:
        urls = [s.get("url", "") for r in results for s in r.get("sources", [])]
        if not any(u and u in answer for u in urls):
            raise ValueError("final answer does not cite a returned source URL")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--with-python", action="store_true", help="requires an authenticated paid plan")
    args = parser.parse_args(argv)
    key = os.environ.get("ARTEMIS_SMOKE_API_KEY", "")
    probes = [
        ("web_search", "Use web_search to find Python programming language documentation. Summarize one result and cite its full URL.", None),
        ("web_fetch", "Use web_fetch to read https://en.wikipedia.org/wiki/Python_(programming_language). Summarize the page and cite its full URL.", None),
    ]
    if args.with_python:
        a, b = secrets.randbelow(900) + 100, secrets.randbelow(900) + 100
        probes.append(("run_python", f"Use run_python to execute print({a}**2 + {b}**2). State the result from stdout.", a*a + b*b))
    failed = False
    for name, prompt, expected in probes:
        try:
            check(list(conversation(args.url, prompt, key)), name, expected)
            print(f"PASS {name}: model used tool and completed answer")
        except (ValueError, urllib.error.URLError, TimeoutError, OSError) as e:
            # HTTPError text contains the URL/status, never the bearer key.
            print(f"FAIL {name}: {e}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
