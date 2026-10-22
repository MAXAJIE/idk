"""Tiny .env loader (no extra dependency).

Looks for backend/.env then backend/app/.env. Tolerates `KEY = value`,
quotes and comments. Never overrides variables already set in the shell.
"""
from __future__ import annotations

import os
from pathlib import Path

_BACKEND = Path(os.path.abspath(__file__)).parents[1]
CANDIDATES = (_BACKEND / ".env", _BACKEND / "app" / ".env")


def parse(text: str) -> dict[str, str]:
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        elif " #" in v:
            v = v.split(" #", 1)[0].rstrip()
        if k:
            out[k] = v
    return out


def load_env(paths=CANDIDATES) -> list[str]:
    loaded = []
    for p in paths:
        p = Path(p)
        if p.is_file():
            for k, v in parse(p.read_text(encoding="utf-8")).items():
                if v and not os.environ.get(k):
                    os.environ[k] = v
            loaded.append(str(p))
    return loaded
