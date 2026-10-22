"""LLM narration behind a narrow interface. Narration explains already-decided
structured records; it never feeds back into trading decisions.

openai_compatible works with Cortensor Portal, OpenRouter or Gemini's
OpenAI-compatible endpoint. Server-side env only (backend/.env is read
automatically): LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.

provider = "auto" uses the LLM when LLM_BASE_URL and LLM_MODEL are set,
otherwise the deterministic offline template.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

EV_RE = re.compile(r"ev:[a-z_]+:\d+")
SYSTEM = ("You explain a simulated paper-trading decision. Use ONLY the JSON provided. Cite evidence IDs "
          "verbatim (format ev:kind:ts). Never invent numbers. 4 sentences max. Research only, not advice.")


class Provider:
    name = "base"

    def narrate(self, record: dict) -> str:
        raise NotImplementedError


class OfflineProvider(Provider):
    name = "offline-template-1"

    def narrate(self, record: dict) -> str:
        p, r = record["proposal"], record["risk"]
        ev = ", ".join(p["evidence_ids"][:3]) or "none"
        return (f"Trader proposed {p['side']} ({p['discussion_adoption']} discussion) citing {ev}. "
                f"Risk Control: {r['outcome']} ({'; '.join(r['reasons'])}). {p['rationale']}.")


class LLMError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def llm_env() -> dict:
    return {"base_url": os.environ.get("LLM_BASE_URL", "").strip().rstrip("/"),
            "api_key": os.environ.get("LLM_API_KEY", "").strip(),
            "model": os.environ.get("LLM_MODEL", "").strip()}


def llm_status() -> dict:
    env = llm_env()
    missing = [k for k, v in (("LLM_BASE_URL", env["base_url"]),
                              ("LLM_MODEL", env["model"]), ("LLM_API_KEY", env["api_key"])) if not v]
    # Only public origin/path, never credentials or query parameters.
    parsed = urlsplit(env["base_url"])
    public_url = f"{parsed.scheme}://{parsed.hostname or ''}{parsed.path}" if parsed.scheme else ""
    return {"configured": not missing, "base_url": public_url, "model": env["model"],
            "api_key_set": bool(env["api_key"]), "missing": missing,
            "connectivity": "untested"}


def _content(data: dict) -> str:
    """Accept the usual OpenAI shape plus common variants (text completions, list content)."""
    if isinstance(data, dict) and data.get("error"):
        raise LLMError(f"provider error: {str(data['error'])[:300]}")
    ch = (data.get("choices") or [None])[0] if isinstance(data, dict) else None
    if not ch:
        raise LLMError(f"no choices in response: {json.dumps(data)[:300]}")
    msg = ch.get("message") or {}
    c = msg.get("content") if msg else ch.get("text")
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
    if not c or not str(c).strip():
        raise LLMError("empty completion")
    return str(c).strip()


class OpenAICompatibleProvider(Provider):
    def __init__(self, timeout: float, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, transport=None, retries: int = 3, sleep=time.sleep):
        env = llm_env()
        self.base = (base_url or env["base_url"]).rstrip("/")
        self.key = api_key if api_key is not None else env["api_key"]
        self.model = model or env["model"]
        missing = [n for n, v in (("LLM_BASE_URL", self.base), ("LLM_MODEL", self.model)) if not v]
        if missing:
            raise LLMError(f"missing {', '.join(missing)} (set in backend/.env or the shell)")
        if self.base.endswith("/chat/completions"):
            self.base = self.base[: -len("/chat/completions")]
        parsed = urlsplit(self.base)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise LLMError("LLM_BASE_URL must be an http(s) API base without credentials, query or fragment")
        self.timeout = timeout
        self.name = f"openai-compatible:{self.model}"
        self._transport = transport or self._http
        self.retries, self._sleep = retries, sleep

    def _http(self, url: str, body: bytes, headers: dict) -> dict:
        req = urllib.request.Request(url, data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise LLMError(f"HTTP {e.code} from {url}: {detail}", e.code) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"cannot reach provider: {getattr(e, 'reason', str(e))}", 408) from None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise LLMError("provider returned non-JSON response; verify the API base URL") from None

    def chat(self, system: str, user: str, max_tokens: int = 400) -> str:
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": "cgtrs/3", "HTTP-Referer": "http://localhost", "X-OpenRouter-Title": "CGTRS"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        body = json.dumps({"model": self.model, "temperature": 0, "max_tokens": max_tokens, "stream": False,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode()
        url, last = f"{self.base}/chat/completions", None
        for attempt in range(self.retries + 1):
            try:
                return _content(self._transport(url, body, headers))
            except LLMError as e:  # busy routers (Cortensor/OpenRouter) answer 429/503: back off and retry
                if self.key:
                    e = LLMError(str(e).replace(self.key, "[redacted]"), e.status)
                last = e
                if e.status not in RETRY_STATUS or attempt == self.retries:
                    raise e from None
                self._sleep(min(2 ** attempt * 2, 15))
        raise last  # pragma: no cover

    def narrate(self, record: dict) -> str:
        return self.chat(SYSTEM, json.dumps(record, default=str)[:12000])


class FailedProvider(Provider):
    """Configured LLM could not be built: every narration surfaces the reason."""

    def __init__(self, reason: str):
        self.name, self.reason = "llm-unavailable", reason

    def narrate(self, record: dict) -> str:
        raise LLMError(self.reason)


def get_provider(kind: str, timeout: float) -> Provider:
    if kind == "offline":
        return OfflineProvider()
    env = llm_env()
    if kind == "auto" and not any(env.values()):
        return OfflineProvider()
    try:
        return OpenAICompatibleProvider(timeout)
    except LLMError as e:
        return FailedProvider(str(e))


def narrate_checked(provider: Provider, record: dict, valid_ids: set[str]) -> dict:
    """Narration plus citation validation. Errors are surfaced, never replaced by fake text."""
    try:
        text = provider.narrate(record)
    except Exception as e:  # noqa: BLE001
        return {"provider": provider.name, "text": None, "error": f"{type(e).__name__}: {str(e)[:300]}",
                "unsupported_ids": [], "valid": False}
    bad = sorted(set(EV_RE.findall(text)) - valid_ids)
    return {"provider": provider.name, "text": text, "error": None, "unsupported_ids": bad, "valid": not bad}
