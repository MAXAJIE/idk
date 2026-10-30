"""Live indicators + bull/bear debate on public Binance candles (read-only).

Uses only closed candles, the same feature pipeline and the same deterministic
discussion used by the trader, so the console shows what the agents would see.
"""
from __future__ import annotations

from app.agents import discussion as disc_mod
from app.api import live_market
from app.data.ingest import Bar
from app.data.point_in_time import PITView
from app.features import pipeline
from app.features.trend import atr, ema, rsi

_LLM_DEBATE_CACHE: dict[tuple[str, int, int], dict] = {}


def _series(bars: list[Bar]) -> dict:
    closes = [b.close for b in bars]
    e20, e50 = ema(closes, 20), ema(closes, 50)
    out = {"ts": [], "ema20": [], "ema50": [], "rsi14": [], "atr14": []}
    for k, b in enumerate(bars):
        out["ts"].append(b.ts)
        out["ema20"].append(round(e20[k], 6))
        out["ema50"].append(round(e50[k], 6))
        out["rsi14"].append(round(rsi(closes[: k + 1]), 2))
        out["atr14"].append(round(atr(bars[: k + 1]), 6))
    return out


def analyse(symbol: str, tf: int = 60, limit: int = 300, agents: str = "rules", rounds: int = 2, refresh: bool = False) -> dict:
    import os
    snap = live_market.snapshot(symbol, tf, limit)
    rows = [c for c in snap["candles"] if c[6]]  # closed only: no look-ahead
    bars = [Bar(i, c[0], c[1], c[2], c[3], c[4], c[5], snap["funding_rate"]) for i, c in enumerate(rows)]
    base = {"symbol": snap["symbol"], "tf_minutes": tf, "server_time": snap["server_time"],
            "candles": [c[:6] for c in rows]}
    if not bars:
        return base | {"valid": False, "warnings": ["no closed candles"], "evidence": {}, "debate": None,
                       "series": _series([])}
    eb = pipeline.build(PITView(bars, len(bars) - 1))
    debate = disc_mod.discuss(eb).model_dump() if eb.valid else None
    reports = [disc_mod.aggregate(eb).model_dump(), disc_mod.technical(eb).model_dump()]
    result = base | {"valid": eb.valid, "warnings": eb.warnings, "bar_ts": eb.ts,
                     "evidence": {v["kind"]: v["value"] for v in eb.items.values()},
                     "analysts": reports, "debate": debate, "series": _series(bars)}
    if agents == "llm" and eb.valid:
        key = (snap["symbol"], tf, eb.ts, rounds)
        if not refresh and key in _LLM_DEBATE_CACHE:
            result["agent_debate"] = _LLM_DEBATE_CACHE[key]
        else:
            from app.agents.provider import OpenAICompatibleProvider, llm_status
            status = llm_status()
            if not status["configured"]:
                result["agent_debate"] = {"status": "unavailable", "reason": "LLM provider not configured. Set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY in backend/.env."}
            else:
                try:
                    timeout = float(os.environ.get("LLM_TIMEOUT", "30"))
                    provider = OpenAICompatibleProvider(timeout=timeout, retries=1)
                    report = disc_mod.discuss_with_llm(eb, provider, rounds=rounds)
                    payload = {"status": "complete", **report.model_dump(),
                               "provider": provider.name,
                               "rounds": rounds,
                               "bar_ts": eb.ts,
                               "scope": "research display only; cannot alter trading decisions or risk gates"}
                    _LLM_DEBATE_CACHE[key] = payload
                    if len(_LLM_DEBATE_CACHE) > 128:
                        _LLM_DEBATE_CACHE.pop(next(iter(_LLM_DEBATE_CACHE)))
                    result["agent_debate"] = payload
                except Exception as exc:  # noqa: BLE001 provider output is untrusted
                    result["agent_debate"] = {"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:240]}"}
    return result
