"""Explain completed records independently of qualification, without changing them."""
from __future__ import annotations

import time

from app.agents.provider import get_provider, narrate_checked


def explain_records(store, eid, cfg, decisions, scope, provider=None):
    if cfg.llm.narrate == "none":
        return []
    provider = provider or get_provider(cfg.llm.provider, cfg.llm.timeout_s)
    rows = []
    # Include no_trade / DEFER / rejected decisions: these also need explanations.
    for d in decisions[-cfg.llm.max_narrated_decisions:] if cfg.llm.max_narrated_decisions > 0 else []:
        ids = set(d["proposal"].get("evidence_ids", []))
        n = {"decision_id": d["proposal"]["decision_id"], "scope": scope,
             **narrate_checked(provider, d, ids)}
        rows.append(n)
    if rows:
        store.put(eid, "narration", str(time.time_ns()), {"scope": scope, "narration": rows})
    store.append(eid, "narration", {"id": None, "scope": scope, "provider": provider.name,
                                   "n": len(rows), "ok": sum(bool(n["text"]) for n in rows),
                                   "errors": sorted({n["error"] for n in rows if n["error"]}) or
                                   (["No persisted decisions available; use llm-test to test connectivity."] if not rows else [])})
    return rows


def explain_experiment(store, eid, cfg):
    """Re-explain existing immutable records with current local LLM configuration."""
    holds = store.artifacts(eid, "holdout")
    candidates = store.artifacts(eid, "candidate")
    source = next((h for h in holds if h.get("decisions")), None)
    scope = "holdout"
    if source is None:
        eligible = [c for c in candidates if c.get("decisions")]
        source = max(eligible, key=lambda c: c.get("oos", {}).get("scorecard", {}).get("net_pnl", 0), default=None)
        scope = "oos_research_not_qualified"
    if source is not None:
        decisions = source["decisions"]
    else:
        decisions = [e["payload"] for e in store.events(eid, limit=100000)
                     if e["type"] in ("decision", "order") and "proposal" in e["payload"]]
        scope = "forward_test"
    return explain_records(store, eid, cfg, decisions, scope)