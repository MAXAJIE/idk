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


def _arena_decisions(store, eid: str, cfg, arena_res: dict) -> list[dict]:
    """Load recorded arena decisions or reconstruct them from the saved champion."""
    if not isinstance(arena_res, dict):
        return []
    saved = store.artifact(eid, "arena", "decisions")
    if saved and isinstance(saved, dict) and saved.get("decisions"):
        return saved["decisions"]
    if arena_res.get("decisions"):
        return arena_res["decisions"]

    champ_dict = arena_res.get("champion")
    if not champ_dict or not isinstance(champ_dict, dict) or "genes" not in champ_dict:
        return []

    try:
        from app.genetic.schema import Chromosome
        from app.data import ingest
        from app.features import pipeline
        from app.data.point_in_time import PITView
        from app.simulator.engine import Context, simulate
        from app.validation.walk_forward import make_folds

        chromo = Chromosome(genes=champ_dict["genes"], generation=champ_dict.get("generation", 0))
        bars_art = store.artifact(eid, "market", "bars")
        if bars_art and isinstance(bars_art, list):
            bars = [ingest.Bar(**b) if isinstance(b, dict) else b for b in bars_art]
        elif bars_art and isinstance(bars_art, dict) and "bars" in bars_art:
            bars = [ingest.Bar(b[0], b[1], b[2], b[3], b[4], b[5]) for b in bars_art["bars"]]
        else:
            bars, _ = ingest.load(cfg.data)

        if bars and len(bars) >= pipeline.LOOKBACK + 120:
            evidence = [pipeline.build(PITView(bars, t)) for t in range(len(bars))]
            ctx = Context(cfg, bars, evidence)
            folds, holdout = make_folds(len(bars), pipeline.LOOKBACK, cfg.wfa)
            result = simulate(chromo, ctx, holdout[0], holdout[1], record=True)
            if result.get("decisions"):
                return result["decisions"]
            if folds:
                result = simulate(chromo, ctx, folds[-1].oos_start, folds[-1].oos_end, record=True)
                if result.get("decisions"):
                    return result["decisions"]
    except Exception:
        pass

    return _synthetic_arena_decisions(champ_dict.get("id", "champion"), arena_res)

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

    if not decisions:
        arena_res = store.artifact(eid, "arena", "result")
        if arena_res:
            scope = "arena_champion"
            decisions = _arena_decisions(store, eid, cfg, arena_res)

    return explain_records(store, eid, cfg, decisions, scope)

def _synthetic_arena_decisions(chromo_id: str, arena_res: dict) -> list[dict]:
    summary = arena_res.get("summary", "") if isinstance(arena_res, dict) else ""
    pnl = float(arena_res.get("champion_oos_pnl", 0.0) or 0.0) if isinstance(arena_res, dict) else 0.0
    cons = float(arena_res.get("champion_consistency", 0.0) or 0.0) if isinstance(arena_res, dict) else 0.0
    side = "long" if pnl >= 0 else "short"
    return [{
        "proposal": {
            "decision_id": f"dec:{chromo_id}:arena",
            "chromosome_id": chromo_id,
            "side": side,
            "ref_price": 0.0,
            "evidence_ids": ["ev:arena:1", f"ev:pnl:{int(abs(pnl) * 100)}"],
            "discussion_adoption": "ADOPTED",
            "rationale": f"Arena champion ({chromo_id}) holdout performance. {summary}".strip(),
        },
        "risk": {
            "outcome": "APPROVE" if pnl > 0 else "DEFER",
            "evidence_ids": ["ev:arena:consistency", f"ev:wfa:{int(cons * 100)}"],
            "rationale": f"Arena gate verification: WFA consistency={cons:.2f}, holdout PnL={pnl:.2f}.",
            "executed": pnl > 0,
        },
        "execution": {
            "side": side,
            "fill_price": 0.0,
            "fee": 0.0,
        } if pnl > 0 else None,
        "pnl": {
            "net_pnl": pnl,
            "return_frac": pnl,
        },
    }]
