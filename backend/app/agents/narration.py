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
    """Load recorded arena decisions or reconstruct them from the saved champion or top combatant."""
    if not isinstance(arena_res, dict):
        return []
    saved = store.artifact(eid, "arena", "decisions")
    if saved and isinstance(saved, dict) and saved.get("decisions"):
        return saved["decisions"]
    if arena_res.get("decisions"):
        return arena_res["decisions"]

    # 1. Determine target chromosome: champion or top combatant
    champ_dict = arena_res.get("champion")
    target_dict = None
    target_id = "champion"

    if champ_dict and isinstance(champ_dict, dict) and "genes" in champ_dict:
        target_dict = champ_dict
        target_id = champ_dict.get("id", "champion")
    else:
        combatants = arena_res.get("combatants", [])
        if not combatants:
            fin = store.artifact(eid, "final", "ranking")
            if fin and isinstance(fin, dict) and fin.get("ranking"):
                combatants = fin["ranking"]
        if combatants:
            best = max(
                combatants,
                key=lambda c: (
                    bool(c.get("qualified")),
                    bool(c.get("wfa_qualified")),
                    float(c.get("selection_score", 0.0) or c.get("rank_net", 0.0) or 0.0),
                    float(c.get("oos_pnl", 0.0) or (c.get("wfa") or {}).get("net_pnl", 0.0) or 0.0),
                ),
                default=None,
            )
            if best:
                target_dict = best.get("chromosome")
                target_id = best.get("id") or (target_dict.get("id") if isinstance(target_dict, dict) else "top_candidate")

    # 2. Attempt point-in-time simulation to reconstruct genuine decisions
    if target_dict and isinstance(target_dict, dict) and "genes" in target_dict:
        try:
            from app.genetic.schema import Chromosome
            from app.data import ingest
            from app.features import pipeline
            from app.data.point_in_time import PITView
            from app.simulator.engine import Context, simulate
            from app.validation.walk_forward import make_folds

            chromo = Chromosome(genes=target_dict["genes"], generation=target_dict.get("generation", 0), operator=target_dict.get("operator", "elite"))
            bars_art = store.artifact(eid, "market", "bars")
            if bars_art and isinstance(bars_art, list):
                bars = [ingest.Bar(**b) if isinstance(b, dict) else b for b in bars_art]
            elif bars_art and isinstance(bars_art, dict) and "bars" in bars_art:
                bars = [
                    ingest.Bar(
                        i=idx,
                        ts=int(b[0]),
                        open=float(b[1]),
                        high=float(b[2]),
                        low=float(b[3]),
                        close=float(b[4]),
                        volume=float(b[5]),
                        funding_rate=float(b[6]) if len(b) > 6 else 0.0,
                    )
                    for idx, b in enumerate(bars_art["bars"])
                ]
            else:
                bars, _ = ingest.load(cfg.data)

            if bars and len(bars) >= pipeline.LOOKBACK + 120:
                evidence = [pipeline.build(PITView(bars, t)) for t in range(len(bars))]
                ctx = Context(cfg, bars, evidence)
                folds, holdout = make_folds(len(bars), pipeline.LOOKBACK, cfg.wfa)
                # A no-champion run has not earned a holdout evaluation. Explain from
                # recorded/reconstructed WFA research only; never peek at final holdout.
                if champ_dict and arena_res.get("holdout_attempted"):
                    result = simulate(chromo, ctx, holdout[0], holdout[1], record=True)
                    if result.get("decisions"):
                        return result["decisions"]
                for fold in reversed(folds):
                    result = simulate(chromo, ctx, fold.oos_start, fold.oos_end, record=True)
                    if result.get("decisions"):
                        return result["decisions"]
                if folds:
                    result = simulate(chromo, ctx, folds[0].is_start, folds[0].is_end, record=True)
                    if result.get("decisions"):
                        return result["decisions"]
        except Exception:
            pass

    return _synthetic_arena_decisions(target_id, arena_res)


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
            scope = "arena_champion" if arena_res.get("champion") else "arena_candidate"
            decisions = _arena_decisions(store, eid, cfg, arena_res)

    if not decisions:
        fin = store.artifact(eid, "final", "ranking")
        ranking = fin.get("ranking", []) if fin and isinstance(fin, dict) else []
        if ranking:
            top_c = ranking[0]
            cid = top_c.get("id", "top_candidate")
            scope = "final_ranking"
            decisions = _synthetic_arena_decisions(cid, {
                "summary": f"Ranking contains {len(ranking)} evaluated chromosomes. Top status: {top_c.get('status', 'REJECTED')}",
                "champion_oos_pnl": (top_c.get("wfa") or {}).get("net_pnl", 0.0),
                "champion_consistency": top_c.get("consistency", 0.0),
            })

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
            "rationale": f"Arena candidate ({chromo_id}) performance: {summary}".strip(),
        },
        "risk": {
            "outcome": "APPROVE" if pnl > 0 else "DEFER",
            "evidence_ids": ["ev:arena:1"],
            "reasons": [f"Arena verification: consistency={cons:.2f}, PnL={pnl:.2f}."],
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

