"""Experiment runner: data -> features -> per-fold IS evolution -> freeze -> OOS
-> Judge -> WFA gate -> holdout. All outputs persisted as immutable events/artifacts."""
from __future__ import annotations

import hashlib
import json
import random
import time

from app.agents import discussion as disc_mod
from app.agents.judge import review
from app.agents.provider import get_provider, narrate_checked
from app.agents.narration import explain_experiment
from app.config import ExperimentConfig
from app.data import ingest, quality
from app.data.point_in_time import PITView
from app.features import pipeline
from app.genetic import population
from app.orchestration.state_machine import StateMachine
from app.persistence.db import Store
from app.risk.scorecard import fitness, gates
from app.simulator.engine import Context, simulate
from app.validation import robustness
from app.validation.walk_forward import make_folds

WARMUP = pipeline.LOOKBACK


def _slim(res: dict) -> dict:
    return {k: v for k, v in res.items() if k != "decisions"}


def run(cfg: ExperimentConfig, store: Store) -> str:
    eid = f"exp_{cfg.fingerprint()}_{int(time.time())}"
    store.create_experiment(eid, cfg.name, cfg.model_dump())
    emit = lambda t, p: store.append(eid, t, p)  # noqa: E731
    print(f"experiment {eid}", flush=True)

    def on_change(old, new, info):
        store.set_state(eid, new)
        emit("state", {"from": old, "to": new, **info})
        print(f"{new}" + (f" fold={info['fold']}" if 'fold' in info else "") +
              (f" reason={info['reason']}" if 'reason' in info else ""), flush=True)

    sm = StateMachine(on_change)
    emit("experiment_created", {"config_fingerprint": cfg.fingerprint(), "risk_version": cfg.risk.version,
                                "feature_version": cfg.feature_version, "schema_version": cfg.schema_version,
                                "ga_seed": cfg.ga.seed, "operator_version": cfg.ga.operator_version,
                                "llm_provider": cfg.llm.provider, "mode": "BACKTEST"})

    sm.to("DATA_VALIDATION")
    try:
        bars, snap = ingest.load(cfg.data)
    except Exception as e:  # noqa: BLE001
        sm.to("ABORTED_INTEGRITY", reason=f"load failed: {e}")
        return eid
    errs = quality.check(bars, cfg.data.timeframe_minutes, cfg.data.max_gap_bars)
    store.put(eid, "dataset", "snapshot", snap.to_dict() | {"quality_errors": errs})
    emit("dataset", snap.to_dict() | {"quality_errors": errs})
    if errs:
        sm.to("ABORTED_INTEGRITY", reason="data quality", errors=errs)
        return eid
    try:
        folds, holdout = make_folds(len(bars), WARMUP, cfg.wfa)
    except (ValueError, AssertionError) as e:
        sm.to("ABORTED_INTEGRITY", reason=str(e))
        return eid
    store.put(eid, "folds", "all", {"folds": [f.to_dict() for f in folds], "holdout": list(holdout),
                                    "embargo": cfg.wfa.embargo_bars})

    sm.to("FEATURE_BUILD")
    evidence = [pipeline.build(PITView(bars, t)) for t in range(len(bars))]
    ctx = Context(cfg, bars, evidence)
    store.put(eid, "market", "bars", {"bars": [[b.ts, b.open, b.high, b.low, b.close, b.volume] for b in bars]})
    emit("features_built", {"n": len(evidence), "version": cfg.feature_version})

    rng = random.Random(cfg.ga.seed)
    archive: list = []  # retained (RETAIN verdict) chromosomes across folds
    retained_meta: dict[str, dict] = {}
    for fold in folds:
        sm.to("IS_EVOLUTION", fold=fold.k)
        pop = population.initial(rng, cfg.ga.population)
        scored: list = []
        for gen in range(cfg.ga.generations):
            scored = []
            n_valid = 0
            t_gen = time.time()
            for c in pop:
                res = simulate(c, ctx, fold.is_start, fold.is_end)
                if res["reconcile_error"] > 1e-6:
                    sm.to("ABORTED_INTEGRITY", reason=f"ledger reconcile {res['reconcile_error']}")
                    return eid
                if res["failed_risk"]:
                    c.state = "FAILED_RISK"
                    emit("chromosome_failed_risk", {"fold": fold.k, "gen": gen, "id": c.id, **res["failed_risk"]})
                gr = gates(res["scorecard"], cfg.gates)
                f = fitness(res["scorecard"], gr)
                n_valid += (not res["failed_risk"]) and all(g["passed"] for g in gr)
                scored.append((c, f))
                store.put(eid, "chromosome", f"{fold.k}:{gen}:{c.id}",
                          c.to_dict() | {"fold": fold.k, "fitness": f, "is_scorecard": res["scorecard"]})
            scored.sort(key=lambda x: -x[1])
            fits = [x[1] for x in scored]
            emit("generation", {"fold": fold.k, "gen": gen, "best": round(scored[0][1], 4),
                                "mean": round(sum(fits) / len(fits), 4), "worst": round(fits[-1], 4),
                                "valid": n_valid, "population": len(scored),
                                "generations": cfg.ga.generations, "folds": len(folds),
                                "seconds": round(time.time() - t_gen, 3),
                                "failed_risk": sum(c.state == "FAILED_RISK" for c, _ in scored),
                                "best_id": scored[0][0].id})
            print(f"  fold={fold.k} gen={gen + 1}/{cfg.ga.generations} best={scored[0][1]:.4f} "
                  f"valid={n_valid}/{len(scored)}", flush=True)
            if gen < cfg.ga.generations - 1:
                pop, discarded = population.next_generation(rng, scored, archive, cfg.ga.population, gen + 1, cfg.ga)
                emit("mutation", {"fold": fold.k, "gen": gen + 1, "operator": cfg.ga.operator_version,
                                  "offspring": len(pop), "discarded_invalid": discarded,
                                  "changes": {c.id: c.changed for c in pop}})

        sm.to("CANDIDATE_FREEZE", fold=fold.k)
        frozen = population.select_diverse_elite([x for x in scored if x[0].state != "FAILED_RISK"], cfg.ga.elite_size)
        for c in frozen:
            c.state = "FROZEN"
        snap_hash = hashlib.sha256(json.dumps([c.genes for c in frozen], sort_keys=True).encode()).hexdigest()[:16]
        emit("candidates_frozen", {"fold": fold.k, "ids": [c.id for c in frozen], "snapshot": snap_hash})

        sm.to("OOS_EVALUATION", fold=fold.k)
        oos_res = {c.id: simulate(c, ctx, fold.oos_start, fold.oos_end, record=True) for c in frozen}
        sm.to("JUDGE_REVIEW", fold=fold.k)
        for c in frozen:
            is_res = simulate(c, ctx, fold.is_start, fold.is_end, record=True)
            oos = oos_res[c.id]
            nb = [simulate(n, ctx, fold.is_start, fold.is_end)["scorecard"]["net_pnl"] for n in robustness.neighbours(c)]
            rob = {"retention": robustness.retention(is_res["scorecard"]["net_pnl"], nb), "neighbour_nets": nb}
            all_dec = {d["proposal"]["decision_id"]: d["proposal"] for d in is_res["decisions"] + oos["decisions"]}
            valid_ids = {i for e in (evidence[d["t"]] for d in is_res["decisions"] + oos["decisions"]) for i in e.items}
            packet = {"trades": is_res["trades"] + oos["trades"],
                      "exit_range_atr": is_res["exit_range_atr"] | oos["exit_range_atr"],
                      "decisions": all_dec, "valid_evidence": valid_ids, "robustness": rob,
                      "oos_scorecard": oos["scorecard"]}
            oos_gates = gates(oos["scorecard"], cfg.gates)
            v = review(c.id, packet, cfg.judge, cfg.gates, cfg.risk.initial_equity)
            # sequential qualification: hard gates first; Judge may only remove
            if oos["failed_risk"]:
                status = "FAILED_RISK"
            elif oos["scorecard"]["n_trades"] < cfg.gates.min_trades:
                status = "INSUFFICIENT_EVIDENCE"
            elif not all(g["passed"] for g in oos_gates):
                status = "REJECTED"
            else:
                status = {"RETAIN": "RETAINED", "REJECT": "REJECTED", "DEFER": "INSUFFICIENT_EVIDENCE"}[v.verdict]
            c.state = status
            if status == "RETAINED":
                archive.append(c)
                retained_meta[c.id] = {"fold": fold.k, "adjusted_net": v.adjusted_net_pnl}
            store.put(eid, "candidate", f"{fold.k}:{c.id}", {
                "fold": fold.k, "chromosome": c.to_dict(), "status": status, "oos_gates": oos_gates,
                "judge": v.model_dump(), "robustness": rob, "is": _slim(is_res), "oos": _slim(oos),
                "decisions": oos["decisions"][-40:], "rewards": is_res["rewards"]})
            emit("judge_verdict", {"fold": fold.k, "id": c.id, "verdict": v.verdict, "status": status,
                                   "flags": v.flags, "rationale": v.rationale})
        sm.to("FOLD_DONE", fold=fold.k)

    sm.to("WFA_GATE")
    final = []
    for c in archive:
        k0 = retained_meta[c.id]["fold"]
        later = [simulate(c, ctx, f.oos_start, f.oos_end)["scorecard"] for f in folds if f.k > k0]
        windows = [store.artifact(eid, "candidate", f"{k0}:{c.id}")["oos"]["scorecard"]] + later
        good = [w for w in windows if w["net_return"] > 0 and w["max_drawdown"] <= cfg.gates.max_drawdown_frac]
        cons = len(good) / len(windows)
        if len(windows) < cfg.wfa.min_oos_folds:
            st = "INSUFFICIENT_EVIDENCE"
        else:
            st = "QUALIFIED" if cons >= cfg.wfa.min_consistency_frac else "REJECTED"
        final.append({"id": c.id, "fold": k0, "windows": len(windows), "consistency": round(cons, 4),
                      "status": st, "rank_net": retained_meta[c.id]["adjusted_net"], "chromosome": c.to_dict()})
    final.sort(key=lambda x: (x["status"] != "QUALIFIED", -x["rank_net"]))

    sm.to("HOLDOUT")
    provider = get_provider(cfg.llm.provider, cfg.llm.timeout_s)
    emit("llm_provider", {"requested": cfg.llm.provider, "using": provider.name,
                          "reason": getattr(provider, "reason", None)})
    for i, row in enumerate(final):
        if row["status"] != "QUALIFIED":
            continue
        c = next(a for a in archive if a.id == row["id"])
        h = simulate(c, ctx, holdout[0], holdout[1], record=True)
        row["holdout"] = h["scorecard"]
        if cfg.llm.narrate == "elite_only" and not store.artifacts(eid, "holdout"):
            narr = []
            for d in h["decisions"][-cfg.llm.max_narrated_decisions:] if cfg.llm.max_narrated_decisions > 0 else []:
                n = narrate_checked(provider, d, set(evidence[d["t"]].items))
                narr.append({"decision_id": d["proposal"]["decision_id"], **n})
            emit("narration", {"id": c.id, "provider": provider.name, "n": len(narr),
                               "ok": sum(1 for x in narr if x["text"]),
                               "errors": sorted({x["error"] for x in narr if x["error"]})[:3]})
            store.put(eid, "holdout", c.id, {"result": _slim(h), "decisions": h["decisions"][-40:], "narration": narr,
                                             "analysts": [disc_mod.aggregate(evidence[holdout[0]]).model_dump(),
                                                          disc_mod.technical(evidence[holdout[0]]).model_dump()]})
    if cfg.llm.narrate == "elite_only" and not any(r["status"] == "QUALIFIED" for r in final):
        explain_experiment(store, eid, cfg)
    store.put(eid, "final", "ranking", {"ranking": final, "holdout_window": list(holdout)})
    emit("final_ranking", {"qualified": [r["id"] for r in final if r["status"] == "QUALIFIED"], "n_archive": len(archive)})
    sm.to("COMPLETED")
    return eid
