"""Operator CLI (local terminal only — not reachable from the browser).

  python -m app.cli run                       # fetches BTCUSDT from Binance automatically
  python -m app.cli run   --config configs/default.json   # offline synthetic demo
  python -m app.cli fetch --symbol BTCUSDT --tf 60 --n 3000 --out data/btc.csv [--backend ccxt]
  python -m app.cli trade --config configs/binance.json --experiment <exp_id> [--dry-run] [--once]
  python -m app.cli serve
  python -m app.cli llm-test                  # checks LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from app.env import load_env

load_env()  # backend/.env or backend/app/.env, before anything reads os.environ

from app.config import ExperimentConfig, load_config  # noqa: E402
from app.persistence.db import Store  # noqa: E402

BACKEND = Path(os.path.abspath(__file__)).parents[1]
DB = os.environ.get("CGTRS_DB", str(BACKEND / "data/cgtrs.sqlite"))
os.environ.setdefault("CGTRS_SNAPSHOT_DIR", str(BACKEND / "data/snapshots"))


def backend_path(value):
    path = Path(value)
    return str(path if path.is_absolute() else BACKEND / path)


def prepare_trade_config(cfg: ExperimentConfig, dry_run: bool) -> tuple[ExperimentConfig, bool]:
    """Allow offline research configs only for a no-orders Binance testnet preview."""
    if cfg.data.source == "binance":
        return cfg, False
    if not dry_run:
        raise ValueError("trade requires a Binance research experiment, not synthetic/CSV data")

    # The default offline config uses a simulator-only symbol; use the matching
    # Binance market for preview prices while preserving any explicit symbol.
    symbol = "BTCUSDT" if cfg.data.symbol in {"BTC-PERP", "SIM"} else cfg.data.symbol
    data = cfg.data.model_copy(update={
        "source": "binance",
        "symbol": symbol,
        "exchange": "BINANCE_USDM",
    })
    return cfg.model_copy(update={"data": data}), True


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--config", default="configs/binance.json"); r.add_argument("--db", default=DB)
    f = sub.add_parser("fetch")
    f.add_argument("--symbol", default="BTCUSDT"); f.add_argument("--tf", type=int, default=60)
    f.add_argument("--n", type=int, default=3000); f.add_argument("--out", required=True)
    f.add_argument("--backend", choices=["rest", "ccxt"], default="rest")
    t = sub.add_parser("trade")
    t.add_argument("--config"); t.add_argument("--db", default=DB)
    t.add_argument("--experiment", default="best", help="experiment ID or 'best' for automatic best qualified chromosome (default)"); t.add_argument("--dry-run", action="store_true")
    t.add_argument("--once", action="store_true"); t.add_argument("--allow-unqualified", action="store_true")
    t.add_argument("--state"); t.add_argument("--kill-file", default="KILL")
    t.add_argument("--auto-re-evolve", action="store_true", help="enable rolling walk-forward candidate re-evolution and safe hot-swap")
    s = sub.add_parser("serve"); s.add_argument("--db", default=DB); s.add_argument("--port", type=int, default=8000)
    sub.add_parser("llm-test")
    x = sub.add_parser("experiments"); x.add_argument("--db", default=DB)
    x = sub.add_parser("explain"); x.add_argument("--db", default=DB)
    x.add_argument("--experiment", default="latest")
    tr = sub.add_parser("train", help="Mass-produce chromosomes on history data and run anti-overfitting arena combat")
    tr.add_argument("--config", default="configs/default.json")
    tr.add_argument("--db", default=DB)
    tr.add_argument("--pop-size", type=int, default=32)
    tr.add_argument("--generations", type=int, default=3)
    tr.add_argument("--min-consistency", type=float, default=0.6)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--auto", action="store_true", help="Auto-train continuously across adaptive rounds until a qualified champion is found, then stop and analyze")
    tr.add_argument("--max-rounds", type=int, default=10, help="Max auto-train rounds when --auto is specified")
    a = ap.parse_args()
    if hasattr(a, "db"):
        a.db = backend_path(a.db)
        print(f"database: {a.db}", flush=True)
    if a.cmd == "run":
        from app.orchestration.runner import run
        cfg = load_config(backend_path(a.config))
        if cfg.data.source == "binance":
            print(f"fetching {cfg.data.n_bars} x {cfg.data.timeframe_minutes}m {cfg.data.symbol} from Binance ...", flush=True)
        eid = run(cfg, Store(a.db))
        st = next((e["state"] for e in Store(a.db, readonly=True).experiments() if e["id"] == eid), "?")
        print(eid, st)
        print(f'Inspect: python -m app.cli experiments --db "{a.db}"')
        if st != "COMPLETED":
            raise SystemExit(1)
    elif a.cmd == "fetch":
        from app.data import binance
        bars = binance.fetch(a.symbol, a.tf, a.n, a.backend)
        out = backend_path(a.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        binance.write_csv(bars, out)
        print(f"{len(bars)} closed bars -> {out}")
    elif a.cmd == "trade":
        from app.live.runner import LiveTrader, pick_chromosome, resolve_experiment, find_best_chromosome
        from app.live.testnet import TestnetClient
        if a.allow_unqualified and not a.dry_run:
            ap.error("--allow-unqualified requires --dry-run; unqualified strategies may not place orders")
        store = Store(a.db)
        if a.experiment in ("best", "auto", "latest"):
            chromo, eid, pick_meta = find_best_chromosome(store, a.experiment, a.allow_unqualified)
            print(f"AUTO-SELECTED chromosome {chromo.id} from {eid}: {pick_meta}", flush=True)
        else:
            eid = resolve_experiment(store, a.experiment)
            chromo = pick_chromosome(store, eid, a.allow_unqualified)
        exp = next(e for e in store.experiments() if e["id"] == eid)
        cfg = load_config(backend_path(a.config)) if a.config else ExperimentConfig.model_validate(exp["config"])
        if a.auto_re_evolve:
            cfg = cfg.model_copy(update={"live": cfg.live.model_copy(update={"auto_re_evolve": True})})
        try:
            cfg, previewing_non_binance_research = prepare_trade_config(cfg, a.dry_run)
        except ValueError as e:
            ap.error(str(e))
        if previewing_non_binance_research:
            original_source = exp["config"].get("data", {}).get("source", "unknown")
            print(
                f"warning: dry-run only; strategy was researched with {original_source} data, "
                f"while preview prices come from Binance testnet ({cfg.data.symbol}); no orders will be sent",
                flush=True,
            )
        state = backend_path(a.state or ("data/dry_state.json" if a.dry_run else "data/live_state.json"))
        print(f"source experiment: {eid}; mode: {'DRY_RUN (no orders)' if a.dry_run else 'BINANCE_TESTNET'}", flush=True)
        lt = LiveTrader(cfg, chromo, store,
                        TestnetClient(cfg.live.base_url), state, a.dry_run, backend_path(a.kill_file), eid)
        print("live experiment", lt.eid)
        print(lt.loop(1 if a.once else None))
    elif a.cmd == "llm-test":
        from app.agents.provider import LLMError, OpenAICompatibleProvider, llm_env
        env = llm_env()
        print(f"LLM_BASE_URL={env['base_url'] or '(missing)'}  LLM_MODEL={env['model'] or '(missing)'}  "
              f"LLM_API_KEY={'set (' + str(len(env['api_key'])) + ' chars)' if env['api_key'] else '(missing)'}")
        try:
            p = OpenAICompatibleProvider(30.0)
            print("reply:", p.chat("Reply with exactly: OK", "ping", max_tokens=128))
        except LLMError as e:
            raise SystemExit(f"LLM FAILED: {e}")
    elif a.cmd == "experiments":
        store = Store(a.db)
        xs = store.experiments()
        if not xs:
            print("No experiments in this database. Run python -m app.cli run first.")
        for e in xs:
            fin = store.artifact(e["id"], "final", "ranking")
            rows = fin["ranking"] if fin else []
            print(f"{e['id']}  {e['state']}  ranking={'yes' if fin is not None else 'no'} "
                  f"qualified={sum(r['status'] == 'QUALIFIED' for r in rows)}  name={e['name']}")
            states = [v["payload"] for v in store.events(e["id"], limit=100000) if v["type"] == "state"]
            if states and states[-1].get("reason"):
                print("  reason:", states[-1]["reason"])
    elif a.cmd == "train":
        import time
        from app.data import ingest
        from app.learning.arena import train_and_combat, auto_train_until_survivor, analyze_arena_result
        cfg = load_config(backend_path(a.config))
        cfg = cfg.model_copy(update={"ga": cfg.ga.model_copy(update={"population": a.pop_size, "generations": a.generations})})
        bars, snap = ingest.load(cfg.data)
        os.makedirs(os.path.dirname(a.db), exist_ok=True)
        st = Store(a.db)
        eid = f"arena_{int(time.time() * 1000)}"
        st.create_experiment(eid, f"arena:{cfg.data.symbol}", cfg.model_dump())
        st.put(eid, "arena", "plan", {"auto": a.auto, "max_rounds": a.max_rounds if a.auto else 1,
                                       "population": a.pop_size, "generations": a.generations,
                                       "mode": "arena"})
        st.set_state(eid, "RUNNING")
        st.append(eid, "state", {"to": "ARENA_START", "reason": "Historical arena training started", "fold": 0})

        def emit_training_event(etype, payload):
            st.append(eid, etype, payload)

        print(f"Loaded {len(bars)} bars. Starting arena training (pop={a.pop_size}, gens={a.generations})...")
        try:
            if a.auto:
                res, rounds = auto_train_until_survivor(
                    bars, cfg, pop_size=a.pop_size, generations=a.generations,
                    min_consistency=a.min_consistency, seed=a.seed, max_rounds=a.max_rounds,
                    progress=emit_training_event,
                )
            else:
                res = train_and_combat(bars, cfg, pop_size=a.pop_size, generations=a.generations,
                                       min_consistency=a.min_consistency, seed=a.seed,
                                       progress=emit_training_event)
                rounds = 1
            print(analyze_arena_result(res, cfg, round_num=rounds))
            st.put(eid, "arena", "result", res.to_dict())
            if res.decisions:
                st.put(eid, "arena", "decisions", {"decisions": res.decisions})

            # Save combined WFA artifacts so arena training also has full WFA ranking & inspectability
            from app.validation.walk_forward import make_folds
            from app.features import pipeline
            try:
                folds, holdout = make_folds(len(bars), pipeline.LOOKBACK, cfg.wfa)
                st.put(eid, "dataset", "snapshot", snap.to_dict())
                st.put(eid, "market", "bars", {"bars": [[b.ts, b.open, b.high, b.low, b.close, b.volume] for b in bars]})
                st.put(eid, "folds", "all", {"folds": [f.to_dict() for f in folds], "holdout": list(holdout), "embargo": cfg.wfa.embargo_bars})
            except Exception:
                holdout = [0, len(bars)]

            final_ranking = []
            for comb in res.combatants:
                final_ranking.append({
                    "id": comb.chromosome.id,
                    "fold": 0,
                    "windows": comb.total_oos_folds,
                    "consistency": comb.consistency,
                    "status": "QUALIFIED" if comb.qualified else ("WFA_QUALIFIED" if comb.wfa_qualified else "REJECTED"),
                    "rank_net": comb.selection_score,
                    "chromosome": comb.chromosome.to_dict(),
                    "wfa": {
                        "net_pnl": comb.oos_pnl,
                        "net_return": comb.oos_return,
                        "max_drawdown": comb.oos_drawdown,
                        "n_trades": comb.oos_trades,
                        "win_rate": comb.oos_win_rate,
                    },
                    "holdout": ({
                        "net_pnl": comb.holdout_pnl,
                        "max_drawdown": comb.holdout_drawdown,
                        "n_trades": comb.holdout_trades,
                        "win_rate": comb.holdout_win_rate,
                        "passed": comb.holdout_passed,
                    } if comb.holdout_pnl is not None else None),
                })
            final_ranking.sort(key=lambda x: (x["status"] != "QUALIFIED", -x["rank_net"] * (0.5 + 0.5 * x["consistency"])))
            st.put(eid, "final", "ranking", {"ranking": final_ranking, "holdout_window": list(holdout)})

            for comb in res.combatants:
                st.put(eid, "candidate", f"0:{comb.chromosome.id}", {
                    "fold": 0,
                    "chromosome": comb.chromosome.to_dict(),
                    "status": "QUALIFIED" if comb.qualified else ("WFA_QUALIFIED" if comb.wfa_qualified else "REJECTED"),
                    "consistency": comb.consistency,
                    "robustness": {"retention": comb.neighbour_retention},
                    "oos": {"scorecard": {"net_pnl": comb.oos_pnl, "net_return": comb.oos_return, "max_drawdown": comb.oos_drawdown, "n_trades": comb.oos_trades, "win_rate": comb.oos_win_rate}},
                })

            if res.champion and res.holdout_attempted:
                st.put(eid, "holdout", res.champion.id, {
                    "result": {"scorecard": {"net_pnl": res.champion_oos_pnl}},
                    "decisions": (res.decisions or [])[-40:],
                    "narration": [],
                })

            st.append(eid, "final_ranking", {
                "qualified": [c.chromosome.id for c in res.combatants if c.qualified],
                "wfa_qualified": [c.chromosome.id for c in res.combatants if c.wfa_qualified],
                "n_combatants": len(res.combatants),
                "champion": res.champion.id if res.champion else None,
            })

            st.set_state(eid, "COMPLETED")
            print(f"Saved arena results to experiment {eid} in database {a.db}")
        except Exception as exc:
            st.append(eid, "state", {"to": "ARENA_FAILED", "reason": f"{type(exc).__name__}: {exc}"})
            st.set_state(eid, "FAILED")
            raise
    elif a.cmd == "explain":
        from app.live.runner import resolve_experiment
        from app.agents.narration import explain_experiment
        store = Store(a.db)
        eid = resolve_experiment(store, a.experiment)
        exp = next(e for e in store.experiments() if e["id"] == eid)
        cfg = ExperimentConfig.model_validate(exp["config"])
        cfg = cfg.model_copy(update={"llm": cfg.llm.model_copy(update={"provider": "openai_compatible", "narrate": "elite_only"})})
        rows = explain_experiment(store, eid, cfg)
        for n in rows:
            print(n["decision_id"], n["text"] or n["error"])
        if any(n.get("error") for n in rows):
            raise SystemExit("LLM explanation incomplete; run python -m app.cli llm-test for diagnostics.")
        if not rows:
            print(f"No persisted decisions available to explain in experiment {eid}.")
    else:
        import uvicorn
        from app.api.read_only import create_app
        Store(a.db)
        print(f"Console: http://127.0.0.1:{a.port}/  (tabs: Training / Live testnet / Debate & indicators)", flush=True)
        print("Do not use the old frontend dev server on :5173 - it is the legacy spectator UI.", flush=True)
        uvicorn.run(create_app(a.db), host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
